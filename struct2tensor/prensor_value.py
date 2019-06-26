# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A concrete prensor tree.

PrensorValue represents a tree where all the nodes are represented as ndarrays,
instead of tensors.

prensor = ...
with tf.Session() as sess:
  prensor_value = materialize(prensor, sess)

"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import numpy as np
import tensorflow as tf

from typing import FrozenSet, Optional, Mapping, Sequence, Text, Union
from struct2tensor import path
from struct2tensor import prensor


def materialize(pren,
                sess,
                feed_dict=None,
                options=None,
                run_metadata=None):
  """Convert a prensor to a prensor value."""
  tensor_map = _get_tensor_map(pren, sess, feed_dict, options, run_metadata)
  return _map_prensor(pren, tensor_map)


class RootNodeValue(object):
  """The value of the root."""

  def __init__(self, size):
    """Creates a root node.

    Args:
      size: how many root objects there are.
    """
    self._size = size

  @property
  def size(self):
    return self._size

  @property
  def is_repeated(self):
    return True

  def schema_string(self):
    return "repeated"

  def data_string(self):
    return "size: {}".format(self._size)

  def __str__(self):
    return "RootNode"


class ChildNodeValue(object):
  """The value of an intermediate node."""

  def __init__(self, parent_index, is_repeated):
    """Creates a child node.

    Args:
      parent_index: a 1-D int64 ndarray where parent_index[i] represents the
        parent index of the ith child.
      is_repeated: a bool indicating if there can be more than one child per
        parent.
    """
    self._parent_index = parent_index
    self._is_repeated = is_repeated

  @property
  def size(self):
    """Returns the size, as if this was the root prensor.

    Returns:
      A 1-D ndarray of size 1.
    """
    return tf.shape(self.parent_index, out_type=tf.int64)

  @property
  def parent_index(self):
    return self._parent_index

  @property
  def is_repeated(self):
    return self._is_repeated

  def schema_string(self):
    return "repeated" if self.is_repeated else "optional"

  def data_string(self):
    return "parent_index: {}".format(self._parent_index)

  def __str__(self):
    return "ChildNode {} {}".format(self.schema_string(), self.data_string())


class LeafNodeValue(object):
  """The value of a leaf node."""

  def __init__(self, parent_index, values,
               is_repeated):
    """Creates a leaf node.

    Args:
      parent_index: a 1-D int64 ndarray where parent_index[i] represents the
        parent index of values[i]
      values: a 1-D ndarray of equal length to parent_index.
      is_repeated: a bool indicating if there can be more than one child per
        parent.
    """
    self._parent_index = parent_index
    self._values = values
    self._is_repeated = is_repeated

  @property
  def parent_index(self):
    return self._parent_index

  @property
  def is_repeated(self):
    return self._is_repeated

  @property
  def values(self):
    return self._values

  def data_string(self):
    return "parent_index: {} values: {}".format(self._parent_index,
                                                self._values)

  def schema_string(self):
    return u"{} {}".format("repeated" if self.is_repeated else "optional",
                           str(self.values.dtype))

  def __str__(self):
    return "{} {}".format("repeated" if self.is_repeated else "optional",
                          str(self.values.dtype))


NodeValue = Union[RootNodeValue, ChildNodeValue, LeafNodeValue]  # pylint: disable=invalid-name


class PrensorValue(object):
  """A tree of NodeValue objects."""

  def __init__(self, node,
               children):
    """Construct a PrensorValue.

    Do not call directly, instead call materialize(...) below.

    Args:
      node: the NodeValue of the root.
      children: a map from edge to subtree.
    """
    self._node = node
    self._children = children

  # TODO(martinz): This could be Value.
  @property
  def node(self):
    """The node of the root of the subtree."""
    return self._node

  def get_child(self, field_name):
    """Gets the child at field_name."""
    return self._children.get(field_name)

  def is_leaf(self):
    """True iff the node value is a LeafNodeValue."""
    return isinstance(self._node, LeafNodeValue)

  def get_child_or_error(self, field_name):
    """Gets the child at field_name."""
    result = self._children.get(field_name)
    if result is not None:
      return result
    raise ValueError("Field not found: {}".format(str(field_name)))

  def get_descendant(self, p):
    """Finds the descendant at the path."""
    result = self
    for field_name in p.field_list:
      result = result.get_child(field_name)
      if result is None:
        return None
    return result

  def get_descendant_or_error(self, p):
    """Finds the descendant at the path."""
    result = self.get_descendant(p)
    if result is None:
      raise ValueError("Missing path: {}".format(str(p)))
    return result

  def get_children(self):
    """A map from field name to subtree."""
    return self._children

  def get_descendants(self):
    """A map from paths to all subtrees."""
    result = {path.Path([]): self}
    for k, v in self._children.items():
      subtree_descendants = v.get_descendants()
      for k2, v2 in subtree_descendants.items():
        result[path.Path([k]).concat(k2)] = v2
    return result

  def field_names(self):
    """Returns the field names of the children."""
    return frozenset(self._children.keys())

  def _string_helper(self, field_name):  # pylint: disable=g-ambiguous-str-annotation
    """Helper for __str__ that outputs a list of lines."""
    result = [
        "{} {} {}".format(self.node.schema_string(), str(field_name),
                          self.node.data_string())
    ]
    for k, v in self._children.items():
      recursive = v._string_helper(k)  # pylint: disable=protected-access
      result.extend(["  {}".format(x) for x in recursive])
    return result

  def _schema_string_helper(self, field_name):  # pylint: disable=g-ambiguous-str-annotation
    """Helper for __str__ that outputs a list of lines."""
    result = [u"{} {}".format(self.node.schema_string(), str(field_name))]
    for k, v in self._children.items():
      recursive = v._string_helper(k)  # pylint: disable=protected-access
      result.extend([u"  {}".format(x) for x in recursive])
    return result

  def schema_string(self):
    """Returns a string representing the schema of the Prensor."""
    return u"\n".join(self._schema_string_helper(""))

  def __str__(self):
    """Returns a string representing the schema of the Prensor."""
    return "\n".join(self._string_helper(""))


def _get_tensors(tree):
  """Get all the tensors in all the tensor nodes in the tree."""
  node = tree.node
  result = []
  if isinstance(node, prensor.LeafNodeTensor):
    result.append(node.parent_index)
    result.append(node.values)
  elif isinstance(node, prensor.ChildNodeTensor):
    result.append(node.parent_index)
  elif isinstance(node, prensor.RootNodeTensor):
    result.append(node.size)
  for _, child in tree.get_children().items():
    result.extend(_get_tensors(child))
  return result


# Maps tensors ids to np.ndarray
_TensorMap = Mapping[int, np.ndarray]


def _get_tensor_map(t, sess, feed_dict, options,
                    run_metadata):
  tensor_list = _get_tensors(t)
  numpy_list = sess.run(
      tensor_list,
      feed_dict=feed_dict,
      options=options,
      run_metadata=run_metadata)
  return {id(t): n for t, n in zip(tensor_list, numpy_list)}


def _map_node_tensor(node,
                     tensor_map):
  if isinstance(node, prensor.LeafNodeTensor):
    return LeafNodeValue(tensor_map[id(node.parent_index)],
                         tensor_map[id(node.values)], node.is_repeated)
  elif isinstance(node, prensor.ChildNodeTensor):
    return ChildNodeValue(tensor_map[id(node.parent_index)], node.is_repeated)
  else:
    # isinstance(node, RootNodeTensor)
    return RootNodeValue(tensor_map[id(node.size)])


def _map_prensor(pren, tensor_map):
  return PrensorValue(
      _map_node_tensor(pren.node, tensor_map),
      {k: _map_prensor(v, tensor_map) for k, v in pren.get_children().items()})
