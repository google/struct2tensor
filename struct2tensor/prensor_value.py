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
assert isinstance(prensor, struct2tensor.Prensor)
with tf.Session() as sess:
  prensor_value = sess.run(prensor)
  assert isinstance(prensor_value, struct2tensor.PrensorValue)

"""

import collections
from typing import FrozenSet, Iterator, Mapping, Optional, Sequence, Union

import numpy as np
from struct2tensor import path
from struct2tensor import prensor
import tensorflow as tf

from tensorflow.python.client import session as session_lib  # pylint: disable=g-direct-tensorflow-import


class RootNodeValue(object):
  """The value of the root."""

  __slots__ = ["_size"]

  def __init__(self, size: np.int64):
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

  __slots__ = ["_parent_index", "_is_repeated"]

  def __init__(self, parent_index: np.ndarray, is_repeated: bool):
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

  def schema_string(self) -> str:
    return "repeated" if self.is_repeated else "optional"

  def data_string(self):
    return "parent_index: {}".format(self._parent_index)

  def __str__(self):
    return "ChildNode {} {}".format(self.schema_string(), self.data_string())


class LeafNodeValue(object):
  """The value of a leaf node."""

  __slots__ = ["_parent_index", "_values", "_is_repeated"]

  def __init__(self, parent_index: np.ndarray, values: np.ndarray,
               is_repeated: bool):
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

  def schema_string(self) -> str:
    return u"{} {}".format("repeated" if self.is_repeated else "optional",
                           str(self.values.dtype))

  def __str__(self):
    return "{} {}".format("repeated" if self.is_repeated else "optional",
                          str(self.values.dtype))


NodeValue = Union[RootNodeValue, ChildNodeValue, LeafNodeValue]  # pylint: disable=invalid-name


class PrensorValue(object):
  """A tree of NodeValue objects."""

  __slots__ = ["_node", "_children"]

  def __init__(self, node: NodeValue,
               children: "collections.OrderedDict[path.Step, PrensorValue]"):
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
  def node(self) -> NodeValue:
    """The node of the root of the subtree."""
    return self._node

  def get_child(self, field_name: path.Step) -> Optional["PrensorValue"]:
    """Gets the child at field_name."""
    return self._children.get(field_name)

  def is_leaf(self) -> bool:
    """True iff the node value is a LeafNodeValue."""
    return isinstance(self._node, LeafNodeValue)

  def get_child_or_error(self, field_name: path.Step) -> "PrensorValue":
    """Gets the child at field_name."""
    result = self._children.get(field_name)
    if result is not None:
      return result
    raise ValueError("Field not found: {}".format(str(field_name)))

  def get_descendant(self, p: path.Path) -> Optional["PrensorValue"]:
    """Finds the descendant at the path."""
    result = self
    for field_name in p.field_list:
      result = result.get_child(field_name)
      if result is None:
        return None
    return result

  def get_descendant_or_error(self, p: path.Path) -> "PrensorValue":
    """Finds the descendant at the path."""
    result = self.get_descendant(p)
    if result is None:
      raise ValueError("Missing path: {}".format(str(p)))
    return result

  def get_children(self) -> Mapping[path.Step, "PrensorValue"]:
    """A map from field name to subtree."""
    return self._children

  def get_descendants(self) -> Mapping[path.Path, "PrensorValue"]:
    """A map from paths to all subtrees."""
    result = {path.Path([]): self}
    for k, v in self._children.items():
      subtree_descendants = v.get_descendants()
      for k2, v2 in subtree_descendants.items():
        result[path.Path([k]).concat(k2)] = v2
    return result

  def field_names(self) -> FrozenSet[path.Step]:
    """Returns the field names of the children."""
    return frozenset(self._children.keys())

  def _string_helper(self, field_name: str) -> Sequence[str]:
    """Helper for __str__ that outputs a list of lines."""
    result = [
        "{} {} {}".format(self.node.schema_string(), str(field_name),
                          self.node.data_string())
    ]
    for k, v in self._children.items():
      recursive = v._string_helper(k)  # pylint: disable=protected-access
      result.extend(["  {}".format(x) for x in recursive])
    return result

  def _schema_string_helper(self, field_name: str) -> Sequence[str]:
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


def _prensor_value_from_type_spec_and_component_values(
    prensor_type_spec: prensor._PrensorTypeSpec,
    component_values: Iterator[Union[int, np.ndarray]]) -> PrensorValue:
  """Creates a PrensorValue from a _PrensorTypeSpec and components."""
  # pylint: disable=protected-access
  if prensor_type_spec._node_type == prensor_type_spec._NodeType.ROOT:
    size = next(component_values)
    assert isinstance(size, (int, np.integer)), type(size)
    node = RootNodeValue(np.int64(size))
  elif prensor_type_spec._node_type == prensor_type_spec._NodeType.CHILD:
    node = ChildNodeValue(next(component_values),
                          prensor_type_spec._is_repeated)
  else:
    parent_index = next(component_values)
    values = next(component_values)
    node = LeafNodeValue(parent_index, values, prensor_type_spec._is_repeated)

  step_to_child = collections.OrderedDict()
  for step, child_spec in prensor_type_spec._children_specs:
    step_to_child[step] = _prensor_value_from_type_spec_and_component_values(
        child_spec, component_values)
  return PrensorValue(node, step_to_child)


def _prensor_value_fetch(prensor_tree: prensor.Prensor):
  """Fetch function for PrensorValue. See the document in session_lib."""
  # pylint: disable=protected-access
  type_spec = prensor_tree._type_spec
  components = type_spec._to_components(prensor_tree)
  def _construct_prensor_value(component_values):
    return _prensor_value_from_type_spec_and_component_values(
        type_spec, iter(component_values))

  return components, _construct_prensor_value


session_lib.register_session_run_conversion_functions(
    prensor.Prensor,
    _prensor_value_fetch,
    feed_function=None,
    feed_function_for_partial_run=None)
