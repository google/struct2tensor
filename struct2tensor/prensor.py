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
"""A representation of structured data as tensors.

A Prensor is usually created by calling calculate.calculate_prensor() on an
Expression. Prensors can be converted into SparseTensors, RaggedTensors,
or into expressions.

For operations on the expression, see prensor_util.py and create_expression.py.

"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import collections
import enum
import six
from struct2tensor import path
from struct2tensor import tf_types
import tensorflow as tf
from typing import FrozenSet, Iterator, List, Mapping, Optional, Sequence, Tuple, Union


# TODO(martinz): Consider creating node.py with the LeafNodeTensor,
# ChildNodeTensor, and RootNodeTensor, allowing expression.py to depend upon
# node.py.
class RootNodeTensor(object):
  """The value of the root."""

  __slots__ = ["_size"]

  def __init__(self, size):
    """Creates a root node.

    Args:
      size: A scalar int64 tensor saying how many root objects there are.
    """
    self._size = size

  @property
  def size(self):
    return self._size

  @property
  def is_repeated(self):
    return True

  def __str__(self):
    return "RootNodeTensor"


class ChildNodeTensor(object):
  """The value of an intermediate node."""

  __slots__ = ["_parent_index", "_is_repeated"]

  def __init__(self, parent_index, is_repeated):
    """Creates a child node.

    Args:
      parent_index: a 1-D int64 tensor where parent_index[i] represents the
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
      A scalar int64 tensor.
    """
    return tf.size(self._parent_index, out_type=tf.int64)

  @property
  def parent_index(self):
    return self._parent_index

  @property
  def is_repeated(self):
    return self._is_repeated

  def __str__(self):
    cardinality = "repeated" if self.is_repeated else "optional"
    return "{} ChildNodeTensor".format(cardinality)


class LeafNodeTensor(object):
  """The value of a leaf node."""

  __slots__ = ["_parent_index", "_values", "_is_repeated"]

  def __init__(self, parent_index, values,
               is_repeated):
    """Creates a LeafNodeTensor.

    Args:
      parent_index: a 1-D int64 tensor where parent_index[i] represents the
        parent index of values[i]
      values: a 1-D tensor of equal length to parent_index.
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

  def __str__(self):
    return "{} {}".format("repeated" if self.is_repeated else "optional",
                          str(self.values.dtype))


def create_required_leaf_node(values):
  """Create a required leaf node."""
  return LeafNodeTensor(
      tf.range(tf.size(values, out_type=tf.int64)), values, False)


NodeTensor = Union[LeafNodeTensor, ChildNodeTensor, RootNodeTensor]  # pylint: disable=invalid-name


class _PrensorTypeSpec(tf_types.TypeSpec):
  """TypeSpec for Prensor."""

  class _NodeType(enum.IntEnum):
    ROOT = 1
    CHILD = 2
    LEAF = 3

  __slots__ = [
      "_is_repeated", "_node_type", "_value_dtype", "_children_specs"]

  def __init__(self, is_repeated, node_type,
               value_dtype,
               children_specs):
    self._is_repeated = is_repeated
    self._node_type = node_type
    self._value_dtype = value_dtype
    self._children_specs = children_specs

  @property
  def value_type(self):
    return Prensor

  def _append_to_components(self,
                            value,
                            components):
    node = value.node
    if self._node_type == self._NodeType.ROOT:
      assert isinstance(node, RootNodeTensor)
      components.append(node.size)
    elif self._node_type == self._NodeType.CHILD:
      assert isinstance(node, ChildNodeTensor)
      components.append(node.parent_index)
    else:
      assert isinstance(node, LeafNodeTensor)
      components.append(node.parent_index)
      components.append(node.values)

    for (_, child_spec), child in six.moves.zip(
        self._children_specs, six.itervalues(value.get_children())):
      child_spec._append_to_components(  # pylint: disable=protected-access
          child, components)

  def _to_components(self, value):
    """Encodes a Prensor into a tuple of lists of Tensors.

    Args:
      value: A Prensor.

    Returns:
      A list of Tensors which are what each NodeTensor in a Prensor consists of.
      They are grouped by their owning NodeTensors in pre-order (the children
      of one node are ordered the same as the _children OrderedDict of that
      node).
    """
    result = []
    self._append_to_components(value, result)
    return result

  def _from_component_iter(
      self,
      component_iter):
    if self._node_type == self._NodeType.ROOT:
      node = RootNodeTensor(next(component_iter))
    elif self._node_type == self._NodeType.CHILD:
      node = ChildNodeTensor(next(component_iter), self._is_repeated)
    else:
      leaf_parent_index = next(component_iter)
      leaf_values = next(component_iter)
      node = LeafNodeTensor(leaf_parent_index, leaf_values, self._is_repeated)
    step_to_child = collections.OrderedDict()
    for step, child_spec in self._children_specs:
      step_to_child[step] = (
          child_spec._from_component_iter(  # pylint: disable=protected-access
              component_iter))
    return Prensor(node, step_to_child)

  def _from_components(self, components):
    """Creates a Prensor from the components encoded by _to_components()."""
    return self._from_component_iter(iter(components))

  def _append_to_component_specs(self, component_specs):
    if self._node_type == self._NodeType.ROOT:
      component_specs.append(tf.TensorSpec([], tf.int64))
    elif self._node_type == self._NodeType.CHILD:
      component_specs.append(tf.TensorSpec([None], tf.int64))
    else:
      component_specs.append(tf.TensorSpec([None], tf.int64))
      component_specs.append(
          tf.TensorSpec([None], self._value_dtype))
    for _, child_spec in self._children_specs:
      child_spec._append_to_component_specs(  # pylint: disable=protected-access
          component_specs)

  @property
  def _component_specs(self):
    """Returns TensorSpecs for each tensors returned by _to_components().

    Returns:
      a Tuple of Lists of the same structure as _to_components() Returns.
    """
    result = []
    self._append_to_component_specs(result)
    return result

  def _serialize(self):  # pylint: disable=g-bare-generic
    return (self._is_repeated, int(self._node_type), self._value_dtype,
            tuple((step,
                   child_spec._serialize())  # pylint: disable=protected-access
                  for step, child_spec in self._children_specs))

  @classmethod
  def _deserialize(cls, serialization):  # pylint: disable=g-bare-generic
    children_serializations = serialization[3]
    children_specs = [(step, cls._deserialize(child_serialization))
                      for step, child_serialization in children_serializations]
    return cls(
        serialization[0],
        cls._NodeType(serialization[1]),
        serialization[2],
        children_specs)

  def _to_legacy_output_types(self):
    return tuple(spec.dtype for spec in self._component_specs)

  def _to_legacy_output_shapes(self):
    return tuple(spec.shape for spec in self._component_specs)

  def _to_legacy_output_classes(self):
    return tuple(tf.Tensor for spec in self._component_specs)


class Prensor(tf_types.CompositeTensorMixin):
  """A expression of NodeTensor objects."""

  __slots__ = ["_node", "_children"]

  def __init__(self, node,
               children):
    """Construct a Prensor.

    Do not call directly, instead call either:
      create_prensor_from_descendant_nodes or
      create_prensor_from_root_and_children

    Args:
      node: the NodeTensor of the root.
      children: a map from edge to subexpression.
    """
    self._node = node
    self._children = children

  @property
  def node(self):
    """The node of the root of the subtree."""
    return self._node

  def get_child(self, field_name):
    """Gets the child at field_name."""
    return self._children.get(field_name)

  @property
  def is_leaf(self):
    """True iff the node value is a LeafNodeTensor."""
    return isinstance(self._node, LeafNodeTensor)

  def get_child_or_error(self, field_name):
    """Gets the child at field_name."""
    result = self._children.get(field_name)
    if result is not None:
      return result
    raise ValueError("Field not found: {} in {}".format(
        str(field_name), str(self)))

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
      raise ValueError("Missing path: {} in {}".format(str(p), str(self)))
    return result

  def get_children(self):
    """A map from field name to subexpression."""
    return self._children

  def get_descendants(self):
    """A map from paths to all subexpressions."""
    result = {path.Path([]): self}
    for k, v in self._children.items():
      subexpression_descendants = v.get_descendants()
      for k2, v2 in subexpression_descendants.items():
        result[path.Path([k]).concat(k2)] = v2
    return result

  def field_names(self):
    """Returns the field names of the children."""
    return frozenset(self._children.keys())

  def _string_helper(self, field_name):  # pylint: disable=g-ambiguous-str-annotation
    """Helper for __str__ that outputs a list of lines.

    Args:
      field_name: the field name for this node in its parent.

    Returns:
      lines to run __str__, that are bytes in Python 2 and unicode in Python 3.
    """
    result = ["{} {}".format(str(self.node), str(field_name))]
    for k, v in self._children.items():
      recursive = v._string_helper(k)  # pylint: disable=protected-access
      result.extend(["  {}".format(x) for x in recursive])
    return result

  def __str__(self):  # pylint: disable=g-ambiguous-str-annotation
    """Returns a string representing the schema of the Prensor."""
    return "\n".join(self._string_helper("root"))

  @property
  def _type_spec(self):
    is_repeated = None
    value_dtype = None
    # pylint: disable=protected-access
    if isinstance(self.node, RootNodeTensor):
      node_type = _PrensorTypeSpec._NodeType.ROOT
    elif isinstance(self.node, ChildNodeTensor):
      is_repeated = self.node.is_repeated
      node_type = _PrensorTypeSpec._NodeType.CHILD
    else:
      is_repeated = self.node.is_repeated
      node_type = _PrensorTypeSpec._NodeType.LEAF
      value_dtype = self.node.values.dtype
    return _PrensorTypeSpec(
        is_repeated,
        node_type,
        value_dtype,
        [(step, child._type_spec)
         for step, child in six.iteritems(self.get_children())])
    # pylint: enable=protected-access


def create_prensor_from_descendant_nodes(
    nodes):
  """Create a prensor from a map of paths to NodeTensor.

  If a path is a key in the map, all prefixes of that path must be present.

  Args:
    nodes: A map from paths to NodeTensors.

  Returns:
    A Prensor.

  Raises:
    ValueError: if there is a prefix of a path missing.
  """
  subexpressions = collections.OrderedDict()
  root_node = None
  for k, v in sorted(nodes.items()):
    if not k:
      root_node = v
    else:
      first_step = k.field_list[0]
      suffix = k.suffix(1)
      if first_step not in subexpressions:
        subexpressions[first_step] = {}
      subexpressions[first_step][suffix] = v
  if root_node is None:
    raise ValueError("No root found: {}".format(str(nodes)))
  return create_prensor_from_root_and_children(
      root_node,
      collections.OrderedDict((k, create_prensor_from_descendant_nodes(v))
                              for k, v in subexpressions.items()))


def create_prensor_from_root_and_children(
    root, children):
  if isinstance(children, collections.OrderedDict):
    ordered_children = children
  else:
    ordered_children = collections.OrderedDict(sorted(children.items()))
  return Prensor(root, ordered_children)
