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

For operations on the expression, see create_expression.py.

"""

import collections
import enum
from typing import FrozenSet, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

from struct2tensor import calculate_options
from struct2tensor import path
from struct2tensor.ops import struct2tensor_ops
import tensorflow as tf

from tensorflow.python.framework import composite_tensor  # pylint: disable=g-direct-tensorflow-import


# TODO(martinz): Consider creating node.py with the LeafNodeTensor,
# ChildNodeTensor, and RootNodeTensor, allowing expression.py to depend upon
# node.py.
class RootNodeTensor(object):
  """The value of the root."""

  __slots__ = ["_size"]

  def __init__(self, size: tf.Tensor):
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

  def get_positional_index(self) -> tf.Tensor:
    """Gets the positional index for this RootNodeTensor.

    The positional index relative to the node's parent, and thus is always
    monotonically increasing at step size 1 for a RootNodeTensor.

    Returns:
      A tensor of positional indices.
    """
    return tf.range(self.size)

  def __str__(self):
    return "RootNodeTensor"


class ChildNodeTensor(object):
  """The value of an intermediate node."""

  __slots__ = ["_parent_index", "_is_repeated", "_index_to_value"]

  def __init__(self,
               parent_index: tf.Tensor,
               is_repeated: bool,
               index_to_value: Optional[tf.Tensor] = None):
    """Creates a child node.

    Args:
      parent_index: a 1-D int64 tensor where parent_index[i] represents the
        parent index of the ith child.
      is_repeated: a bool indicating if there can be more than one child per
        parent.
      index_to_value: a 1-D int64 tensor where index_to_value[i] represents the
        `value` of the ith child. Where `value` is a subtree.
    """
    self._parent_index = parent_index
    self._is_repeated = is_repeated
    self._index_to_value = index_to_value

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

  @property
  def index_to_value(self):
    return self._index_to_value

  # LINT.IfChange(child_node_tensor)
  def get_positional_index(self) -> tf.Tensor:
    """Gets the positional index for this ChildNodeTensor.

    The positional index tells us which index of the parent an element is.

    For example, with the following parent indices: [0, 0, 2]
    we would have positional index:
    [
      0, # The 0th element of the 0th parent.
      1, # The 1st element of the 0th parent.
      0  # The 0th element of the 2nd parent.
    ].

    For more information, view ops/run_length_before_op.cc

    This is the same for Leaf NodeTensors.

    Returns:
      A tensor of positional indices.
    """
    return struct2tensor_ops.run_length_before(self.parent_index)
  # LINT.ThenChange(:leaf_node_tensor)

  def __str__(self):
    cardinality = "repeated" if self.is_repeated else "optional"
    return "{} ChildNodeTensor".format(cardinality)


class LeafNodeTensor(object):
  """The value of a leaf node."""

  __slots__ = ["_parent_index", "_values", "_is_repeated"]

  def __init__(self, parent_index: tf.Tensor, values: tf.Tensor,
               is_repeated: bool):
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

  # LINT.IfChange(leaf_node_tensor)
  def get_positional_index(self) -> tf.Tensor:
    """Gets the positional index for this LeafNodeTensor.

    The positional index tells us which index of the parent an element is.

    For example, with the following parent indices: [0, 0, 2]
    we would have positional index:
    [
      0, # The 0th element of the 0th parent.
      1, # The 1st element of the 0th parent.
      0  # The 0th element of the 2nd parent.
    ].

    For more information, view ops/run_length_before_op.cc

    This is the same for Child NodeTensors.

    Returns:
      A tensor of positional indices.
    """
    return struct2tensor_ops.run_length_before(self.parent_index)
  # LINT.ThenChange(:child_node_tensor)

  def __str__(self):
    return "{} {}".format("repeated" if self.is_repeated else "optional",
                          str(self.values.dtype))


def create_required_leaf_node(values: tf.Tensor) -> LeafNodeTensor:
  """Create a required leaf node."""
  return LeafNodeTensor(
      tf.range(tf.size(values, out_type=tf.int64)), values, False)


NodeTensor = Union[LeafNodeTensor, ChildNodeTensor, RootNodeTensor]  # pylint: disable=invalid-name


class _PrensorTypeSpec(tf.TypeSpec):
  """TypeSpec for Prensor."""

  class _NodeType(enum.IntEnum):
    ROOT = 1
    CHILD = 2
    LEAF = 3

  __slots__ = [
      "_is_repeated", "_node_type", "_value_dtype", "_children_specs"]

  def __init__(self, is_repeated: Optional[bool], node_type: _NodeType,
               value_dtype: Optional[tf.DType],
               children_specs: List[Tuple[path.Step, "_PrensorTypeSpec"]]):
    self._is_repeated = is_repeated
    self._node_type = node_type
    self._value_dtype = value_dtype
    self._children_specs = children_specs

  @property
  def value_type(self):
    return Prensor

  def _append_to_components(self,
                            value: "Prensor",
                            components: List[tf.Tensor]):
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

    for (_, child_spec), child in zip(
        self._children_specs, value.get_children().values()):
      child_spec._append_to_components(  # pylint: disable=protected-access
          child, components)

  def _to_components(self, value: "Prensor") -> List[tf.Tensor]:
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
      component_iter: Iterator[tf.Tensor]) -> "Prensor":
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

  def _from_components(self, components: List[tf.Tensor]) -> "Prensor":
    """Creates a Prensor from the components encoded by _to_components()."""
    return self._from_component_iter(iter(components))

  def _append_to_component_specs(self, component_specs: List[tf.TensorSpec]):
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
  def _component_specs(self) -> List[tf.TensorSpec]:
    """Returns TensorSpecs for each tensors returned by _to_components().

    Returns:
      a Tuple of Lists of the same structure as _to_components() Returns.
    """
    result = []
    self._append_to_component_specs(result)
    return result

  def _serialize(self) -> Tuple[bool, int, tf.DType, Tuple]:  # pylint: disable=g-bare-generic
    return (self._is_repeated, int(self._node_type), self._value_dtype,
            tuple((step,
                   child_spec._serialize())  # pylint: disable=protected-access
                  for step, child_spec in self._children_specs))

  @classmethod
  def _deserialize(cls, serialization: Tuple[bool, int, tf.DType, Tuple]):  # pylint: disable=g-bare-generic
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


class Prensor(composite_tensor.CompositeTensor):
  """A expression of NodeTensor objects."""

  __slots__ = ["_node", "_children"]

  def __init__(self, node: NodeTensor,
               children: "collections.OrderedDict[path.Step, Prensor]"):
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
  def node(self) -> NodeTensor:
    """The node of the root of the subtree."""
    return self._node

  def get_child(self, field_name: path.Step) -> Optional["Prensor"]:
    """Gets the child at field_name."""
    return self._children.get(field_name)

  @property
  def is_leaf(self) -> bool:
    """True iff the node value is a LeafNodeTensor."""
    return isinstance(self._node, LeafNodeTensor)

  def get_child_or_error(self, field_name: path.Step) -> "Prensor":
    """Gets the child at field_name."""
    result = self._children.get(field_name)
    if result is not None:
      return result
    raise ValueError("Field not found: {} in {}".format(
        str(field_name), str(self)))

  def get_descendant(self, p: path.Path) -> Optional["Prensor"]:
    """Finds the descendant at the path."""
    result = self
    for field_name in p.field_list:
      result = result.get_child(field_name)
      if result is None:
        return None
    return result

  def get_descendant_or_error(self, p: path.Path) -> "Prensor":
    """Finds the descendant at the path."""
    result = self.get_descendant(p)
    if result is None:
      raise ValueError("Missing path: {} in {}".format(str(p), str(self)))
    return result

  def get_children(self) -> "collections.OrderedDict[path.Step, Prensor]":
    """A map from field name to subexpression."""
    return self._children

  def get_descendants(self) -> Mapping[path.Path, "Prensor"]:
    """A map from paths to all subexpressions."""
    result = {path.Path([]): self}
    for k, v in self._children.items():
      subexpression_descendants = v.get_descendants()
      for k2, v2 in subexpression_descendants.items():
        # Since k is already a path.Step, we know it is valid and needn't
        # validate it again.
        result[path.Path([k], validate_step_format=False).concat(k2)] = v2
    return result

  def field_names(self) -> FrozenSet[path.Step]:
    """Returns the field names of the children."""
    return frozenset(self._children.keys())

  def get_ragged_tensors(
      self,
      options: calculate_options.Options = calculate_options
      .get_default_options()
  ) -> Mapping[path.Path, tf.RaggedTensor]:
    """Gets ragged tensors for all the leaves of the prensor expression.

    Args:
      options: Options for calculating ragged tensors.

    Returns:
      A map from paths to ragged tensors.
    """
    return _get_ragged_tensors(self, options=options)

  def get_ragged_tensor(
      self,
      p: path.Path,
      options: calculate_options.Options = calculate_options
      .get_default_options()
  ) -> tf.RaggedTensor:
    """Get a ragged tensor for a path.

    All steps are represented in the ragged tensor.

    Args:
      p: the path to a leaf node in `t`.
      options: Options for calculating ragged tensors.

    Returns:
      A ragged tensor containing values of the leaf node, preserving the
      structure along the path. Raises an error if the path is not found.
    """
    return _get_ragged_tensor(self, p, options=options)

  def get_sparse_tensor(
      self,
      p: path.Path,
      options: calculate_options.Options = calculate_options
      .get_default_options()
  ) -> tf.SparseTensor:
    """Gets a sparse tensor for path p.

    Note that any optional fields are not registered as dimensions, as they
    can't be represented in a sparse tensor.

    Args:
      p: The path to a leaf node in `t`.
      options: Currently unused.

    Returns:
      A sparse tensor containing values of the leaf node, preserving the
      structure along the path. Raises an error if the path is not found.
    """
    return _get_sparse_tensor(self, p, options=options)

  def get_sparse_tensors(
      self,
      options: calculate_options.Options = calculate_options
      .get_default_options()
  ) -> Mapping[path.Path, tf.SparseTensor]:
    """Gets sparse tensors for all the leaves of the prensor expression.

    Args:
      options: Currently unused.

    Returns:
      A map from paths to sparse tensors.
    """
    return _get_sparse_tensors(self, options=options)

  def _string_helper(self, field_name: path.Step) -> Sequence[str]:
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

  def __str__(self) -> str:
    """Returns a string representing the schema of the Prensor."""
    return "\n".join(self._string_helper("root"))

  @property
  def _type_spec(self) -> _PrensorTypeSpec:
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
         for step, child in self.get_children().items()])
    # pylint: enable=protected-access


def create_prensor_from_descendant_nodes(
    nodes: Mapping[path.Path, NodeTensor]) -> "Prensor":
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
    root: NodeTensor, children: Mapping[path.Step, Prensor]) -> Prensor:
  if isinstance(children, collections.OrderedDict):
    ordered_children = children
  else:
    ordered_children = collections.OrderedDict(sorted(children.items()))
  return Prensor(root, ordered_children)

######## Below code is for converting prensor to ragged/sparse tensors.#########


class _LeafNodePath(object):
  """A path ending in a leaf.

  In order to avoid type checks and casting in the heart of different methods
  using the Prensor object to get a ragged or sparse tensor, we first create a
  typed "list" of nodes. A _LeafNodePath always begins with the root and ends
  with a leaf. Notice that we can get a suffix by casting a child node to a
  root node.
  """

  def __init__(self, head: RootNodeTensor, middle: Sequence[ChildNodeTensor],
               tail: LeafNodeTensor):
    self._head = head
    self._middle = middle
    self._tail = tail

  @property
  def head(self) -> RootNodeTensor:
    return self._head

  @property
  def middle(self) -> Sequence[ChildNodeTensor]:
    return self._middle

  @property
  def tail(self) -> LeafNodeTensor:
    return self._tail


class _ChildNodePath(object):
  """A _ChildNodePath is a path that ends with a child node.

  It keeps same triple structure as _LeafNodePath.
  We use these in _get_dewey_encoding.
  """

  def __init__(self, head: RootNodeTensor, middle: Sequence[ChildNodeTensor],
               tail: ChildNodeTensor):
    self._head = head
    self._middle = middle
    self._tail = tail

  @property
  def head(self) -> RootNodeTensor:
    return self._head

  @property
  def middle(self) -> Sequence[ChildNodeTensor]:
    return self._middle

  @property
  def tail(self) -> ChildNodeTensor:
    return self._tail


def _as_root_node_tensor(node_tensor: NodeTensor) -> RootNodeTensor:
  if isinstance(node_tensor, RootNodeTensor):
    return node_tensor
  if isinstance(node_tensor, ChildNodeTensor):
    return RootNodeTensor(node_tensor.size)
  raise ValueError("Must be child or root node tensor (found {})".format(
      type(node_tensor)))


def _get_leaf_node_path(p: path.Path, t: Prensor) -> _LeafNodePath:
  """Creates a _LeafNodePath to p."""
  leaf_node = t.get_descendant_or_error(p).node
  if not isinstance(leaf_node, LeafNodeTensor):
    raise ValueError("Expected Leaf Node at {} in {}".format(str(p), str(t)))
  if not p:
    raise ValueError("Leaf should not be at the root")
  # If there is a leaf at the root, this will return a ValueError.
  root_node = _as_root_node_tensor(t.node)

  # Not the root, not p.
  strict_ancestor_paths = [p.prefix(i) for i in range(1, len(p))]

  child_node_pairs = [(t.get_descendant_or_error(ancestor).node, ancestor)
                      for ancestor in strict_ancestor_paths]
  bad_struct_paths = [
      ancestor for node, ancestor in child_node_pairs
      if not isinstance(node, ChildNodeTensor)
  ]
  if bad_struct_paths:
    raise ValueError("Expected ChildNodeTensor at {} in {}".format(
        " ".join([str(x) for x in bad_struct_paths]), str(t)))
  # This should select all elements: the isinstance is for type-checking.
  child_nodes = [
      node for node, ancestor in child_node_pairs
      if isinstance(node, ChildNodeTensor)
  ]
  assert len(child_nodes) == len(child_node_pairs)
  return _LeafNodePath(root_node, child_nodes, leaf_node)


def _get_leaf_node_path_suffix(p: _LeafNodePath) -> _LeafNodePath:
  """Get the suffix of a LeafNodePath."""
  return _LeafNodePath(_as_root_node_tensor(p.middle[0]), p.middle[1:], p.tail)


def _get_node_path_parent(
    p: Union[_LeafNodePath, _ChildNodePath]) -> _ChildNodePath:
  return _ChildNodePath(p.head, p.middle[:-1], p.middle[-1])


def _get_leaf_node_paths(t: Prensor) -> Mapping[path.Path, _LeafNodePath]:
  """Gets a map of paths to leaf nodes in the expression."""
  return {
      k: _get_leaf_node_path(k, t)
      for k, v in t.get_descendants().items()
      if isinstance(v.node, LeafNodeTensor)
  }


#################### Code for _get_sparse_tensors(...) #########################


def _get_dewey_encoding(
    p: Union[_LeafNodePath, _ChildNodePath]) -> Tuple[tf.Tensor, tf.Tensor]:
  """Gets a pair of the indices and shape of these protos.

  See http://db.ucsd.edu/static/cse232B-s05/papers/tatarinov02.pdf

  Args:
    p: the path to encode.

  Returns:
    A pair of an indices matrix and a dense_shape
  """
  parent = p.middle[-1] if p.middle else p.head
  parent_size = tf.reshape(parent.size, [1])
  positional_index = p.tail.get_positional_index()
  # tf.reduce_max([]) == -kmaxint64 but we need it to be 0.
  current_size = tf.maximum(
      tf.reshape(tf.reduce_max(positional_index) + 1, [1]), [0])
  if not p.middle:
    if p.tail.is_repeated:
      return (tf.stack([p.tail.parent_index, positional_index],
                       axis=1), tf.concat([parent_size, current_size], 0))
    else:
      return tf.expand_dims(p.tail.parent_index, -1), parent_size
  else:
    parent_dewey_encoding, parent_size = _get_dewey_encoding(
        _get_node_path_parent(p))
    if p.tail.is_repeated:
      positional_index_as_matrix = tf.expand_dims(
          p.tail.get_positional_index(), -1)
      indices = tf.concat([
          tf.gather(parent_dewey_encoding, p.tail.parent_index),
          positional_index_as_matrix
      ], 1)
      size = tf.concat([parent_size, current_size], 0)
      return (indices, size)
    else:
      return tf.gather(parent_dewey_encoding, p.tail.parent_index), parent_size


def _get_sparse_tensor_from_leaf_node_path(p: _LeafNodePath) -> tf.SparseTensor:
  indices, dense_shape = _get_dewey_encoding(p)
  return tf.SparseTensor(
      indices=indices, values=p.tail.values, dense_shape=dense_shape)


def _get_sparse_tensor(
    t: Prensor,
    p: path.Path,
    options: calculate_options.Options = calculate_options.get_default_options(
    )
) -> tf.SparseTensor:
  """Gets a sparse tensor for path p.

  Note that any optional fields are not registered as dimensions, as they can't
  be represented in a sparse tensor.

  Args:
    t: The Prensor to extract tensors from.
    p: The path to a leaf node in `t`.
    options: Currently unused.

  Returns:
    A sparse tensor containing values of the leaf node, preserving the
    structure along the path. Raises an error if the path is not found.
  """
  del options
  leaf_node_path = _get_leaf_node_path(p, t)
  return _get_sparse_tensor_from_leaf_node_path(leaf_node_path)


def _get_sparse_tensors(
    t: Prensor,
    options: calculate_options.Options = calculate_options.get_default_options(
    )
) -> Mapping[path.Path, tf.SparseTensor]:
  """Gets sparse tensors for all the leaves of the prensor expression.

  Args:
    t: The Prensor to extract tensors from.
    options: Currently unused.

  Returns:
    A map from paths to sparse tensors.
  """

  del options
  return {
      p: _get_sparse_tensor_from_leaf_node_path(v)
      for p, v in _get_leaf_node_paths(t).items()
  }


#################### Code for _get_ragged_tensors(...) #########################


def from_value_rowids_bridge(values,
                             value_rowids=None,
                             nrows=None,
                             validate=True):
  """validate option is only available internally for tf 0.13.1."""
  return tf.RaggedTensor.from_value_rowids(
      values, value_rowids=value_rowids, nrows=nrows, validate=validate)


def _get_ragged_tensor_from_leaf_node_path(
    nodes: _LeafNodePath,
    options: calculate_options.Options = calculate_options.get_default_options(
    )
) -> tf.RaggedTensor:
  """Gets a ragged tensor from a leaf node path."""
  if not nodes.middle:
    return from_value_rowids_bridge(
        nodes.tail.values,
        value_rowids=nodes.tail.parent_index,
        nrows=nodes.head.size,
        validate=options.ragged_checks)
  deeper_ragged = _get_ragged_tensor_from_leaf_node_path(
      _get_leaf_node_path_suffix(nodes), options)
  first_child_node = nodes.middle[0]
  return from_value_rowids_bridge(
      deeper_ragged,
      value_rowids=first_child_node.parent_index,
      nrows=nodes.head.size,
      validate=options.ragged_checks)


def _get_ragged_tensor(
    t: Prensor,
    p: path.Path,
    options: calculate_options.Options = calculate_options.get_default_options(
    )
) -> tf.RaggedTensor:
  """Get a ragged tensor for a path.

  All steps are represented in the ragged tensor.

  Args:
    t: The Prensor to extract tensors from.
    p: the path to a leaf node in `t`.
    options: used to pass options for calculating ragged tensors.

  Returns:
    A ragged tensor containing values of the leaf node, preserving the
    structure along the path. Raises an error if the path is not found.
  """
  leaf_node_path = _get_leaf_node_path(p, t)
  return _get_ragged_tensor_from_leaf_node_path(leaf_node_path, options)


def _get_ragged_tensors(
    t: Prensor,
    options: calculate_options.Options = calculate_options.get_default_options(
    )
) -> Mapping[path.Path, tf.RaggedTensor]:
  """Gets ragged tensors for all the leaves of the prensor expression.

  Args:
    t: The Prensor to extract tensors from.
    options: used to pass options for calculating ragged tensors.

  Returns:
    A map from paths to ragged tensors.
  """
  return {
      p: _get_ragged_tensor_from_leaf_node_path(v, options)
      for p, v in _get_leaf_node_paths(t).items()
  }
