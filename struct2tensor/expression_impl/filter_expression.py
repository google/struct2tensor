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
r"""Create a new expression that is a filtered version of an original one.

There are two public methods in this module: filter_by_sibling and
filter_by_child. As with most other operations, these create a new tree which
has all the original paths of the original tree, but with a new subtree.

filter_by_sibling allows you to filter an expression by a boolean sibling field.

Beginning with the struct:

```
root =
         -----*----------------------------------------------------
        /                       \                                  \
     root0                    root1-----------------------      root2 (empty)
      /   \                   /    \               \      \
      |  keep_my_sib0:False  |  keep_my_sib1:True   | keep_my_sib2:False
    doc0-----               doc1---------------    doc2--------
     |       \                \           \    \               \
    bar:"a"  keep_me:False    bar:"b" bar:"c" keep_me:True      bar:"d"

# Note, keep_my_sib and doc must have the same shape (e.g., each root
has the same number of keep_my_sib children as doc children).
root_2 = filter_expression.filter_by_sibling(
    root, path.create_path("doc"), "keep_my_sib", "new_doc")

End with the struct (suppressing original doc):
         -----*----------------------------------------------------
        /                       \                                  \
    root0                    root1------------------        root2 (empty)
        \                   /    \                  \
        keep_my_sib0:False  |  keep_my_sib1:True   keep_my_sib2:False
                           new_doc0-----------
                             \           \    \
                             bar:"b" bar:"c" keep_me:True
```

filter_by_sibling allows you to filter an expression by a optional boolean
child field.

The following call will have the same effect as above:

```
root_2 = filter_expression.filter_by_child(
    root, path.create_path("doc"), "keep_me", "new_doc")
```

"""

from typing import FrozenSet, Optional, Sequence, Union

from struct2tensor import calculate_options
from struct2tensor import expression
from struct2tensor import expression_add
from struct2tensor import path
from struct2tensor import prensor
from struct2tensor.ops import struct2tensor_ops
import tensorflow as tf


def filter_by_sibling(expr: expression.Expression, p: path.Path,
                      sibling_field_name: path.Step,
                      new_field_name: path.Step) -> expression.Expression:
  """Filter an expression by its sibling.


  This is similar to boolean_mask. The shape of the path being filtered and
  the sibling must be identical (e.g., each parent object must have an
  equal number of source and sibling children).

  Args:
    expr: the root expression.
    p: a path to the source to be filtered.
    sibling_field_name: the sibling to use as a mask.
    new_field_name: a new sibling to create.

  Returns:
    a new root.
  """
  origin = expr.get_descendant_or_error(p)
  parent_path = p.get_parent()
  sibling = expr.get_descendant_or_error(
      parent_path.get_child(sibling_field_name))
  new_expr = _FilterBySiblingExpression(origin, sibling)
  new_path = parent_path.get_child(new_field_name)
  return expression_add.add_paths(expr, {new_path: new_expr})


def filter_by_child(expr: expression.Expression, p: path.Path,
                    child_field_name: path.Step,
                    new_field_name: path.Step) -> expression.Expression:
  """Filter an expression by an optional boolean child field.

  If the child field is present and True, then keep that parent.
  Otherwise, drop the parent.

  Args:
    expr: the original expression
    p: the path to filter.
    child_field_name: the boolean child field to use to filter.
    new_field_name: the new, filtered version of path.

  Returns:
    The new root expression.
  """
  origin = expr.get_descendant_or_error(p)
  child = origin.get_child_or_error(child_field_name)
  new_expr = _FilterByChildExpression(origin, child)
  new_path = p.get_parent().get_child(new_field_name)

  return expression_add.add_paths(expr, {new_path: new_expr})


#################### Private methods and classes follow ########################


class _FilterRootNodeTensor(prensor.RootNodeTensor):
  """The value of the root."""

  def __init__(self, size: tf.Tensor, indices_to_keep: tf.Tensor):
    """Initialize a root tensor that has indices_to_keep.

    Args:
      size: an int64 scalar tensor
      indices_to_keep: a 1D int64 tensor (int64 vector)
    """
    super().__init__(size)
    self._indices_to_keep = indices_to_keep

  @property
  def indices_to_keep(self) -> tf.Tensor:
    return self._indices_to_keep

  def __str__(self):
    return "_FilterRootNodeTensor"


class _FilterChildNodeTensor(prensor.ChildNodeTensor):
  """The value of an intermediate node."""

  def __init__(self, parent_index: tf.Tensor, is_repeated: bool,
               indices_to_keep: tf.Tensor):
    """Initialize a child node tensor with indices_to_keep.

    Args:
      parent_index: an int64 1D tensor (an int64 vector)
      is_repeated: true if there can be more than one element per parent
      indices_to_keep: an int64 1D tensor (an int64 vector)
    """
    super().__init__(parent_index, is_repeated)
    self._indices_to_keep = indices_to_keep

  def __str__(self):
    return "{} FilterChildNode".format(
        "repeated" if self.is_repeated else "optional")

  @property
  def indices_to_keep(self) -> tf.Tensor:
    return self._indices_to_keep


def _filter_by_self_indices_to_keep(node_value: prensor.NodeTensor,
                                    self_indices_to_keep: tf.Tensor
                                   ) -> prensor.NodeTensor:
  """Filter the node by the indices you want to keep."""
  if isinstance(node_value, prensor.RootNodeTensor):
    return _FilterRootNodeTensor(
        tf.size(self_indices_to_keep), self_indices_to_keep)
  if isinstance(node_value, prensor.ChildNodeTensor):
    return _FilterChildNodeTensor(
        tf.gather(node_value.parent_index, self_indices_to_keep),
        node_value.is_repeated, self_indices_to_keep)
  if isinstance(node_value, prensor.LeafNodeTensor):
    return prensor.LeafNodeTensor(
        tf.gather(node_value.parent_index, self_indices_to_keep),
        tf.gather(node_value.values, self_indices_to_keep),
        node_value.is_repeated)
  raise ValueError("Unknown NodeValue type")


def _filter_by_parent_indices_to_keep(
    node_value: Union[prensor.ChildNodeTensor, prensor.LeafNodeTensor],
    parent_indices_to_keep: tf.Tensor) -> prensor.NodeTensor:
  """Filter by parent indices to keep."""
  [new_parent_index, self_indices_to_keep
  ] = struct2tensor_ops.equi_join_indices(parent_indices_to_keep,
                                          node_value.parent_index)
  if isinstance(node_value, prensor.ChildNodeTensor):
    return _FilterChildNodeTensor(new_parent_index, node_value.is_repeated,
                                  self_indices_to_keep)
  if isinstance(node_value, prensor.LeafNodeTensor):
    return prensor.LeafNodeTensor(
        new_parent_index, tf.gather(node_value.values, self_indices_to_keep),
        node_value.is_repeated)
  raise ValueError("Unknown NodeValue type")


class _FilterChildByParentIndicesToKeepExpression(expression.Expression):
  """Filter all descendants of a _FilterBy(Sibling/Child)Expression.

  This expression is used to represent the descendants of
  _FilterBySiblingExpression and _FilterByChildExpression.
  """

  def __init__(self, origin: expression.Expression,
               parent: expression.Expression):
    super(_FilterChildByParentIndicesToKeepExpression, self).__init__(
        origin.is_repeated,
        origin.type,
        validate_step_format=origin.validate_step_format,
    )
    self._origin = origin
    self._parent = parent

  def get_source_expressions(self) -> Sequence[expression.Expression]:
    return [self._origin, self._parent]

  def calculate(
      self,
      sources: Sequence[prensor.NodeTensor],
      destinations: Sequence[expression.Expression],
      options: calculate_options.Options,
      side_info: Optional[prensor.Prensor] = None) -> prensor.NodeTensor:
    [origin_value, parent_value] = sources
    if (not isinstance(parent_value,
                       (_FilterChildNodeTensor, _FilterRootNodeTensor))):
      raise ValueError("Parent must be a filtered node")

    if (not isinstance(origin_value,
                       (prensor.ChildNodeTensor, prensor.LeafNodeTensor))):
      raise ValueError("Original must be a child or leaf node")

    parent_indices_to_keep = parent_value.indices_to_keep
    return _filter_by_parent_indices_to_keep(origin_value,
                                             parent_indices_to_keep)

  def calculation_is_identity(self) -> bool:
    return False

  def calculation_equal(self, expr: expression.Expression) -> bool:
    return isinstance(self, _FilterChildByParentIndicesToKeepExpression)

  def _get_child_impl(self,
                      field_name: path.Step) -> Optional[expression.Expression]:
    original_child = self._origin.get_child(field_name)
    if original_child is None:
      return None
    return _FilterChildByParentIndicesToKeepExpression(original_child, self)

  def known_field_names(self) -> FrozenSet[path.Step]:
    return self._origin.known_field_names()


def _self_indices_where_true(leaf_node: prensor.LeafNodeTensor) -> tf.Tensor:
  self_index = tf.range(
      tf.size(leaf_node.parent_index, out_type=tf.int64), dtype=tf.int64)
  return tf.boolean_mask(self_index, leaf_node.values)


def _parent_indices_where_true(leaf_node: prensor.LeafNodeTensor) -> tf.Tensor:
  return tf.boolean_mask(leaf_node.parent_index, leaf_node.values)


class _FilterBySiblingExpression(expression.Expression):
  """Project all subfields of an expression."""

  def __init__(self, origin: expression.Expression,
               sibling: expression.Expression):
    super().__init__(
        origin.is_repeated,
        origin.type,
        validate_step_format=origin.validate_step_format,
    )
    self._origin = origin
    self._sibling = sibling
    if sibling.type != tf.bool:
      raise ValueError("Sibling must be a boolean leaf.")

  def get_source_expressions(self) -> Sequence[expression.Expression]:
    return [self._origin, self._sibling]

  def calculate(
      self,
      sources: Sequence[prensor.NodeTensor],
      destinations: Sequence[expression.Expression],
      options: calculate_options.Options,
      side_info: Optional[prensor.Prensor] = None) -> prensor.NodeTensor:
    [origin_value, sibling_value] = sources
    if not isinstance(origin_value,
                      (prensor.ChildNodeTensor, prensor.LeafNodeTensor)):
      raise ValueError("Origin should not be a root")
    if not isinstance(sibling_value, prensor.LeafNodeTensor):
      raise ValueError("Sibling should be a leaf")
    # Check that they are the same shape.
    tf.assert_equal(sibling_value.parent_index, origin_value.parent_index)
    self_indices_to_keep = _self_indices_where_true(sibling_value)
    return _filter_by_self_indices_to_keep(origin_value, self_indices_to_keep)

  def calculation_is_identity(self) -> bool:
    return False

  def calculation_equal(self, expr: expression.Expression) -> bool:
    return isinstance(self, _FilterBySiblingExpression)

  def _get_child_impl(self,
                      field_name: path.Step) -> Optional[expression.Expression]:
    original = self._origin.get_child(field_name)
    if original is None:
      return None
    return _FilterChildByParentIndicesToKeepExpression(original, self)

  def known_field_names(self) -> FrozenSet[path.Step]:
    return self._origin.known_field_names()


class _FilterByChildExpression(expression.Expression):
  """Project all subfields of an expression."""

  def __init__(self, origin: expression.Expression,
               child: expression.Expression):
    super().__init__(
        origin.is_repeated,
        origin.type,
        validate_step_format=origin.validate_step_format,
    )
    self._origin = origin
    self._child = child
    if child.type != tf.bool:
      raise ValueError("Child must be boolean leaf")
    if child.is_repeated:
      raise ValueError("Child must not be repeated")

  def get_source_expressions(self) -> Sequence[expression.Expression]:
    return [self._origin, self._child]

  def calculate(
      self,
      sources: Sequence[prensor.NodeTensor],
      destinations: Sequence[expression.Expression],
      options: calculate_options.Options,
      side_info: Optional[prensor.Prensor] = None) -> prensor.NodeTensor:
    [origin_value, child_value] = sources
    if not isinstance(origin_value,
                      (prensor.ChildNodeTensor, prensor.LeafNodeTensor)):
      raise ValueError("Origin is the wrong type.")
    if not isinstance(child_value, prensor.LeafNodeTensor):
      raise ValueError("Child is not a leaf node.")
    # Here, we already know it is an optional field, so we don't need to check
    # the shape of parent_index.
    self_indices_to_keep = _parent_indices_where_true(child_value)
    return _filter_by_self_indices_to_keep(origin_value, self_indices_to_keep)

  def calculation_is_identity(self) -> bool:
    return False

  def calculation_equal(self, expr: expression.Expression) -> bool:
    return isinstance(self, _FilterByChildExpression)

  def _get_child_impl(self,
                      field_name: path.Step) -> Optional[expression.Expression]:
    original = self._origin.get_child(field_name)
    if original is None:
      return None
    return _FilterChildByParentIndicesToKeepExpression(original, self)

  def known_field_names(self) -> FrozenSet[path.Step]:
    return self._origin.known_field_names()
