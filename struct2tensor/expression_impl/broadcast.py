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
r"""Methods for broadcasting a path in a tree.

This provides methods for broadcasting a field anonymously (that is used in
promote_and_broadcast), or with an explicitly given name.

Suppose you have an expr representing:

```
+
|
+-session*   (stars indicate repeated)
     |
     +-event*
     |
     +-val*-int64

session: {
  event: {}
  event: {}
  val: 10
  val: 11
}
session: {
  event: {}
  event: {}
  val: 20
}
```

Then:

```
broadcast.broadcast(expr, path.Path(["session","val"]), "event", "nv")
```

becomes:

```
+
|
+---session*   (stars indicate repeated)
       |
       +-event*
       |   |
       |   +---nv*-int64
       |
       +-val*-int64

session: {
  event: {
    nv: 10
    nv:11
  }
  event: {
    nv: 10
    nv:11
  }
  val: 10
  val: 11
}
session: {
  event: {nv: 20}
  event: {nv: 20}
  val: 20
}
```

"""

from typing import FrozenSet, Optional, Sequence, Tuple

from struct2tensor import calculate_options
from struct2tensor import expression
from struct2tensor import expression_add
from struct2tensor import path
from struct2tensor import prensor
from struct2tensor.ops import struct2tensor_ops
import tensorflow as tf


class _BroadcastExpression(expression.Leaf):
  """A broadcast field."""

  def __init__(self, origin: expression.Expression,
               sibling: expression.Expression):
    super().__init__(origin.is_repeated, origin.type)
    if origin.type is None:
      raise ValueError("Can only broadcast a field")
    self._origin = origin
    self._sibling = sibling

  def get_source_expressions(self) -> Sequence[expression.Expression]:
    return [self._origin, self._sibling]

  def calculate(
      self,
      sources: Sequence[prensor.NodeTensor],
      destinations: Sequence[expression.Expression],
      options: calculate_options.Options,
      side_info: Optional[prensor.Prensor] = None) -> prensor.NodeTensor:
    [origin_value, sibling_value] = sources
    if not isinstance(origin_value, prensor.LeafNodeTensor):
      raise ValueError("origin not a LeafNodeTensor")
    if not isinstance(sibling_value, prensor.ChildNodeTensor):
      raise ValueError("sibling value is not a ChildNodeTensor")
    # For each i, for each v, if there exist exactly n values j such that:
    # sibling_value.parent_index[i]==origin_value.parent_index[j]
    # then there exists exactly n values k such that:
    # new_parent_index[k] = i
    # new_values[k] = origin_value.values[j]
    # (Ordering is also preserved).
    [broadcasted_to_sibling_index, index_to_values
    ] = struct2tensor_ops.equi_join_indices(sibling_value.parent_index,
                                            origin_value.parent_index)
    new_values = tf.gather(origin_value.values, index_to_values)
    return prensor.LeafNodeTensor(broadcasted_to_sibling_index, new_values,
                                  self.is_repeated)

  def calculation_is_identity(self) -> bool:
    return False

  def calculation_equal(self, expr: expression.Expression) -> bool:
    return isinstance(expr, _BroadcastExpression)


class _RecalculateExpression(expression.Expression):
  r"""Expression for recalculating a broadcasted subtree's parent indices.

  This is needed because when a subtree is broadcasted, the nested fields
  could be duplicated. Their parent indices need to be updated.

  For example, given the expression tree:
                    _DirectExpression(root)
                    /                      \
      _DirectExpression(parent)     _DirectExpression(sibling)
                  /
  _DirectExpression(submessage)

  If we broadcast (parent) into (sibling), then we get:
                    _DirectExpression(root)
                    /                      \
      _DirectExpression(parent)     _DirectExpression(sibling)
                  /                          \
  _DirectExpression(submessage)    _BroadcastChildExpression(new_parent)
                                               \
                                      _RecalculateExpression(new_submessage)

  This tree represents a field `parent` broadcasted into `sibling`. In this case
  the `origin` of `_RecalculateExpression(new_submessage)` would be
  `_DirectExpression(submessage)` and the `parent` of
  `_RecalculateExpression(new_submessage)` would be
  `_BroadcastChildExpression(new_parent)`

  Attributes:
    origin: The origin expression should be a _DirectExpression that represents
      the original subtree's field. It can represent a Child or Leaf NodeTensor.
    parent: The parent expression should be a _BroadcastChildExpression or
      _RecalculateExpression that represents the 'broadcasted' parent's
      ChildNodeTensor.
  """

  def __init__(self, origin: expression.Expression,
               parent: expression.Expression):
    super().__init__(
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

    # We should never be recalculating a RootNodeTensor.
    assert not isinstance(origin_value, prensor.RootNodeTensor), origin_value

    # The parent cannot be a LeafNodeTensor or RootNodeTensor, because
    #  a) a leaf node cannot have a submessage
    #  b) you cannot broadcast into a root
    assert isinstance(parent_value, prensor.ChildNodeTensor), parent_value

    # We use equi_join_any_indices on the parent's `index_to_value` because it
    # represents which child nodes were duplicated. Thus, which origin values
    # also need to be duplicated.
    [broadcasted_to_sibling_index, index_to_values
    ] = struct2tensor_ops.equi_join_any_indices(parent_value.index_to_value,
                                                origin_value.parent_index)

    if isinstance(origin_value, prensor.LeafNodeTensor):
      new_values = tf.gather(origin_value.values, index_to_values)
      return prensor.LeafNodeTensor(broadcasted_to_sibling_index, new_values,
                                    self.is_repeated)
    else:
      return prensor.ChildNodeTensor(broadcasted_to_sibling_index,
                                     self.is_repeated, index_to_values)

  def calculation_is_identity(self) -> bool:
    return False

  def calculation_equal(self, expr: expression.Expression) -> bool:
    return isinstance(expr, _RecalculateExpression)

  def _get_child_impl(self,
                      field_name: path.Step) -> Optional[expression.Expression]:
    """Gets the child expression.

    All children of a _RecalculateExpression should also be
    _RecalculateExpressions, because we all fields in the subtree need to be
    recalculated. This child is inherited from the origin's child.

    Args:
      field_name: the name of the child.

    Returns:
      an expression if a child exists, otherwise None.
    """
    original_child = self._origin.get_child(field_name)
    if original_child is None:
      return None
    return _RecalculateExpression(original_child, self)

  def known_field_names(self) -> FrozenSet[path.Step]:
    return self._origin.known_field_names()


class _BroadcastChildExpression(expression.Expression):
  """A broadcast expression for subtree.

  This expression represents the root of the subtree that is broadcasted.
  It wraps all of its children as `_RecalculateExpression`s.
  """

  def __init__(self, origin: expression.Expression,
               sibling: expression.Expression):
    super().__init__(
        origin.is_repeated,
        origin.type,
        validate_step_format=origin.validate_step_format,
    )
    self._origin = origin
    self._sibling = sibling

  def get_source_expressions(self) -> Sequence[expression.Expression]:
    return [self._origin, self._sibling]

  def calculate(
      self,
      sources: Sequence[prensor.NodeTensor],
      destinations: Sequence[expression.Expression],
      options: calculate_options.Options,
      side_info: Optional[prensor.Prensor] = None) -> prensor.NodeTensor:
    [origin_value, sibling_value] = sources
    if not isinstance(origin_value, prensor.ChildNodeTensor):
      raise ValueError("origin not a ChildNodeTensor")
    if not isinstance(sibling_value, prensor.ChildNodeTensor):
      raise ValueError("sibling value is not a ChildNodeTensor")

    [broadcasted_to_sibling_index, index_to_values
    ] = struct2tensor_ops.equi_join_any_indices(sibling_value.parent_index,
                                                origin_value.parent_index)
    return prensor.ChildNodeTensor(
        broadcasted_to_sibling_index,
        self.is_repeated,
        index_to_value=index_to_values)

  def calculation_is_identity(self) -> bool:
    return False

  def calculation_equal(self, expr: expression.Expression) -> bool:
    return isinstance(expr, _BroadcastChildExpression)

  def _get_child_impl(self,
                      field_name: path.Step) -> Optional[expression.Expression]:
    return _RecalculateExpression(self._origin.get_child(field_name), self)

  def known_field_names(self) -> FrozenSet[path.Step]:
    return self._origin.known_field_names()


def _broadcast_impl(
    root: expression.Expression, origin: path.Path, sibling: path.Step,
    new_field_name: path.Step) -> Tuple[expression.Expression, path.Path]:
  """Broadcasts origin to sibling for an expression."""
  sibling_path = origin.get_parent().get_child(sibling)

  origin_expression = root.get_descendant_or_error(origin)

  broadcast_expression_factory = (
      _BroadcastExpression
      if origin_expression.is_leaf else _BroadcastChildExpression)

  new_expr = broadcast_expression_factory(
      origin_expression,
      root.get_descendant_or_error(origin.get_parent().get_child(sibling)))
  new_path = sibling_path.get_child(new_field_name)
  result = expression_add.add_paths(root, {new_path: new_expr})

  return result, new_path


def broadcast_anonymous(
    root: expression.Expression, origin: path.Path,
    sibling: path.Step) -> Tuple[expression.Expression, path.Path]:
  return _broadcast_impl(root, origin, sibling, path.get_anonymous_field())


def broadcast(root: expression.Expression, origin: path.Path,
              sibling_name: path.Step,
              new_field_name: path.Step) -> expression.Expression:
  return _broadcast_impl(root, origin, sibling_name, new_field_name)[0]
