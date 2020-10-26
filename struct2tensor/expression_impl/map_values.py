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
"""Maps the values of various leaves of the same child to a single result.

All inputs must have the same shape (parent_index must be equal).

The output is given the same shape (output of function must be of equal length).

Note that the operations are on 1-D tensors (as opposed to scalars).

"""

from typing import Callable, FrozenSet, Optional, Sequence, Tuple

from struct2tensor import calculate_options
from struct2tensor import expression
from struct2tensor import expression_add
from struct2tensor import path
from struct2tensor import prensor
import tensorflow as tf


def map_many_values(
    root: expression.Expression, parent_path: path.Path,
    source_fields: Sequence[path.Step], operation: Callable[..., tf.Tensor],
    dtype: tf.DType,
    new_field_name: path.Step) -> Tuple[expression.Expression, path.Path]:
  """Map multiple sibling fields into a new sibling.

  All source fields must have the same shape, and the shape of the output
  must be the same as well.

  Args:
    root: original root.
    parent_path: parent path of all sources and the new field.
    source_fields: source fields of the operation. Must have the same shape.
    operation: operation from source_fields to new field.
    dtype: type of new field.
    new_field_name: name of the new field.

  Returns:
    The new expression and the new path as a pair.
  """
  new_path = parent_path.get_child(new_field_name)
  return expression_add.add_paths(
      root, {
          new_path:
              _MapValuesExpression([
                  root.get_descendant_or_error(parent_path.get_child(f))
                  for f in source_fields
              ], operation, dtype)
      }), new_path


def map_values_anonymous(
    root: expression.Expression, source_path: path.Path,
    operation: Callable[[tf.Tensor], tf.Tensor],
    dtype: tf.DType) -> Tuple[expression.Expression, path.Path]:
  """Map field into a new sibling.

  The shape of the output must be the same as the input.

  Args:
    root: original root.
    source_path: source of the operation.
    operation: operation from source_fields to new field.
    dtype: type of new field.

  Returns:
    The new expression and the new path as a pair.
  """
  if not source_path:
    raise ValueError('Cannot map the root.')
  return map_many_values(root, source_path.get_parent(),
                         [source_path.field_list[-1]], operation, dtype,
                         path.get_anonymous_field())


def map_values(root: expression.Expression, source_path: path.Path,
               operation: Callable[[tf.Tensor], tf.Tensor], dtype: tf.DType,
               new_field_name: path.Step) -> expression.Expression:
  """Map field into a new sibling.

  The shape of the output must be the same as the input.

  Args:
    root: original root.
    source_path: source of the operation.
    operation: operation from source_fields to new field.
    dtype: type of new field.
    new_field_name: name of the new field.

  Returns:
    The new expression.
  """
  if not source_path:
    raise ValueError('Cannot map the root.')
  return map_many_values(root, source_path.get_parent(),
                         [source_path.field_list[-1]], operation, dtype,
                         new_field_name)[0]


def _leaf_node_or_error(node: prensor.NodeTensor) -> prensor.LeafNodeTensor:
  if isinstance(node, prensor.LeafNodeTensor):
    return node
  raise ValueError('node is {} not LeafNodeTensor'.format(str(type(node))))


class _MapValuesExpression(expression.Expression):
  """Map the values of the given expression.

  _MapValuesExpression is intended to be a sibling of origin.
  The operation should return a tensor that is the same size as its input.
  """

  def __init__(self, origin: Sequence[expression.Expression],
               operation: Callable[..., tf.Tensor], dtype: tf.DType):
    super().__init__(origin[0].is_repeated, dtype)
    assert all([self.is_repeated == x.is_repeated for x in origin])
    self._origin = origin
    self._operation = operation

  def get_source_expressions(self) -> Sequence[expression.Expression]:
    return self._origin

  def calculate(
      self,
      sources: Sequence[prensor.NodeTensor],
      destinations: Sequence[expression.Expression],
      options: calculate_options.Options,
      side_info: Optional[prensor.Prensor] = None) -> prensor.NodeTensor:
    source_leaves = [_leaf_node_or_error(s) for s in sources]
    source_values = [s.values for s in source_leaves]
    # TODO(martinz): Check that:
    # source_values have equal parent_index.
    # output_value has the same size as the input.
    return prensor.LeafNodeTensor(source_leaves[0].parent_index,
                                  self._operation(*source_values),
                                  self._is_repeated)

  def calculation_is_identity(self) -> bool:
    return False

  def calculation_equal(self, expr: expression.Expression) -> bool:
    return self is expr

  def _get_child_impl(self,
                      field_name: path.Step) -> Optional[expression.Expression]:
    return None

  def known_field_names(self) -> FrozenSet[path.Step]:
    return frozenset()
