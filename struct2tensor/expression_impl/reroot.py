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
"""Reroot to a subtree, maintaining an input proto index.

reroot is similar to get_descendant_or_error. However, this method allows
you to call create_proto_index(...) later on, that gives you a reference to the
original proto.

"""
from typing import FrozenSet, Optional, Sequence

from struct2tensor import calculate_options
from struct2tensor import expression
from struct2tensor import expression_add
from struct2tensor import path
from struct2tensor import prensor
import tensorflow as tf


def reroot(root: expression.Expression,
           source_path: path.Path) -> expression.Expression:
  """Reroot to a new path, maintaining a input proto index.

  Similar to root.get_descendant_or_error(source_path): however, this
  method retains the ability to get a map to the original index.

  Args:
    root: the original root.
    source_path: the path to the new root.

  Returns:
    the new root.
  """

  new_root = root
  for step in source_path.field_list:
    new_root = _RerootExpression(new_root, step)
  return new_root


def create_proto_index_field(root: expression.Expression,
                             new_field_name: path.Step
                            ) -> expression.Expression:
  return expression_add.add_paths(
      root, {path.Path([new_field_name]): _InputProtoIndexExpression(root)})


class _RerootRootNodeTensor(prensor.RootNodeTensor):
  """The reroot root node.

  This contains a map from a current index to the original index of a proto.
  """

  def __init__(self, size: tf.Tensor, input_proto_index: tf.Tensor):
    super().__init__(size)
    self._input_proto_index = input_proto_index

  @property
  def input_proto_index(self):
    return self._input_proto_index


def _get_proto_index_parent_index(node: prensor.RootNodeTensor):
  return tf.range(node.size)


def _get_input_proto_index(node: prensor.RootNodeTensor):
  if isinstance(node, _RerootRootNodeTensor):
    return node.input_proto_index
  return _get_proto_index_parent_index(node)


class _RerootExpression(expression.Expression):
  """Reroot to a new path, maintaining a input proto index."""

  def __init__(self, original_root: expression.Expression,
               field_name: path.Step):
    super().__init__(
        True, None, validate_step_format=original_root.validate_step_format
    )
    self._field_name = field_name
    self._original_root = original_root
    self._new_root = original_root.get_child_or_error(field_name)
    if self._new_root.type is not None:
      raise ValueError("New root must be a message type: {}".format(
          str(self._field_name)))
    # TODO(martinz): Check that the "original root source expression" has a type
    # in (_RerootExpression, prensor._ProtoRootExpression)
    # To do this, we need a general technique similar to
    # expression_add._is_true_source_expression: however, this should also cover
    # intermediate operations like "project".
    # Since this check is not present, if it should have fired, there will be
    # an error when calculate(...) is called.

  def get_source_expressions(self) -> Sequence[expression.Expression]:
    return [self._original_root, self._new_root]

  def calculate(
      self,
      sources: Sequence[prensor.NodeTensor],
      destinations: Sequence[expression.Expression],
      options: calculate_options.Options,
      side_info: Optional[prensor.Prensor] = None) -> prensor.NodeTensor:
    [old_root_value, new_root_value] = sources
    if isinstance(old_root_value, prensor.RootNodeTensor) and isinstance(
        new_root_value, prensor.ChildNodeTensor):
      old_input_proto_index = _get_input_proto_index(old_root_value)
      # Notice that the "gather" operation is similar to promote.
      return _RerootRootNodeTensor(
          tf.size(new_root_value.parent_index, out_type=tf.int64),
          tf.gather(old_input_proto_index, new_root_value.parent_index))
    raise ValueError("Source types incorrect")

  def calculation_is_identity(self) -> bool:
    return False

  def calculation_equal(self, expr: expression.Expression) -> bool:
    # Although path can vary, it is not used in the calculation, just to
    return isinstance(expr, _RerootExpression)

  def _get_child_impl(self,
                      field_name: path.Step) -> Optional[expression.Expression]:
    return self._new_root.get_child(field_name)

  def known_field_names(self) -> FrozenSet[path.Step]:
    return self._new_root.known_field_names()


class _InputProtoIndexExpression(expression.Leaf):
  """A proto index expression."""

  def __init__(self, root: expression.Expression):
    """Constructor for proto index expression.

    Args:
      root: an expression that must return a RootNodeTensor.
    """
    super().__init__(is_repeated=False, my_type=tf.int64)
    self._root = root

  def get_source_expressions(self) -> Sequence[expression.Expression]:
    return [self._root]

  def calculate(
      self,
      sources: Sequence[prensor.NodeTensor],
      destinations: Sequence[expression.Expression],
      options: calculate_options.Options,
      side_info: Optional[prensor.Prensor] = None) -> prensor.NodeTensor:
    [root_node] = sources
    # The following check ensures not just that we can calculate the value,
    # but that no "improper" reroots were done.
    if isinstance(root_node, prensor.RootNodeTensor):
      return prensor.LeafNodeTensor(
          _get_proto_index_parent_index(root_node),
          _get_input_proto_index(root_node),
          is_repeated=False)
    raise ValueError(
        "Illegal operation: expected a true root node: got {}".format(
            str(root_node)))

  def calculation_is_identity(self) -> bool:
    return False

  def calculation_equal(self, expr: expression.Expression) -> bool:
    # Although path can vary, it is not used in the calculation, just to
    return isinstance(expr, _InputProtoIndexExpression)
