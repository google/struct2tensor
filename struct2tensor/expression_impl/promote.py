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
r"""Promote an expression to be a child of its grandparent.

Promote is part of the standard flattening of data, promote_and_broadcast,
which takes structured data and flattens it. By directly accessing promote,
one can perform simpler operations.

For example, suppose an expr represents:

```
+
|
+-session*   (stars indicate repeated)
     |
     +-event*
         |
         +-val*-int64

session: {
  event: {
    val: 111
  }
  event: {
    val: 121
    val: 122
  }
}

session: {
  event: {
    val: 10
    val: 7
  }
  event: {
    val: 1
  }
}

```

```
promote.promote(expr, path.Path(["session", "event", "val"]), nval)
```

produces:

```
+
|
+-session*   (stars indicate repeated)
     |
     +-event*
     |    |
     |    +-val*-int64
     |
     +-nval*-int64

session: {
  event: {
    val: 111
  }
  event: {
    val: 121
    val: 122
  }
  nval: 111
  nval: 121
  nval: 122
}

session: {
  event: {
    val: 10
    val: 7
  }
  event: {
    val: 1
  }
  nval: 10
  nval: 7
  nval: 1
}
```

"""

from typing import FrozenSet, Optional, Sequence, Tuple

from struct2tensor import calculate_options
from struct2tensor import expression
from struct2tensor import expression_add
from struct2tensor import path
from struct2tensor import prensor
import tensorflow as tf

from tensorflow_metadata.proto.v0 import schema_pb2


class PromoteExpression(expression.Leaf):
  """A promoted leaf."""

  def __init__(self, origin: expression.Expression,
               origin_parent: expression.Expression):

    super().__init__(
        origin.is_repeated or origin_parent.is_repeated,
        origin.type,
        schema_feature=_get_promote_schema_feature(
            origin.schema_feature, origin_parent.schema_feature))
    self._origin = origin
    self._origin_parent = origin_parent
    if self.type is None:
      raise ValueError("Can only promote a field")
    if self._origin_parent.type is not None:
      raise ValueError("origin_parent cannot be a field")

  def get_source_expressions(self) -> Sequence[expression.Expression]:
    return [self._origin, self._origin_parent]

  def calculate(
      self,
      sources: Sequence[prensor.NodeTensor],
      destinations: Sequence[expression.Expression],
      options: calculate_options.Options,
      side_info: Optional[prensor.Prensor] = None) -> prensor.NodeTensor:
    [origin_value, origin_parent_value] = sources
    if not isinstance(origin_value, prensor.LeafNodeTensor):
      raise ValueError("origin_value must be a leaf")
    if not isinstance(origin_parent_value, prensor.ChildNodeTensor):
      raise ValueError("origin_parent_value must be a child node")
    new_parent_index = tf.gather(origin_parent_value.parent_index,
                                 origin_value.parent_index)
    return prensor.LeafNodeTensor(new_parent_index, origin_value.values,
                                  self.is_repeated)

  def calculation_is_identity(self) -> bool:
    return False

  def calculation_equal(self, expr: expression.Expression) -> bool:
    return isinstance(expr, PromoteExpression)


class PromoteChildExpression(expression.Expression):
  """The root of the promoted sub tree."""

  def __init__(self, origin: expression.Expression,
               origin_parent: expression.Expression):

    super().__init__(
        origin.is_repeated or origin_parent.is_repeated,
        origin.type,
        schema_feature=_get_promote_schema_feature(
            origin.schema_feature, origin_parent.schema_feature
        ),
        validate_step_format=origin.validate_step_format,
    )
    self._origin = origin
    self._origin_parent = origin_parent
    if self._origin_parent.type is not None:
      raise ValueError("origin_parent cannot be a field")

  def get_source_expressions(self) -> Sequence[expression.Expression]:
    return [self._origin, self._origin_parent]

  def calculate(
      self,
      sources: Sequence[prensor.NodeTensor],
      destinations: Sequence[expression.Expression],
      options: calculate_options.Options,
      side_info: Optional[prensor.Prensor] = None) -> prensor.NodeTensor:
    [origin_value, origin_parent_value] = sources
    if not isinstance(origin_value, prensor.ChildNodeTensor):
      raise ValueError("origin_value must be a child")
    if not isinstance(origin_parent_value, prensor.ChildNodeTensor):
      raise ValueError("origin_parent_value must be a child node")
    new_parent_index = tf.gather(origin_parent_value.parent_index,
                                 origin_value.parent_index)
    return prensor.ChildNodeTensor(new_parent_index, self.is_repeated)

  def calculation_is_identity(self) -> bool:
    return False

  def calculation_equal(self, expr: expression.Expression) -> bool:
    return isinstance(expr, PromoteChildExpression)

  def _get_child_impl(self,
                      field_name: path.Step) -> Optional[expression.Expression]:
    return self._origin.get_child(field_name)

  def known_field_names(self) -> FrozenSet[path.Step]:
    return self._origin.known_field_names()


def _lifecycle_stage_number(a) -> int:
  """Return a number indicating the quality of the lifecycle stage.

  When there is more than one input field, the minimum lifecycle stage could be
  used.

  Args:
    a: an Optional[LifecycleStage]

  Returns:
    An integer that corresponds to the lifecycle stage of 'a'.
  """
  stages = [
      schema_pb2.LifecycleStage.DEPRECATED, schema_pb2.LifecycleStage.DISABLED,
      schema_pb2.LifecycleStage.PLANNED, schema_pb2.LifecycleStage.ALPHA,
      schema_pb2.LifecycleStage.DEBUG_ONLY, None,
      schema_pb2.LifecycleStage.UNKNOWN_STAGE, schema_pb2.LifecycleStage.BETA,
      schema_pb2.LifecycleStage.PRODUCTION
  ]
  return stages.index(a)


def _min_lifecycle_stage(a, b):
  """Get the minimum lifecycle stage.

  Args:
    a: an Optional[LifecycleStage]
    b: an Optional[LifecycleStage]

  Returns:
    the minimal lifecycle stage.
  """
  if _lifecycle_stage_number(b) < _lifecycle_stage_number(a):
    return b
  return a


def _feature_is_dense(feature: schema_pb2.Feature) -> bool:
  return (feature.presence.min_fraction == 1.0 and
          feature.value_count.HasField("min") and
          feature.value_count.HasField("max") and
          feature.value_count.min == feature.value_count.max)


def _copy_domain_info(origin: schema_pb2.Feature, dest: schema_pb2.Feature):
  """Copy the domain info."""
  one_of_field_name = origin.WhichOneof("domain_info")
  if one_of_field_name is None:
    return

  origin_field = getattr(origin, one_of_field_name)

  field_descriptor = origin.DESCRIPTOR.fields_by_name.get(one_of_field_name)
  if field_descriptor is None or field_descriptor.message_type is None:
    setattr(dest, one_of_field_name, origin_field)
  else:
    dest_field = getattr(dest, one_of_field_name)
    dest_field.CopyFrom(origin_field)


def _get_promote_schema_feature(original: Optional[schema_pb2.Feature],
                                parent: Optional[schema_pb2.Feature]
                               ) -> Optional[schema_pb2.Feature]:
  """Generate the schema feature for the field resulting from promote.

  Note that promote results in the exact same number of values.

  Note that min_count is never propagated.

  Args:
    original: the original feature
    parent: the parent feature

  Returns:
    the schema of the new field.
  """
  if original is None or parent is None:
    return None
  result = schema_pb2.Feature()
  result.lifecycle_stage = _min_lifecycle_stage(original.lifecycle_stage,
                                                parent.lifecycle_stage)
  result.type = original.type
  if original.HasField("distribution_constraints"):
    result.distribution_constraints.CopyFrom(original.distribution_constraints)
  _copy_domain_info(original, result)

  if _feature_is_dense(parent):
    parent_size = parent.value_count.min
    if original.value_count.HasField("min"):
      result.value_count.min = parent_size * original.value_count.min
    if original.value_count.HasField("max"):
      result.value_count.max = parent_size * original.value_count.max
    if original.presence.HasField("min_fraction"):
      if original.presence.min_fraction == 1:
        result.presence.min_fraction = 1
      else:
        result.presence.min_fraction = (
            original.presence.min_fraction / parent_size)
    if original.presence.HasField("min_count"):
      # If the parent is dense then the count can
      # be reduced by the number of children.
      # E.g. {{"a"},{"b"}},{{"c"},{"d"}},{{"e"},{"f"}}
      # with a count of 6, with a parent size of 2 becomes:
      # can become {"a","b"}, {"c", "d"}, {"e", "f"}
      # which has a count of 3.
      result.presence.min_count = original.presence.min_count // parent_size
  return result


def _promote_impl(root: expression.Expression, p: path.Path,
                  new_field_name: path.Step
                 ) -> Tuple[expression.Expression, path.Path]:
  """Promotes a path to be a child of its grandparent, and gives it a name.

  Args:
    root: The root expression.
    p: The path to promote. This can be the path to a leaf or child node.
    new_field_name: The name of the promoted field.

  Returns:
    An _AddPathsExpression that wraps a PromoteExpression.
  """
  if len(p) < 2:
    raise ValueError("Cannot do a promotion beyond the root: {}".format(str(p)))
  parent_path = p.get_parent()
  grandparent_path = parent_path.get_parent()

  p_expression = root.get_descendant_or_error(p)
  new_path = grandparent_path.get_child(new_field_name)

  if p_expression.is_leaf:
    promote_expression_factory = PromoteExpression
  else:
    promote_expression_factory = PromoteChildExpression

  return expression_add.add_paths(
      root, {
          new_path:
              promote_expression_factory(
                  p_expression, root.get_descendant_or_error(parent_path))
      }), new_path


def promote_anonymous(root: expression.Expression,
                      p: path.Path) -> Tuple[expression.Expression, path.Path]:
  """Promote a path to be a new anonymous child of its grandparent."""
  return _promote_impl(root, p, path.get_anonymous_field())


def promote(root: expression.Expression, p: path.Path,
            new_field_name: path.Step) -> expression.Expression:
  """Promote a path to be a child of its grandparent, and give it a name."""
  return _promote_impl(root, p, new_field_name)[0]
