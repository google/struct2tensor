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
"""Apply a schema to an expression.

A tensorflow metadata schema (TODO(martinz): link) represents more
detailed information about the data: specifically, it presents domain
information (e.g., not just integers, but integers between 0 and 10), and more
detailed structural information (e.g., this field occurs in at least 70% of its
parents, and when it occurs, it shows up 5 to 7 times).

Applying a schema attaches a tensorflow metadata schema to an expression:
namely, it aligns the features in the schema with the expression's children by
name (possibly recursively).

After applying a schema to an expression, one can use promote, broadcast, et
cetera, and the schema for new expressions will be inferred. If you write a
custom expression, you can write code that determines the schema information of
the result.

To get the schema back, call get_schema().

This does not filter out fields not in the schema.


my_expr = ...
my_schema = ...schema here...
my_new_schema = my_expr.apply_schema(my_schema).get_schema()
my_new_schema has semantically identical information on the fields as my_schema.

TODO(martinz): Add utilities to:
1. Get the (non-deprecated) paths from a schema.
2. Check if any paths in the schema are not in the expression.
3. Check if any paths in the expression are not in the schema.
4. Project the expression to paths in the schema.

"""

import abc
from typing import FrozenSet, Optional, Sequence

from struct2tensor import calculate_options
from struct2tensor import expression
from struct2tensor import path
from struct2tensor import prensor

from tensorflow_metadata.proto.v0 import schema_pb2


def apply_schema(expr: expression.Expression,
                 schema: schema_pb2.Schema) -> expression.Expression:
  schema_copy = schema_pb2.Schema()
  schema_copy.CopyFrom(schema)
  for x in schema_copy.feature:
    _normalize_feature(x, schema_copy)
  return _SchemaExpression(expr, schema_copy.feature, None)


def _normalize_feature(feature: schema_pb2.Feature,
                       schema: schema_pb2.Schema) -> None:
  """Make each feature self-contained.

  If the feature references a global domain, copy the global domain locally.
  Also do this for any child features.

  Note: the name of the domain is retained, so if we want to, we could attempt
  to "unnormalize" the feature, recreating global domains.

  Args:
    feature: feature to modify in place.
    schema: schema containing any global domains.
  """

  if feature.HasField("struct_domain"):
    for x in feature.struct_domain.feature:
      _normalize_feature(x, schema)
  if feature.HasField("domain"):
    for string_domain in schema.string_domain:
      if string_domain.name == feature.domain:
        feature.string_domain.CopyFrom(string_domain)
        return
    for int_domain in schema.int_domain:
      if int_domain.name == feature.domain:
        feature.int_domain.CopyFrom(int_domain)
        return
    for float_domain in schema.float_domain:
      if float_domain.name == feature.domain:
        feature.float_domain.CopyFrom(float_domain)
        return
    raise ValueError("Did not find domain {} in schema {}".format(
        feature.domain, schema))


def _clean_feature(feature: schema_pb2.Feature) -> schema_pb2.Feature:
  """Remove name and all children of a feature (if any exist), returning a copy.

  Args:
    feature: input feature

  Returns:
    cleaned feature
  """
  copy = schema_pb2.Feature()
  copy.CopyFrom(feature)
  copy.ClearField("name")
  if copy.HasField("struct_domain"):
    del copy.struct_domain.feature[:]
  return copy


def _apply_feature(original_child: expression.Expression,
                   feature: schema_pb2.Feature):
  """Apply a feature to an expression. Feature should be "unclean"."""
  feature_copy = [x for x in feature.struct_domain.feature
                 ] if feature.HasField("struct_domain") else []
  return _SchemaExpression(original_child, feature_copy,
                           _clean_feature(feature))


class _SchemaExpression(expression.Expression, metaclass=abc.ABCMeta):
  """An expression represents the application of a schema."""

  def __init__(
      self,
      original: expression.Expression,
      child_features: Sequence[schema_pb2.Feature],
      schema_feature: Optional[schema_pb2.Feature],
  ):
    """Create a new _SchemaExpression.

    Args:
      original: the original expression.
      child_features: the uncleaned Feature protos for its children.
      schema_feature: the optional cleaned feature for this node.
    """
    super().__init__(
        original.is_repeated,
        original.type,
        schema_feature=schema_feature,
        validate_step_format=original.validate_step_format,
    )
    self._original = original
    self._child_features = child_features

  def get_source_expressions(self) -> Sequence[expression.Expression]:
    return [self._original]

  def calculate(self, source_tensors: Sequence[prensor.NodeTensor],  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
                destinations: Sequence[expression.Expression],
                options: calculate_options.Options) -> prensor.NodeTensor:
    del destinations, options
    [original_result] = source_tensors
    return original_result

  def calculation_is_identity(self) -> bool:
    return True

  def calculation_equal(self, expr: expression.Expression) -> bool:
    return expr.calculation_is_identity()

  def _find_feature_proto(self, field_name: path.Step
                         ) -> Optional[schema_pb2.Feature]:
    for feature in self._child_features:
      if feature.name == field_name:
        return feature
    return None

  def _get_child_impl(self,
                      field_name: path.Step) -> Optional[expression.Expression]:
    original_child = self._original.get_child(field_name)
    if original_child is None:
      return None
    feature_proto = self._find_feature_proto(field_name)
    if feature_proto is None:
      return original_child
    return _apply_feature(original_child, feature_proto)

  def known_field_names(self) -> FrozenSet[path.Step]:
    result = set(self._original.known_field_names())
    for feature_proto in self._child_features:
      field_name = str(feature_proto.name)
      associated_child = self.get_child(field_name)
      if associated_child is not None:
        result.add(field_name)
    return frozenset(result)
