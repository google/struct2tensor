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
"""Tests for struct2tensor.expression_impl.apply_schema."""

import copy

from absl.testing import absltest
from tensorflow_metadata.proto.v0 import schema_pb2

from struct2tensor import create_expression, path
from struct2tensor.test import prensor_test_util


def _features_as_map(feature_list):
  return {feature.name: feature for feature in feature_list}


class SchemaUtilTest(absltest.TestCase):

  def test_apply_schema(self):
    expr = create_expression.create_expression_from_prensor(
        prensor_test_util.create_big_prensor())
    input_schema = prensor_test_util.create_big_prensor_schema()
    expected_input_schema = copy.deepcopy(input_schema)
    expr2 = expr.apply_schema(input_schema)
    foo_expr = expr2.get_descendant(path.Path(["foo"]))
    self.assertIsNotNone(foo_expr)
    foorepeated_expr = expr2.get_descendant(path.Path(["foorepeated"]))
    self.assertIsNotNone(foorepeated_expr)
    doc_bar_expr = expr2.get_descendant(path.Path(["doc", "bar"]))
    self.assertIsNotNone(doc_bar_expr)

    # Test that a domain already in the feature is maintained.
    self.assertEqual(foo_expr.schema_feature.int_domain.max, 10)
    # Test that an int_domain specified at the schema level is inserted
    # correctly.
    self.assertEqual(foorepeated_expr.schema_feature.int_domain.max, 10)

    # Test that a string_domain specified at the schema level is inserted
    # correctly.
    self.assertEqual(doc_bar_expr.schema_feature.string_domain.value[0], "a")
    self.assertIsNotNone(expr2.get_descendant(path.Path(["user", "friends"])))
    self.assertIsNotNone(expr2.get_descendant(path.Path(["doc", "keep_me"])))

    # Ensure the input schema wasn't mutated.
    self.assertEqual(input_schema, expected_input_schema)

  def test_apply_empty_schema(self):
    """Test that applying an empty schema does not filter out paths."""
    expr = create_expression.create_expression_from_prensor(
        prensor_test_util.create_big_prensor())
    expr2 = expr.apply_schema(schema_pb2.Schema())
    foo_expr = expr2.get_descendant(path.Path(["foo"]))
    self.assertIsNotNone(foo_expr)
    foorepeated_expr = expr2.get_descendant(path.Path(["foorepeated"]))
    self.assertIsNotNone(foorepeated_expr)
    doc_bar_expr = expr2.get_descendant(path.Path(["doc", "bar"]))
    self.assertIsNotNone(doc_bar_expr)
    known_field_names = expr2.known_field_names()
    self.assertIn("doc", known_field_names)
    self.assertIn("foo", known_field_names)
    self.assertIn("foorepeated", known_field_names)
    self.assertIn("user", known_field_names)

  def test_get_schema(self):
    """Integration test between get_schema and apply_schema."""
    expr = create_expression.create_expression_from_prensor(
        prensor_test_util.create_big_prensor())
    expr2 = expr.apply_schema(prensor_test_util.create_big_prensor_schema())
    schema_result = expr2.get_schema()
    feature_map = _features_as_map(schema_result.feature)
    self.assertIn("foo", feature_map)
    # Test that a domain already in the feature is maintained.
    self.assertEqual(feature_map["foo"].int_domain.max, 10)
    # Test that an int_domain specified at the schema level is inserted
    # correctly.
    self.assertEqual(feature_map["foorepeated"].int_domain.max, 10)

  def test_get_schema_lenient_names(self):
    """Test that apply_schema preserves origin expression leniency."""
    expr = create_expression.create_expression_from_prensor(
        prensor_test_util.create_nested_prensor_with_lenient_field_names(),
        validate_step_format=False,
    )
    expr2 = expr.apply_schema(schema_pb2.Schema())
    self.assertFalse(expr2.validate_step_format)
    self.assertLen(expr2.get_known_descendants(), 6)


if __name__ == "__main__":
  absltest.main()
