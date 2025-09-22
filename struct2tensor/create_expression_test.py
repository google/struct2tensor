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
"""Tests for struct2tensor.create_expression."""

import tensorflow as tf
from absl.testing import absltest

from struct2tensor import create_expression, path
from struct2tensor.test import expression_test_util, prensor_test_util


class ExpressionTest(absltest.TestCase):
    def test_root_methods(self):
        """Test all the basic operations on the root of a prensor expression."""
        expression = prensor_test_util.create_simple_prensor()
        expr = create_expression.create_expression_from_prensor(expression)
        self.assertTrue(expr.is_repeated)
        self.assertIsNone(expr.type)
        self.assertFalse(expr.is_leaf)
        self.assertEqual(expr.get_source_expressions(), [])
        self.assertFalse(expr.calculation_is_identity())
        self.assertTrue(expr.calculation_equal(expr))
        root_node = expression_test_util.calculate_value_slowly(expr)
        self.assertEqual(root_node.size.dtype, tf.int64)
        self.assertEqual(expr.known_field_names(), frozenset(["foo", "foorepeated"]))

    def test_foo_node(self):
        """Test all the basic operations an optional leaf of an expression."""
        simple_prensor = prensor_test_util.create_simple_prensor()
        expr = create_expression.create_expression_from_prensor(simple_prensor)
        foo_expr = expr.get_child_or_error("foo")
        self.assertFalse(foo_expr.is_repeated)
        self.assertEqual(foo_expr.type, tf.int32)
        self.assertTrue(foo_expr.is_leaf)
        self.assertEqual(foo_expr.get_source_expressions(), [])
        self.assertFalse(foo_expr.calculation_is_identity())
        self.assertTrue(foo_expr.calculation_equal(foo_expr))
        self.assertFalse(foo_expr.calculation_equal(expr))
        leaf_node = expression_test_util.calculate_value_slowly(foo_expr)
        self.assertEqual(leaf_node.values.dtype, tf.int32)
        self.assertEqual(foo_expr.known_field_names(), frozenset())

    def test_foorepeated_node(self):
        """Test all the basic operations an repeated leaf of an expression."""
        expression = prensor_test_util.create_simple_prensor()
        expr = create_expression.create_expression_from_prensor(expression)
        foorepeated_expr = expr.get_child_or_error("foorepeated")
        self.assertTrue(foorepeated_expr.is_repeated)
        self.assertEqual(foorepeated_expr.type, tf.int32)
        self.assertTrue(foorepeated_expr.is_leaf)
        self.assertEqual(foorepeated_expr.get_source_expressions(), [])
        self.assertFalse(foorepeated_expr.calculation_is_identity())
        self.assertTrue(foorepeated_expr.calculation_equal(foorepeated_expr))
        self.assertFalse(foorepeated_expr.calculation_equal(expr))
        leaf_node = expression_test_util.calculate_value_slowly(foorepeated_expr)
        self.assertEqual(leaf_node.values.dtype, tf.int32)
        self.assertEqual(foorepeated_expr.known_field_names(), frozenset())

    def test_get_descendants(self):
        expression = prensor_test_util.create_nested_prensor()
        expr = create_expression.create_expression_from_prensor(expression)
        expr_user_friends = expr.get_descendant_or_error(path.Path(["user", "friends"]))
        self.assertIs(
            expr_user_friends,
            expr.get_child_or_error("user").get_child_or_error("friends"),
        )

    def test_get_known_descendants(self):
        expression = prensor_test_util.create_nested_prensor()
        expr = create_expression.create_expression_from_prensor(expression)
        expr_map = expr.get_known_descendants()
        self.assertIn(path.Path(["doc"]), expr_map)
        self.assertIn(path.Path(["doc", "bar"]), expr_map)
        self.assertIn(path.Path(["doc", "keep_me"]), expr_map)
        self.assertIn(path.Path(["user"]), expr_map)
        self.assertIn(path.Path(["user", "friends"]), expr_map)

    def test_lenient_formatting_get_known_descendants(self):
        expression = prensor_test_util.create_nested_prensor_with_lenient_field_names()
        expr = create_expression.create_expression_from_prensor(
            expression, validate_step_format=False
        )
        expr_map = expr.get_known_descendants()
        self.assertIn(path.Path(["doc"], validate_step_format=False), expr_map)
        self.assertIn(
            path.Path(["doc", "bar.baz"], validate_step_format=False), expr_map
        )
        self.assertIn(
            path.Path(["doc", "keep_me/x"], validate_step_format=False), expr_map
        )
        self.assertIn(path.Path(["user"], validate_step_format=False), expr_map)
        self.assertIn(
            path.Path(["user", "friends!:)"], validate_step_format=False), expr_map
        )

    def test_lenient_formatting_map_sparse_tensors(self):
        expression = prensor_test_util.create_nested_prensor_with_lenient_field_names()
        expr = create_expression.create_expression_from_prensor(
            expression, validate_step_format=False
        )
        expr_map = expr.map_sparse_tensors(
            "doc",
            ["bar.baz", "keep_me/x"],
            tf.compat.v1.sparse_add,
            False,
            tf.int64,
            "total",
        )
        self.assertIsNotNone(expr_map)

    def test_lenient_formatting_map_ragged_tensors(self):
        expression = prensor_test_util.create_nested_prensor_with_lenient_field_names()
        expr = create_expression.create_expression_from_prensor(
            expression, validate_step_format=False
        )
        expr_map = expr.map_ragged_tensors(
            "doc",
            ["bar.baz", "keep_me/x"],
            lambda x: x,
            False,
            tf.int64,
            "total",
        )
        self.assertIsNotNone(expr_map)

    def test_lenient_formatting_get_paths_with_schema(self):
        expression = prensor_test_util.create_nested_prensor_with_lenient_field_names()
        expr = create_expression.create_expression_from_prensor(
            expression, validate_step_format=False
        )
        paths = expr.get_paths_with_schema()
        self.assertLen(paths, 1)


if __name__ == "__main__":
    absltest.main()
