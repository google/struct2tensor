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
"""Tests for struct2tensor.promote."""

import tensorflow as tf
from absl.testing import absltest
from tensorflow.python.framework import (
    test_util,  # pylint: disable=g-direct-tensorflow-import
)

from struct2tensor import create_expression, path
from struct2tensor.expression_impl import index
from struct2tensor.test import expression_test_util, prensor_test_util


class IndexTest(absltest.TestCase):
    def test_get_positional_index(self):
        expr = create_expression.create_expression_from_prensor(
            prensor_test_util.create_nested_prensor()
        )
        new_root, new_path = index.get_positional_index(
            expr, path.Path(["user", "friends"]), path.get_anonymous_field()
        )
        new_field = new_root.get_descendant_or_error(new_path)
        self.assertTrue(new_field.is_repeated)
        self.assertEqual(new_field.type, tf.int64)
        self.assertTrue(new_field.is_leaf)
        self.assertTrue(new_field.calculation_equal(new_field))
        self.assertFalse(new_field.calculation_equal(expr))
        leaf_node = expression_test_util.calculate_value_slowly(new_field)
        self.assertEqual(leaf_node.values.dtype, tf.int64)
        self.assertEqual(new_field.known_field_names(), frozenset())

    def test_get_index_from_end(self):
        expr = create_expression.create_expression_from_prensor(
            prensor_test_util.create_nested_prensor()
        )
        new_root, new_path = index.get_index_from_end(
            expr, path.Path(["user", "friends"]), path.get_anonymous_field()
        )
        new_field = new_root.get_descendant_or_error(new_path)
        self.assertTrue(new_field.is_repeated)
        self.assertEqual(new_field.type, tf.int64)
        self.assertTrue(new_field.is_leaf)
        self.assertTrue(new_field.calculation_equal(new_field))
        self.assertFalse(new_field.calculation_equal(expr))
        leaf_node = expression_test_util.calculate_value_slowly(new_field)
        self.assertEqual(leaf_node.values.dtype, tf.int64)
        self.assertEqual(new_field.known_field_names(), frozenset())


@test_util.run_all_in_graph_and_eager_modes
class GetIndexValuesTest(tf.test.TestCase):
    def test_get_positional_index_calculate(self):
        expr = create_expression.create_expression_from_prensor(
            prensor_test_util.create_nested_prensor()
        )
        new_root, new_path = index.get_positional_index(
            expr, path.Path(["user", "friends"]), path.get_anonymous_field()
        )
        new_field = new_root.get_descendant_or_error(new_path)
        leaf_node = expression_test_util.calculate_value_slowly(new_field)
        self.assertAllEqual(leaf_node.parent_index, [0, 1, 1, 2, 3])
        self.assertAllEqual(leaf_node.values, [0, 0, 1, 0, 0])

    def test_get_index_from_end_calculate(self):
        expr = create_expression.create_expression_from_prensor(
            prensor_test_util.create_nested_prensor()
        )
        new_root, new_path = index.get_index_from_end(
            expr, path.Path(["user", "friends"]), path.get_anonymous_field()
        )
        print(f"test_get_index_from_end_calculate: new_path: {new_path}")
        new_field = new_root.get_descendant_or_error(new_path)
        print(f"test_get_index_from_end_calculate: new_field: {str(new_field)}")

        leaf_node = expression_test_util.calculate_value_slowly(new_field)
        print(f"test_get_index_from_end_calculate: leaf_node: {str(leaf_node)}")

        self.assertAllEqual(leaf_node.parent_index, [0, 1, 1, 2, 3])
        self.assertAllEqual(leaf_node.values, [-1, -2, -1, -1, -1])


if __name__ == "__main__":
    absltest.main()
