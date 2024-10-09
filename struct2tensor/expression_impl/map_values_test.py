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
"""Tests for struct2tensor.broadcast."""

import tensorflow as tf
from absl.testing import absltest
from tensorflow.python.framework import (
  test_util,  # pylint: disable=g-direct-tensorflow-import
)

from struct2tensor import create_expression, path, prensor
from struct2tensor.expression_impl import map_values
from struct2tensor.test import expression_test_util, prensor_test_util


@test_util.run_all_in_graph_and_eager_modes
class MapValuesTest(tf.test.TestCase):

  def test_map_values_anonymous(self):
    expr = create_expression.create_expression_from_prensor(
        prensor_test_util.create_simple_prensor())

    new_root, p = map_values.map_values_anonymous(expr, path.Path(["foo"]),
                                                  lambda x: x * 2, tf.int64)

    leaf_node = expression_test_util.calculate_value_slowly(
        new_root.get_descendant_or_error(p))
    self.assertAllEqual(leaf_node.parent_index, [0, 1, 2])
    self.assertAllEqual(leaf_node.values, [18, 16, 14])

  def test_map_values(self):
    expr = create_expression.create_expression_from_prensor(
        prensor_test_util.create_simple_prensor())

    new_root = map_values.map_values(expr, path.Path(["foo"]), lambda x: x * 2,
                                     tf.int64, "foo_doubled")

    leaf_node = expression_test_util.calculate_value_slowly(
        new_root.get_descendant_or_error(path.Path(["foo_doubled"])))
    self.assertAllEqual(leaf_node.parent_index, [0, 1, 2])
    self.assertAllEqual(leaf_node.values, [18, 16, 14])

  def test_map_many_values(self):
    expr = create_expression.create_expression_from_prensor(
        prensor.create_prensor_from_descendant_nodes({
            path.Path([]):
                prensor_test_util.create_root_node(3),
            path.Path(["foo"]):
                prensor_test_util.create_optional_leaf_node([0, 2, 3],
                                                            [9, 8, 7]),
            path.Path(["bar"]):
                prensor_test_util.create_optional_leaf_node([0, 2, 3],
                                                            [10, 20, 30])
        }))

    new_root, p = map_values.map_many_values(expr, path.Path([]),
                                             ["foo", "bar"], lambda x, y: x + y,
                                             tf.int64, "new_field")

    leaf_node = expression_test_util.calculate_value_slowly(
        new_root.get_descendant_or_error(p))
    self.assertAllEqual(leaf_node.parent_index, [0, 2, 3])
    self.assertAllEqual(leaf_node.values, [19, 28, 37])


if __name__ == "__main__":
  absltest.main()
