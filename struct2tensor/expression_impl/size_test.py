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

from absl.testing import absltest
from struct2tensor import create_expression
from struct2tensor import path
from struct2tensor.expression_impl import size
from struct2tensor.test import expression_test_util
from struct2tensor.test import prensor_test_util
import tensorflow as tf

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


@test_util.run_all_in_graph_and_eager_modes
class SizeTest(tf.test.TestCase):

  def test_size_anonymous(self):
    expr = create_expression.create_expression_from_prensor(
        prensor_test_util.create_big_prensor())
    new_root, new_path = size.size_anonymous(expr, path.Path(["doc", "bar"]))
    new_field = new_root.get_descendant_or_error(new_path)
    leaf_node = expression_test_util.calculate_value_slowly(new_field)
    self.assertAllEqual(leaf_node.parent_index, [0, 1, 2])
    self.assertAllEqual(leaf_node.values, [1, 2, 1])

  def test_size(self):
    expr = create_expression.create_expression_from_prensor(
        prensor_test_util.create_big_prensor())
    new_root = size.size(expr, path.Path(["doc", "bar"]), "result")
    new_field = new_root.get_descendant_or_error(path.Path(["doc", "result"]))
    leaf_node = expression_test_util.calculate_value_slowly(new_field)
    self.assertAllEqual(leaf_node.parent_index, [0, 1, 2])
    self.assertAllEqual(leaf_node.values, [1, 2, 1])

  def test_size_missing_value(self):
    expr = create_expression.create_expression_from_prensor(
        prensor_test_util.create_big_prensor())
    new_root = size.size(expr, path.Path(["doc", "keep_me"]), "result")
    new_field = new_root.get_descendant_or_error(path.Path(["doc", "result"]))
    leaf_node = expression_test_util.calculate_value_slowly(new_field)
    self.assertAllEqual(leaf_node.parent_index, [0, 1, 2])
    self.assertAllEqual(leaf_node.values, [1, 1, 0])

  def test_has(self):
    expr = create_expression.create_expression_from_prensor(
        prensor_test_util.create_big_prensor())
    new_root = size.has(expr, path.Path(["doc", "keep_me"]), "result")
    new_field = new_root.get_descendant_or_error(path.Path(["doc", "result"]))
    leaf_node = expression_test_util.calculate_value_slowly(new_field)
    self.assertAllEqual(leaf_node.parent_index, [0, 1, 2])
    self.assertAllEqual(leaf_node.values, [True, True, False])


if __name__ == "__main__":
  absltest.main()
