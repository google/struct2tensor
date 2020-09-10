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

from absl.testing import absltest
from struct2tensor import calculate_options
from struct2tensor import path
from struct2tensor import prensor_util
from struct2tensor.test import prensor_test_util
import tensorflow as tf

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import

options_to_test = [
    calculate_options.get_default_options(),
    calculate_options.get_options_with_minimal_checks()
]


@test_util.run_all_in_graph_and_eager_modes
class PrensorUtilTest(tf.test.TestCase):

  def test_get_leaf_node_paths(self):
    """Tests get_sparse_tensors on a deep expression."""
    expression = prensor_test_util.create_nested_prensor()
    leaf_node_paths = prensor_util._get_leaf_node_paths(expression)
    self.assertIn(path.Path(["doc", "bar"]), leaf_node_paths)
    self.assertIn(path.Path(["user", "friends"]), leaf_node_paths)
    self.assertIn(path.Path(["doc", "keep_me"]), leaf_node_paths)

  def test_is_leaf(self):
    """Tests get_sparse_tensors on a deep expression."""
    expression = prensor_test_util.create_nested_prensor()
    self.assertTrue(
        expression.get_descendant_or_error(path.Path(["doc", "bar"])).is_leaf)
    self.assertFalse(
        expression.get_descendant_or_error(path.Path(["doc"])).is_leaf)
    self.assertFalse(expression.get_descendant_or_error(path.Path([])).is_leaf)
    self.assertTrue(
        expression.get_descendant_or_error(path.Path(["user",
                                                      "friends"])).is_leaf)
    self.assertTrue(
        expression.get_descendant_or_error(path.Path(["doc",
                                                      "keep_me"])).is_leaf)

  def test_get_sparse_tensors(self):
    """Tests get_sparse_tensors on a deep expression."""
    for options in options_to_test:
      expression = prensor_test_util.create_nested_prensor()
      sparse_tensor_map = prensor_util.get_sparse_tensors(expression, options)
      string_tensor_map = {str(k): v for k, v in sparse_tensor_map.items()}

      self.assertAllEqual(string_tensor_map["doc.bar"].indices,
                          [[0, 0, 0], [1, 0, 0], [1, 0, 1], [1, 1, 0]])
      self.assertAllEqual(string_tensor_map["doc.bar"].values,
                          [b"a", b"b", b"c", b"d"])
      self.assertAllEqual(string_tensor_map["doc.keep_me"].indices,
                          [[0, 0], [1, 0]])
      self.assertAllEqual(string_tensor_map["doc.keep_me"].values,
                          [False, True])

      self.assertAllEqual(
          string_tensor_map["user.friends"].indices,
          [[0, 0, 0], [1, 0, 0], [1, 0, 1], [1, 1, 0], [2, 0, 0]])
      self.assertAllEqual(string_tensor_map["user.friends"].values,
                          [b"a", b"b", b"c", b"d", b"e"])

  def test_get_sparse_tensors_simple(self):
    """Tests get_sparse_tensors on a deep expression."""
    for options in options_to_test:
      expression = prensor_test_util.create_simple_prensor()
      sparse_tensor_map = prensor_util.get_sparse_tensors(expression, options)
      string_tensor_map = {str(k): v for k, v in sparse_tensor_map.items()}
      self.assertAllEqual(string_tensor_map["foo"].indices, [[0], [1], [2]])
      self.assertAllEqual(string_tensor_map["foo"].dense_shape, [3])

      self.assertAllEqual(string_tensor_map["foo"].values, [9, 8, 7])
      self.assertAllEqual(string_tensor_map["foorepeated"].indices,
                          [[0, 0], [1, 0], [1, 1], [2, 0]])
      self.assertAllEqual(string_tensor_map["foorepeated"].values, [9, 8, 7, 6])
      self.assertAllEqual(string_tensor_map["foorepeated"].dense_shape, [3, 2])

  def test_get_sparse_tensor(self):
    expression = prensor_test_util.create_simple_prensor()
    sparse_tensor = prensor_util.get_sparse_tensor(expression,
                                                   path.create_path("foo"))
    self.assertAllEqual(sparse_tensor.indices, [[0], [1], [2]])
    self.assertAllEqual(sparse_tensor.dense_shape, [3])
    self.assertAllEqual(sparse_tensor.values, [9, 8, 7])

  def test_get_sparse_tensors_simple_dense(self):
    """Tests get_sparse_tensors on a deep expression."""
    for options in options_to_test:
      expression = prensor_test_util.create_simple_prensor()
      sparse_tensor_map = prensor_util.get_sparse_tensors(expression, options)
      string_tensor_map = {
          str(k): tf.sparse.to_dense(v)
          for k, v in sparse_tensor_map.items()
      }

      self.assertAllEqual(string_tensor_map["foo"], [9, 8, 7])
      self.assertAllEqual(string_tensor_map["foorepeated"],
                          [[9, 0], [8, 7], [6, 0]])


  def test_broken_ragged_tensors_no_check(self):
    """Make sure that it doesn't crash. The result is undefined."""
    expression = prensor_test_util.create_broken_prensor()
    ragged_tensor_map = prensor_util.get_ragged_tensors(
        expression, calculate_options.get_options_with_minimal_checks())
    string_tensor_map = {str(k): v for k, v in ragged_tensor_map.items()}
    self.evaluate(string_tensor_map)

  # Okay, need to break this apart to handle the V1/V2 issues.
  def test_get_ragged_tensors(self):
    """Tests get_ragged_tensors on a deep expression."""
    for options in options_to_test:
      expression = prensor_test_util.create_nested_prensor()
      ragged_tensor_map = prensor_util.get_ragged_tensors(expression, options)
      string_tensor_map = {str(k): v for k, v in ragged_tensor_map.items()}
      string_np_map = self.evaluate(string_tensor_map)
      self.assertAllEqual(string_np_map["doc.bar"].to_list(),
                          [[[b"a"]], [[b"b", b"c"], [b"d"]], []])

      self.assertAllEqual(string_np_map["doc.keep_me"].to_list(),
                          [[[False]], [[True], []], []])
      self.assertAllEqual(string_np_map["user.friends"].to_list(),
                          [[[b"a"]], [[b"b", b"c"], [b"d"]], [[b"e"]]])

  def test_get_ragged_tensor(self):
    """Tests get_ragged_tensor on a deep field."""
    for options in options_to_test:
      expression = prensor_test_util.create_nested_prensor()
      ragged_tensor = prensor_util.get_ragged_tensor(
          expression, path.create_path("doc.bar"), options)
      self.assertAllEqual(ragged_tensor,
                          [[[b"a"]], [[b"b", b"c"], [b"d"]], []])


if __name__ == "__main__":
  absltest.main()
