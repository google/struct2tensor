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

"""Tests for struct2tensor.prensor."""

from struct2tensor import calculate_options
from struct2tensor import path
from struct2tensor import prensor
from struct2tensor.test import prensor_test_util
import tensorflow as tf

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import

_OPTIONS_TO_TEST = [
    calculate_options.get_default_options(),
    calculate_options.get_options_with_minimal_checks()
]


@test_util.run_all_in_graph_and_eager_modes
class PrensorTest(tf.test.TestCase):

  def _assert_prensor_equals(self, lhs, rhs):
    if isinstance(lhs.node, prensor.RootNodeTensor):
      self.assertIsInstance(rhs.node, prensor.RootNodeTensor)
      self.assertIs(lhs.node.size, rhs.node.size)
    elif isinstance(lhs.node, prensor.ChildNodeTensor):
      self.assertIsInstance(rhs.node, prensor.ChildNodeTensor)
      self.assertIs(lhs.node.parent_index, rhs.node.parent_index)
      self.assertEqual(lhs.node.is_repeated, rhs.node.is_repeated)
    else:
      self.assertIsInstance(rhs.node, prensor.LeafNodeTensor)
      self.assertIs(lhs.node.parent_index, rhs.node.parent_index)
      self.assertIs(lhs.node.values, rhs.node.values)
      self.assertEqual(lhs.node.is_repeated, rhs.node.is_repeated)

    self.assertEqual(len(lhs.get_children()), len(rhs.get_children()))
    for (l_child_step, l_child), (r_child_step, r_child) in zip(
        lhs.get_children().items(), rhs.get_children().items()):
      self.assertEqual(l_child_step, r_child_step)
      self._assert_prensor_equals(l_child, r_child)

  def test_prensor_children_ordered(self):
    def _recursively_check_sorted(p):
      self.assertEqual(list(p.get_children().keys()),
                       sorted(p.get_children().keys()))
      for c in p.get_children().values():
        _recursively_check_sorted(c)

    for pren in [
        prensor_test_util.create_nested_prensor(),
        prensor_test_util.create_big_prensor(),
        prensor_test_util.create_deep_prensor()
    ]:
      _recursively_check_sorted(pren)

    p = prensor.create_prensor_from_descendant_nodes({
        path.Path([]):
            prensor_test_util.create_root_node(1),
        path.Path(["d"]):
            prensor_test_util.create_optional_leaf_node([0], [True]),
        path.Path(["c"]):
            prensor_test_util.create_optional_leaf_node([0], [True]),
        path.Path(["b"]):
            prensor_test_util.create_optional_leaf_node([0], [True]),
        path.Path(["a"]):
            prensor_test_util.create_optional_leaf_node([0], [True]),
    })
    self.assertEqual(["a", "b", "c", "d"], list(p.get_children().keys()))

  def test_prensor_is_composite_tensor(self):
    for pren in [
        prensor_test_util.create_nested_prensor(),
        prensor_test_util.create_big_prensor(),
        prensor_test_util.create_deep_prensor()
    ]:
      flattened_tensors = tf.nest.flatten(pren, expand_composites=True)
      self.assertIsInstance(flattened_tensors, list)
      for t in flattened_tensors:
        self.assertIsInstance(t, tf.Tensor)
      packed_pren = tf.nest.pack_sequence_as(
          pren, flattened_tensors, expand_composites=True)
      self._assert_prensor_equals(pren, packed_pren)

  def test_prensor_to_ragged_tensors(self):
    for options in _OPTIONS_TO_TEST:
      pren = prensor_test_util.create_nested_prensor()
      ragged_tensor_map = pren.get_ragged_tensors(options=options)
      string_tensor_map = {str(k): v for k, v in ragged_tensor_map.items()}
      string_np_map = self.evaluate(string_tensor_map)
      self.assertAllEqual(string_np_map["doc.bar"].to_list(),
                          [[[b"a"]], [[b"b", b"c"], [b"d"]], []])

      self.assertAllEqual(string_np_map["doc.keep_me"].to_list(),
                          [[[False]], [[True], []], []])
      self.assertAllEqual(string_np_map["user.friends"].to_list(),
                          [[[b"a"]], [[b"b", b"c"], [b"d"]], [[b"e"]]])

  def test_prensor_to_ragged_tensor(self):
    for options in _OPTIONS_TO_TEST:
      pren = prensor_test_util.create_nested_prensor()
      ragged_tensor = pren.get_ragged_tensor(
          path.create_path("doc.bar"), options)
      self.assertAllEqual(ragged_tensor, [[[b"a"]], [[b"b", b"c"], [b"d"]], []])

  def test_prensor_to_sparse_tensors(self):
    for options in _OPTIONS_TO_TEST:
      pren = prensor_test_util.create_nested_prensor()
      sparse_tensor_map = pren.get_sparse_tensors(options=options)
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

  def test_prensor_to_sparse_tensor(self):
    for options in _OPTIONS_TO_TEST:
      pren = prensor_test_util.create_simple_prensor()
      sparse_tensor = pren.get_sparse_tensor(
          path.create_path("foo"), options=options)
      self.assertAllEqual(sparse_tensor.indices, [[0], [1], [2]])
      self.assertAllEqual(sparse_tensor.dense_shape, [3])
      self.assertAllEqual(sparse_tensor.values, [9, 8, 7])

  def test_get_leaf_node_paths(self):
    """Tests get_sparse_tensors on a deep expression."""
    expression = prensor_test_util.create_nested_prensor()
    leaf_node_paths = prensor._get_leaf_node_paths(expression)
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
    for options in _OPTIONS_TO_TEST:
      expression = prensor_test_util.create_nested_prensor()
      sparse_tensor_map = prensor._get_sparse_tensors(expression, options)
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
    for options in _OPTIONS_TO_TEST:
      expression = prensor_test_util.create_simple_prensor()
      sparse_tensor_map = prensor._get_sparse_tensors(expression, options)
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
    sparse_tensor = prensor._get_sparse_tensor(expression,
                                               path.create_path("foo"))
    self.assertAllEqual(sparse_tensor.indices, [[0], [1], [2]])
    self.assertAllEqual(sparse_tensor.dense_shape, [3])
    self.assertAllEqual(sparse_tensor.values, [9, 8, 7])

  def test_get_sparse_tensors_simple_dense(self):
    """Tests get_sparse_tensors on a deep expression."""
    for options in _OPTIONS_TO_TEST:
      expression = prensor_test_util.create_simple_prensor()
      sparse_tensor_map = prensor._get_sparse_tensors(expression, options)
      string_tensor_map = {
          str(k): tf.sparse.to_dense(v) for k, v in sparse_tensor_map.items()
      }

      self.assertAllEqual(string_tensor_map["foo"], [9, 8, 7])
      self.assertAllEqual(string_tensor_map["foorepeated"],
                          [[9, 0], [8, 7], [6, 0]])


  def test_broken_ragged_tensors_no_check(self):
    """Make sure that it doesn't crash. The result is undefined."""
    expression = prensor_test_util.create_broken_prensor()
    ragged_tensor_map = prensor._get_ragged_tensors(
        expression, calculate_options.get_options_with_minimal_checks())
    string_tensor_map = {str(k): v for k, v in ragged_tensor_map.items()}
    self.evaluate(string_tensor_map)

  # Okay, need to break this apart to handle the V1/V2 issues.
  def test_get_ragged_tensors(self):
    """Tests get_ragged_tensors on a deep expression."""
    for options in _OPTIONS_TO_TEST:
      expression = prensor_test_util.create_nested_prensor()
      ragged_tensor_map = prensor._get_ragged_tensors(expression, options)
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
    for options in _OPTIONS_TO_TEST:
      expression = prensor_test_util.create_nested_prensor()
      ragged_tensor = prensor._get_ragged_tensor(expression,
                                                 path.create_path("doc.bar"),
                                                 options)
      self.assertAllEqual(ragged_tensor, [[[b"a"]], [[b"b", b"c"], [b"d"]], []])

# The following are only available post TF 1.14.

if __name__ == "__main__":
  tf.test.main()
