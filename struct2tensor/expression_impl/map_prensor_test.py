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
"""Tests for struct2tensor.map_prensor."""

import tensorflow as tf
from absl.testing import absltest
from tensorflow.python.framework import (
  test_util,  # pylint: disable=g-direct-tensorflow-import
)

from struct2tensor import calculate_options, create_expression, path, prensor
from struct2tensor.expression_impl import map_prensor
from struct2tensor.test import expression_test_util, prensor_test_util


def _create_one_value_prensor():
  """Creates a prensor expression representing a list of flat protocol buffers.

  Returns:
    a RootPrensor representing:
    {}
    {foo:8}
    {}
  """
  return prensor.create_prensor_from_descendant_nodes({
      path.Path([]): prensor_test_util.create_root_node(3),
      path.Path(["foo"]): prensor_test_util.create_optional_leaf_node([1], [8])
  })


options_to_test = [
    calculate_options.get_default_options(),
    calculate_options.get_options_with_minimal_checks()
]


@test_util.run_all_in_graph_and_eager_modes
class MapPrensorTest(tf.test.TestCase):

  def _test_assert_raises(self, test_runner):
    with self.assertRaises(tf.errors.InvalidArgumentError):
      test_runner(calculate_options.get_default_options())
    test_runner(calculate_options.get_options_with_minimal_checks())

  def test_map_sparse_tensor(self):
    expr = create_expression.create_expression_from_prensor(
        prensor_test_util.create_simple_prensor())

    new_root = map_prensor.map_sparse_tensor(expr, path.Path([]),
                                             [path.Path(["foo"])],
                                             lambda x: x * 2, False, tf.int32,
                                             "foo_doubled")

    leaf_node = expression_test_util.calculate_value_slowly(
        new_root.get_descendant_or_error(path.Path(["foo_doubled"])))
    self.assertAllEqual(leaf_node.parent_index, [0, 1, 2])
    self.assertAllEqual(leaf_node.values, [18, 16, 14])

  def test_map_sparse_tensor_one_output(self):
    for options in options_to_test:
      expr = create_expression.create_expression_from_prensor(
          _create_one_value_prensor())

      new_root = map_prensor.map_sparse_tensor(expr, path.Path(
          []), [path.Path(["foo"])], lambda x: x * 2, False, tf.int32,
                                               "foo_doubled")

      leaf_node = expression_test_util.calculate_value_slowly(
          new_root.get_descendant_or_error(path.Path(["foo_doubled"])),
          options=options)
      self.assertAllEqual(leaf_node.parent_index, [1])
      self.assertAllEqual(leaf_node.values, [16])

  def test_map_sparse_tensor_is_repeated(self):
    for options in options_to_test:
      expr = create_expression.create_expression_from_prensor(
          prensor_test_util.create_simple_prensor())

      new_root = map_prensor.map_sparse_tensor(expr, path.Path([]),
                                               [path.Path(["foorepeated"])],
                                               lambda x: x * 2, True, tf.int32,
                                               "foorepeated_doubled")

      leaf_node = expression_test_util.calculate_value_slowly(
          new_root.get_descendant_or_error(path.Path(["foorepeated_doubled"])),
          options=options)
      self.assertAllEqual(leaf_node.parent_index, [0, 1, 1, 2])
      self.assertAllEqual(leaf_node.values, [18, 16, 14, 12])

  def test_map_sparse_tensor_is_repeated_assert(self):

    def _test_runner(options):
      expr = create_expression.create_expression_from_prensor(
          prensor_test_util.create_simple_prensor())
      new_root = map_prensor.map_sparse_tensor(expr, path.Path([]),
                                               [path.Path(["foo"])],
                                               lambda x: x * 2, True, tf.int32,
                                               "foo_doubled")
      leaf_node = expression_test_util.calculate_value_slowly(
          new_root.get_descendant_or_error(path.Path(["foo_doubled"])),
          options=options)
      self.evaluate(leaf_node.parent_index)
      self.evaluate(leaf_node.values)

    self._test_assert_raises(_test_runner)

  def test_map_sparse_tensor_assert(self):
    def _test_runner(options):
      expr = create_expression.create_expression_from_prensor(
          prensor_test_util.create_simple_prensor())

      new_root = map_prensor.map_sparse_tensor(expr, path.Path([]),
                                               [path.Path(["foorepeated"])],
                                               lambda x: x * 2, False, tf.int32,
                                               "foorepeated_doubled")

      leaf_node = expression_test_util.calculate_value_slowly(
          new_root.get_descendant_or_error(path.Path(["foorepeated_doubled"])),
          options=options)
      self.evaluate(leaf_node.parent_index)
      self.evaluate(leaf_node.values)

    self._test_assert_raises(_test_runner)

  def test_map_sparse_tensor_assert_batch_size(self):
    def _test_runner(options):
      expr = create_expression.create_expression_from_prensor(
          prensor_test_util.create_simple_prensor())

      new_root = map_prensor.map_sparse_tensor(
          expr, path.Path([]), [path.Path(["foo"])],
          lambda x: tf.sparse.concat(0, [x, x]), False, tf.int32, "foo_concat")

      leaf_node = expression_test_util.calculate_value_slowly(
          new_root.get_descendant_or_error(path.Path(["foo_concat"])),
          options=options)
      self.evaluate(leaf_node.parent_index)
      self.evaluate(leaf_node.values)

    self._test_assert_raises(_test_runner)


  def test_skip_eager_map_ragged_tensor_repeated(self):
    # This fails in eager, with an inconsistency in the ragged tensor.
    if tf.executing_eagerly():
      return
    for options in options_to_test:
      expr = create_expression.create_expression_from_prensor(
          prensor_test_util.create_simple_prensor())
      new_root = map_prensor.map_ragged_tensor(expr, path.Path([]),
                                               [path.Path(["foorepeated"])],
                                               lambda x: x * 2, False, tf.int32,
                                               "foorepeated_doubled")
      leaf_node = expression_test_util.calculate_value_slowly(
          new_root.get_descendant_or_error(path.Path(["foorepeated_doubled"])),
          options=options)
      self.assertAllEqual(leaf_node.parent_index, [0, 1, 1, 2])
      self.assertAllEqual(leaf_node.values, [18, 16, 14, 12])


if __name__ == "__main__":
  absltest.main()
