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
"""Tests for struct2tensor.calculate."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

from struct2tensor import calculate
from struct2tensor import calculate_options
from struct2tensor import create_expression
from struct2tensor import expression_add
from struct2tensor import path
from struct2tensor.expression_impl import promote
from struct2tensor.expression_impl import proto_test_util
from struct2tensor.test import expression_test_util
from struct2tensor.test import prensor_test_util
import tensorflow as tf

import unittest


def get_mock_linear_graph(length):
  result = [expression_test_util.get_mock_leaf(True, tf.int64, name="Node0")]
  for i in range(1, length):
    result.append(
        expression_test_util.get_mock_leaf(
            False,
            tf.int64,
            name="Node" + str(i),
            source_expressions=[result[i - 1]]))
  return result[-1]


def create_random_graph_helper(size, density, fraction_identity):
  partial = []
  if size:
    partial = create_random_graph_helper(size - 1, density, fraction_identity)
  if random.random() < fraction_identity and partial:
    source_expressions = [partial[random.randint(0, len(partial) - 1)]]
    partial.append(
        expression_test_util.get_mock_leaf(
            False,
            tf.int64,
            name="Node" + str(size - 1),
            source_expressions=source_expressions,
            calculate_is_identity=True))
    return partial

  source_expressions = []
  for x in partial:
    if random.random() < density:
      source_expressions.append(x)
  partial.append(
      expression_test_util.get_mock_leaf(
          False,
          tf.int64,
          name="Node" + str(size - 1),
          source_expressions=source_expressions))
  return partial


def create_random_graph(size, density, fraction_identity):
  return create_random_graph_helper(size, density, fraction_identity)[-1]


options_to_test = [
    calculate_options.get_default_options(),
    calculate_options.get_options_with_minimal_checks(), None
]


class CalculateTest(tf.test.TestCase):

  def test_calculate_mock(self):
    my_input = get_mock_linear_graph(5)
    [result] = calculate.calculate_values([my_input])
    self.assertIs(my_input._calculate_output, result)

  def test_calculate_broken_mock_is_repeated(self):
    with self.assertRaisesRegexp(
        ValueError,
        "Expression Node0 returned the wrong type: expected: repeated <dtype: "
        "'int64'> actual: optional <dtype: 'int64'>."):
      single_node = expression_test_util.get_mock_broken_leaf(
          True, tf.int64, False, tf.int64, name="Node0")
      calculate.calculate_values([single_node])

  def test_calculate_broken_mock_dtype(self):
    with self.assertRaisesRegexp(
        ValueError, "Expression Node0 returned the "
        "wrong type: expected: repeated <dtype: "
        "'int64'> actual: repeated <dtype: 'int32'>."):
      single_node = expression_test_util.get_mock_broken_leaf(
          True, tf.int64, True, tf.int32, name="Node0")
      calculate.calculate_values([single_node])

  def test_calculate_random_mock(self):
    """Test calculate on 1000 graphs with 20 nodes.

    This will have identity operations.
    This will not have equal operations (outside of identity operations).
    All nodes in the graph are leaf expressions.
    """
    random.seed(a=12345)
    for options in options_to_test:
      for _ in range(1000):
        my_input = create_random_graph(20, 0.5, 0.2)
        [result] = calculate.calculate_values([my_input], options=options)
        self.assertIs(my_input._calculate_output, result)

  def test_calculate_root_direct(self):
    """Calculates the value of a node with no sources."""
    for options in options_to_test:
      with self.session(use_gpu=False) as sess:
        tree = create_expression.create_expression_from_prensor(
            prensor_test_util.create_simple_prensor())
        [root_value] = calculate.calculate_values([tree], options=options)
        size_result = sess.run(root_value.size)
        self.assertAllEqual(size_result, 3)

  def test_calculate_root_indirect(self):
    """Calculates the value of a node with one source."""
    for options in options_to_test:
      with self.session(use_gpu=False) as sess:
        tree = create_expression.create_expression_from_prensor(
            prensor_test_util.create_simple_prensor())
        tree_2 = expression_add.add_paths(tree, {})
        [root_value] = calculate.calculate_values([tree_2], options=options)
        size_result = sess.run(root_value.size)
        self.assertAllEqual(size_result, 3)

  def test_calculate_tree_root_direct(self):
    """Calculates the value of a tree with no sources."""
    for options in options_to_test:
      with self.session(use_gpu=False) as sess:
        tree = create_expression.create_expression_from_prensor(
            prensor_test_util.create_simple_prensor())
        [new_expr] = calculate.calculate_prensors([tree], options=options)
        size_result = sess.run(new_expr.node.size)
        self.assertAllEqual(size_result, 3)

  def test_calculate_promote_anonymous(self):
    """Performs promote_test.PromoteValuesTest, but with calculate_values."""
    for options in options_to_test:
      with self.session(use_gpu=False) as sess:
        expr = create_expression.create_expression_from_prensor(
            prensor_test_util.create_nested_prensor())
        new_root, new_path = promote.promote_anonymous(
            expr, path.Path(["user", "friends"]))
        new_field = new_root.get_descendant_or_error(new_path)
        [leaf_node] = calculate.calculate_values([new_field], options=options)
        [parent_index,
         values] = sess.run([leaf_node.parent_index, leaf_node.values])
        self.assertAllEqual(parent_index, [0, 1, 1, 1, 2])
        self.assertAllEqual(values, [b"a", b"b", b"c", b"d", b"e"])

  def test_calculate_promote_named(self):
    """Performs promote_test.PromoteValuesTest, but with calculate_values."""
    for options in options_to_test:
      with self.session(use_gpu=False) as sess:
        expr = create_expression.create_expression_from_prensor(
            prensor_test_util.create_nested_prensor())
        new_root = promote.promote(expr, path.Path(["user", "friends"]),
                                   "new_friends")
        # projected = project.project(new_root, [path.Path(["new_friends"])])
        new_field = new_root.get_child_or_error("new_friends")
        [leaf_node] = calculate.calculate_values([new_field], options=options)
        [parent_index,
         values] = sess.run([leaf_node.parent_index, leaf_node.values])
        self.assertAllEqual(parent_index, [0, 1, 1, 1, 2])
        self.assertAllEqual(values, [b"a", b"b", b"c", b"d", b"e"])

  def test_create_query_and_calculate_event_value(self):
    """Calculating a child value in a proto tests dependencies."""
    for options in options_to_test:
      with self.session(use_gpu=False) as sess:
        expr = proto_test_util._get_expression_from_session_empty_user_info()
        [event_value
        ] = calculate.calculate_values([expr.get_child_or_error("event")],
                                       options=options)
        parent_index = sess.run(event_value.parent_index)
        self.assertAllEqual(parent_index, [0, 0, 0, 1, 1])

  def test_create_query_modify_and_calculate_event_value(self):
    """Calculating a child value in a proto tests dependencies."""
    for options in options_to_test:
      with self.session(use_gpu=False) as sess:
        root = proto_test_util._get_expression_from_session_empty_user_info()
        root_2 = expression_add.add_paths(
            root, {path.Path(["event_copy"]): root.get_child_or_error("event")})
        [event_value] = calculate.calculate_values(
            [root_2.get_child_or_error("event_copy")], options=options)
        parent_index = sess.run(event_value.parent_index)
        self.assertAllEqual(parent_index, [0, 0, 0, 1, 1])


if __name__ == "__main__":
  unittest.main()
