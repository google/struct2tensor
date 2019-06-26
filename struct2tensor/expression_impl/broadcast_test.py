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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from absl.testing import absltest

from struct2tensor import create_expression
from struct2tensor.test import expression_test_util
from struct2tensor import path
from struct2tensor.test import prensor_test_util
from struct2tensor.expression_impl import broadcast


class BroadcastTest(absltest.TestCase):

  def test_broadcast_anonymous(self):
    expr = create_expression.create_expression_from_prensor(
        prensor_test_util.create_big_prensor())
    new_root, p = broadcast.broadcast_anonymous(expr, path.Path(["foo"]),
                                                "user")
    [new_field] = new_root.get_descendant_or_error(p).get_source_expressions()
    self.assertFalse(new_field.is_repeated)
    self.assertEqual(new_field.type, tf.int32)
    self.assertTrue(new_field.is_leaf)
    self.assertTrue(new_field.calculation_equal(new_field))
    self.assertFalse(new_field.calculation_equal(expr))
    leaf_node = expression_test_util.calculate_value_slowly(new_field)
    self.assertEqual(leaf_node.values.dtype, tf.int32)
    self.assertEqual(new_field.known_field_names(), frozenset())

    sources = new_field.get_source_expressions()
    self.assertLen(sources, 2)
    self.assertIs(expr.get_child("foo"), sources[0])
    self.assertIs(expr.get_child("user"), sources[1])

  def test_broadcast(self):
    """Tests broadcast.broadcast(...), and indirectly tests set_path."""
    expr = create_expression.create_expression_from_prensor(
        prensor_test_util.create_big_prensor())
    new_root = broadcast.broadcast(expr, path.Path(["foo"]), "user",
                                   "new_field")
    new_field = new_root.get_child("user").get_child("new_field")
    self.assertIsNotNone(new_field)
    self.assertFalse(new_field.is_repeated)
    self.assertEqual(new_field.type, tf.int32)
    self.assertTrue(new_field.is_leaf)
    leaf_node = expression_test_util.calculate_value_slowly(new_field)
    self.assertEqual(leaf_node.values.dtype, tf.int32)
    self.assertEqual(new_field.known_field_names(), frozenset())


class BroadcastValuesTest(tf.test.TestCase):

  def test_broadcast_and_calculate(self):
    """Tests get_sparse_tensors on a deep tree."""
    with self.session(use_gpu=False) as sess:
      expr = create_expression.create_expression_from_prensor(
          prensor_test_util.create_big_prensor())
      new_root, new_path = broadcast.broadcast_anonymous(
          expr, path.Path(["foo"]), "user")
      new_field = new_root.get_descendant_or_error(new_path)
      leaf_node = expression_test_util.calculate_value_slowly(new_field)
      [parent_index,
       values] = sess.run([leaf_node.parent_index, leaf_node.values])
      self.assertAllEqual(parent_index, [0, 1, 2, 3])
      self.assertAllEqual(values, [9, 8, 8, 7])


if __name__ == "__main__":
  absltest.main()
