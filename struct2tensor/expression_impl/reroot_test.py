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
"""Tests for struct2tensor.reroot."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from struct2tensor import calculate
from struct2tensor import calculate_options
from struct2tensor import create_expression
from struct2tensor import path
from struct2tensor import prensor_util
from struct2tensor.expression_impl import proto_test_util
from struct2tensor.expression_impl import reroot
from struct2tensor.test import expression_test_util
from struct2tensor.test import prensor_test_util
from struct2tensor.test import test_pb2
import tensorflow as tf


from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


@test_util.run_all_in_graph_and_eager_modes
class RerootTest(tf.test.TestCase):

  def test_reroot_and_create_proto_index(self):
    expr = create_expression.create_expression_from_prensor(
        prensor_test_util.create_big_prensor())
    new_root = reroot.reroot(expr, path.Path(["doc"]))
    proto_index = reroot.create_proto_index_field(
        new_root, "proto_index").get_child("proto_index")
    new_field = new_root.get_child("bar")
    leaf_node = expression_test_util.calculate_value_slowly(new_field)
    proto_index_node = expression_test_util.calculate_value_slowly(proto_index)

    self.assertIsNotNone(new_field)
    self.assertTrue(new_field.is_repeated)
    self.assertEqual(new_field.type, tf.string)
    self.assertTrue(new_field.is_leaf)
    self.assertEqual(new_field.known_field_names(), frozenset())
    self.assertEqual(leaf_node.values.dtype, tf.string)

    self.assertIsNotNone(proto_index)
    self.assertFalse(proto_index.is_repeated)
    self.assertEqual(proto_index.type, tf.int64)
    self.assertTrue(proto_index.is_leaf)
    self.assertEqual(proto_index.known_field_names(), frozenset())

    self.assertEqual(proto_index_node.values.dtype, tf.int64)

    self.assertAllEqual([b"a", b"b", b"c", b"d"], leaf_node.values)
    self.assertAllEqual([0, 1, 1, 2], leaf_node.parent_index)
    self.assertAllEqual([0, 1, 1], proto_index_node.values)
    self.assertAllEqual([0, 1, 2], proto_index_node.parent_index)

  def test_reroot_and_create_proto_index_deep(self):
    expr = create_expression.create_expression_from_prensor(
        prensor_test_util.create_deep_prensor())
    new_root = reroot.reroot(expr, path.Path(["event", "doc"]))
    proto_index = reroot.create_proto_index_field(
        new_root, "proto_index").get_child("proto_index")
    new_field = new_root.get_child("bar")
    leaf_node = expression_test_util.calculate_value_slowly(new_field)
    proto_index_node = expression_test_util.calculate_value_slowly(proto_index)

    self.assertIsNotNone(new_field)
    self.assertTrue(new_field.is_repeated)
    self.assertEqual(new_field.type, tf.string)
    self.assertTrue(new_field.is_leaf)
    self.assertEqual(new_field.known_field_names(), frozenset())
    self.assertEqual(leaf_node.values.dtype, tf.string)

    self.assertIsNotNone(proto_index)
    self.assertFalse(proto_index.is_repeated)
    self.assertEqual(proto_index.type, tf.int64)
    self.assertTrue(proto_index.is_leaf)
    self.assertEqual(proto_index.known_field_names(), frozenset())

    self.assertEqual(proto_index_node.values.dtype, tf.int64)

    self.assertAllEqual([b"a", b"b", b"c", b"d"], leaf_node.values)
    self.assertAllEqual([0, 1, 1, 2], leaf_node.parent_index)
    self.assertAllEqual([0, 1, 1], proto_index_node.values)
    self.assertAllEqual([0, 1, 2], proto_index_node.parent_index)

  def test_create_proto_index_directly_reroot_at_action(self):
    sessions = [
        """
        event {
          action {}
          action {}
        }
        event {}
        event { action {} }
        """, "", """
        event {}
        event {
          action {}
          action {}
        }
        event {  }
        """
    ]
    expr = proto_test_util.text_to_expression(sessions, test_pb2.Session)
    reroot_expr = expr.reroot("event.action")
    # Reroot with a depth > 1 (all the other cases are depth == 1)
    proto_index_directly_reroot_at_action = (
        reroot_expr.create_proto_index("proto_index_directly_reroot_at_action")
        .get_child_or_error("proto_index_directly_reroot_at_action"))

    self.assertFalse(proto_index_directly_reroot_at_action.is_repeated)
    result = expression_test_util.calculate_value_slowly(
        proto_index_directly_reroot_at_action)
    self.assertAllEqual(result.parent_index, [0, 1, 2, 3, 4])
    self.assertAllEqual(result.values, [0, 0, 0, 2, 2])

  def test_create_proto_index_directly_reroot_at_action_sparse_dense(self):
    sessions = [
        """
        event {
          action {}
          action {}
        }
        event {}
        event { action {} }
        """, "", """
        event {}
        event {
          action {}
          action {}
        }
        event {  }
        """
    ]
    expr = proto_test_util.text_to_expression(sessions, test_pb2.Session)
    reroot_expr = expr.reroot("event.action")
    # Reroot with a depth > 1 (all the other cases are depth == 1)
    [prensor_tree] = calculate.calculate_prensors([
        reroot_expr.create_proto_index("proto_index_directly_reroot_at_action")
    ])
    proto_index_node = prensor_tree.get_child_or_error(
        "proto_index_directly_reroot_at_action").node
    self.assertFalse(proto_index_node.is_repeated)
    sparse_tensors = prensor_util.get_sparse_tensors(
        prensor_tree, calculate_options.get_default_options())
    proto_index_directly_reroot_at_action = sparse_tensors[path.Path(
        ["proto_index_directly_reroot_at_action"])]
    dense_value = tf.sparse.to_dense(
        proto_index_directly_reroot_at_action)
    sparse_value = proto_index_directly_reroot_at_action

    self.assertAllEqual(sparse_value.values, [0, 0, 0, 2, 2])
    self.assertAllEqual(sparse_value.indices, [[0], [1], [2], [3], [4]])
    self.assertAllEqual(sparse_value.dense_shape, [5])
    self.assertAllEqual(dense_value, [0, 0, 0, 2, 2])


if __name__ == "__main__":
  absltest.main()
