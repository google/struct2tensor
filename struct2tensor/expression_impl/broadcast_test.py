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
from struct2tensor.expression_impl import broadcast
from struct2tensor.test import expression_test_util
from struct2tensor.test import prensor_test_util
import tensorflow as tf

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


class BroadcastTest(tf.test.TestCase):

  def test_broadcast_anonymous(self):
    expr = create_expression.create_expression_from_prensor(
        prensor_test_util.create_big_prensor())
    new_root, p = broadcast.broadcast_anonymous(expr, path.Path(["foo"]),
                                                "user")
    new_field = new_root.get_descendant_or_error(p)
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

  def test_broadcast_substructure(self):
    """Tests broadcast of a submessage.

    The result of broadcasting `user` into `doc` looks like:
    {
      foo: 9,
      foorepeated: [9],
      doc: [{bar:["a"], keep_me:False, new_user: [{friends:["a"]}]}],
      user: [{friends:["a"]}]
    },
    {
      foo: 8,
      foorepeated: [8, 7],
      doc: [
        {
          bar: ["b","c"],
          keep_me: True,
          new_user: [{friends:["b", "c"]},{friends:["d"]}]
        },
        {
          bar: ["d"],
          new_user: [{friends:["b", "c"]},{friends:["d"]}]
        }
      ],
      user: [{friends:["b", "c"]},{friends:["d"]}],
    },
    {
      foo: 7,
      foorepeated: [6],
      user: [{friends:["e"]}]
    }
    """
    expr = create_expression.create_expression_from_prensor(
        prensor_test_util.create_big_prensor())
    new_root = broadcast.broadcast(expr, path.Path(["user"]), "doc", "new_user")
    new_user = new_root.get_child("doc").get_child("new_user")
    self.assertIsNotNone(new_user)
    self.assertTrue(new_user.is_repeated)
    self.assertIsNone(new_user.type)
    self.assertFalse(new_user.is_leaf)

    new_user_node = expression_test_util.calculate_value_slowly(new_user)
    self.assertAllEqual(new_user_node.parent_index, [0, 1, 1, 2, 2])
    self.assertAllEqual(new_user_node.index_to_value, [0, 1, 2, 1, 2])

    new_friends = new_user.get_child("friends")
    self.assertIsNotNone(new_friends)
    self.assertTrue(new_friends.is_repeated)
    self.assertEqual(new_friends.type, tf.string)
    self.assertTrue(new_friends.is_leaf)

    new_friends_node = expression_test_util.calculate_value_slowly(new_friends)
    self.assertEqual(new_friends_node.values.dtype, tf.string)
    self.assertAllEqual(new_friends_node.values,
                        ["a", "b", "c", "d", "b", "c", "d"])
    self.assertAllEqual(new_friends_node.parent_index, [0, 1, 1, 2, 3, 3, 4])

  def test_broadcast_substructure_deep(self):
    """Tests broadcast of a submessage.

    The result of broadcasting `event` into `user` looks like:
    {
      foo: 9,
      foorepeated: [9],
      user: [{
        friends: ["a"],
        new_event: [{
          doc:[{
            bar: ["a"],
            keep_me:False
          }]
        }]
      }],
      event: [{doc:[{bar:["a"], keep_me:False}]}]
    },
    {
      foo: 8,
      foorepeated: [8,7],
      user: [{
        friends: ["b", "c"],
        new_event: [{
          doc:[{
            bar: ["b","c"],
            keep_me:True
          },
          {
            bar:["d"]
          }]
        }]
      },
      {
        friends: ["d"],
        new_event: [{
          doc:[{
            bar: ["b","c"],
            keep_me: True
          },
          {
            bar: ["d"]
          }]
        }]
      }],
      event: [{doc:[{bar:["b","c"], keep_me:True},{bar:["d"]}]}]
    },
    {
      foo:7,
      foorepeated: [6],
      user: [{
        friends:["e"],
        new_event: [{}]
      }],
      event: [{}]
    }
    """
    expr = create_expression.create_expression_from_prensor(
        prensor_test_util.create_deep_prensor())
    new_root = broadcast.broadcast(expr, path.Path(["event"]), "user",
                                   "new_event")
    new_event = new_root.get_child("user").get_child("new_event")
    self.assertIsNotNone(new_event)
    self.assertTrue(new_event.is_repeated)
    self.assertIsNone(new_event.type)
    self.assertFalse(new_event.is_leaf)

    new_event_node = expression_test_util.calculate_value_slowly(new_event)
    self.assertAllEqual(new_event_node.parent_index, [0, 1, 2, 3])
    self.assertAllEqual(new_event_node.index_to_value, [0, 1, 1, 2])

    new_doc = new_event.get_child("doc")
    self.assertIsNotNone(new_doc)
    self.assertTrue(new_doc.is_repeated)
    self.assertIsNone(new_doc.type)
    self.assertFalse(new_doc.is_leaf)

    new_doc_node = expression_test_util.calculate_value_slowly(new_doc)
    self.assertAllEqual(new_doc_node.parent_index, [0, 1, 1, 2, 2])
    self.assertAllEqual(new_doc_node.index_to_value, [0, 1, 2, 1, 2])

    new_bar = new_doc.get_child("bar")
    self.assertIsNotNone(new_bar)
    self.assertTrue(new_doc.is_repeated)
    self.assertEqual(new_bar.type, tf.string)
    self.assertTrue(new_bar.is_leaf)

    new_bar_node = expression_test_util.calculate_value_slowly(new_bar)
    self.assertAllEqual(new_bar_node.values,
                        ["a", "b", "c", "d", "b", "c", "d"])
    self.assertAllEqual(new_bar_node.parent_index, [0, 1, 1, 2, 3, 3, 4])

    new_keep_me = new_doc.get_child("keep_me")
    self.assertIsNotNone(new_keep_me)
    self.assertFalse(new_keep_me.is_repeated)
    self.assertEqual(new_keep_me.type, tf.bool)
    self.assertTrue(new_keep_me.is_leaf)

    new_keep_me_node = expression_test_util.calculate_value_slowly(new_keep_me)
    self.assertAllEqual(new_keep_me_node.values,
                        [False, True, True])
    self.assertAllEqual(new_keep_me_node.parent_index, [0, 1, 3])

  def test_broadcast_with_lenient_names(self):
    """Tests broadcast with lenient step names."""
    expr = create_expression.create_expression_from_prensor(
        prensor_test_util.create_nested_prensor_with_lenient_field_names(),
        validate_step_format=False,
    )
    new_root, new_path = broadcast.broadcast_anonymous(
        expr, path.Path(["doc"], validate_step_format=False), "user"
    )
    new_field = new_root.get_descendant_or_error(new_path)
    leaf_node = expression_test_util.calculate_value_slowly(new_field)
    self.assertAllEqual(leaf_node.parent_index, [0, 1, 1, 2, 2])


@test_util.run_all_in_graph_and_eager_modes
class BroadcastValuesTest(tf.test.TestCase):

  def test_broadcast_and_calculate(self):
    """Tests get_sparse_tensors on a deep tree."""
    expr = create_expression.create_expression_from_prensor(
        prensor_test_util.create_big_prensor())
    new_root, new_path = broadcast.broadcast_anonymous(expr, path.Path(["foo"]),
                                                       "user")
    new_field = new_root.get_descendant_or_error(new_path)
    leaf_node = expression_test_util.calculate_value_slowly(new_field)
    self.assertAllEqual(leaf_node.parent_index, [0, 1, 2, 3])
    self.assertAllEqual(leaf_node.values, [9, 8, 8, 7])


if __name__ == "__main__":
  absltest.main()
