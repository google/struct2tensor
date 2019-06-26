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
"""Tests for struct2tensor.expression.

This copies over the simplest test from each implementation module,
just to make sure the link is done correctly. Some of these used calculations,
others evaluated the resulting expression object's structure.

For further tests on expressions, see v1_compat_test.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest

from struct2tensor import create_expression
from struct2tensor import path
from struct2tensor.test import expression_test_util
from struct2tensor.test import prensor_test_util

import tensorflow as tf

from tensorflow_metadata.proto.v0 import schema_pb2


def _features_as_map(feature_list):
  return {feature.name: feature for feature in feature_list}


class ExpressionTest(absltest.TestCase):

  def test_promote(self):
    expr = create_expression.create_expression_from_prensor(
        prensor_test_util.create_nested_prensor())
    new_root = expr.promote("user.friends", "new_field")
    new_field = new_root.get_child_or_error("new_field")
    self.assertIsNotNone(new_field)
    self.assertTrue(new_field.is_repeated)
    self.assertEqual(new_field.type, tf.string)
    self.assertTrue(new_field.is_leaf)
    leaf_node = expression_test_util.calculate_value_slowly(new_field)
    self.assertEqual(leaf_node.values.dtype, tf.string)
    self.assertEqual(new_field.known_field_names(), frozenset())

  def test_broadcast(self):
    """Tests broadcast.broadcast(...), and indirectly tests set_path."""
    expr = create_expression.create_expression_from_prensor(
        prensor_test_util.create_big_prensor())
    new_root = expr.broadcast("foo", "user", "new_field")
    new_field = new_root.get_child("user").get_child("new_field")
    self.assertIsNotNone(new_field)
    self.assertFalse(new_field.is_repeated)
    self.assertEqual(new_field.type, tf.int32)
    self.assertTrue(new_field.is_leaf)
    leaf_node = expression_test_util.calculate_value_slowly(new_field)
    self.assertEqual(leaf_node.values.dtype, tf.int32)
    self.assertEqual(new_field.known_field_names(), frozenset())

  def test_project(self):
    expr = create_expression.create_expression_from_prensor(
        prensor_test_util.create_nested_prensor())
    projected = expr.project(
        [path.Path(["user", "friends"]),
         path.Path(["doc", "keep_me"])])
    self.assertIsNotNone(
        projected.get_descendant(path.Path(["user", "friends"])))
    self.assertIsNotNone(
        projected.get_descendant(path.Path(["doc", "keep_me"])))
    self.assertIsNone(projected.get_descendant(path.Path(["doc", "bar"])))

  def test_promote_and_broadcast_test(self):
    expr = create_expression.create_expression_from_prensor(
        prensor_test_util.create_big_prensor())
    new_root = expr.promote_and_broadcast({"new_field": "user.friends"},
                                          path.Path(["doc"]))
    new_field = new_root.get_descendant_or_error(
        path.Path(["doc", "new_field"]))
    self.assertTrue(new_field.is_repeated)
    self.assertEqual(new_field.type, tf.string)
    self.assertTrue(new_field.is_leaf)
    self.assertTrue(new_field.calculation_equal(new_field))
    self.assertFalse(new_field.calculation_equal(expr))
    leaf_node = expression_test_util.calculate_value_slowly(new_field)
    self.assertEqual(leaf_node.values.dtype, tf.string)
    self.assertEqual(new_field.known_field_names(), frozenset())

  def test_get_schema(self):
    foo_feature = schema_pb2.Feature()
    foo_feature.int_domain.max = 10
    foo = expression_test_util.MockExpression(
        is_repeated=False, my_type=tf.int64, schema_feature=foo_feature)
    foorepeated = expression_test_util.MockExpression(
        is_repeated=True, my_type=tf.int64)
    bar_feature = schema_pb2.Feature()
    bar_feature.presence.min_count = 17
    bar = expression_test_util.MockExpression(
        is_repeated=True, my_type=tf.string, schema_feature=bar_feature)
    keep_me = expression_test_util.MockExpression(
        is_repeated=False, my_type=tf.bool)

    doc = expression_test_util.MockExpression(
        is_repeated=True, my_type=tf.int64, children={"bar": bar,
                                                      "keep_me": keep_me})
    root = expression_test_util.MockExpression(
        is_repeated=True,
        my_type=None,
        children={
            "foo": foo,
            "foorepeated": foorepeated,
            "doc": doc
        })

    schema_result = root.get_schema()
    feature_map = _features_as_map(schema_result.feature)
    self.assertIn("foo", feature_map)
    # Check the properties of a first-level feature.
    self.assertEqual(feature_map["foo"].int_domain.max, 10)
    self.assertIn("foorepeated", feature_map)

    doc_feature_map = _features_as_map(feature_map["doc"].struct_domain.feature)
    # Test that second level features are correctly handled.
    self.assertIn("bar", doc_feature_map)
    # Test that an string_domain specified at the schema level is inserted
    # correctly.
    self.assertEqual(doc_feature_map["bar"].presence.min_count, 17)
    self.assertIn("keep_me", doc_feature_map)


class ExpressionValuesTest(tf.test.TestCase):

  def test_map_field_values_test(self):
    with self.session(use_gpu=False) as sess:

      expr = create_expression.create_expression_from_prensor(
          prensor_test_util.create_simple_prensor())

      new_root = expr.map_field_values("foo", lambda x: x * 2, tf.int64,
                                       "foo_doubled")

      leaf_node = expression_test_util.calculate_value_slowly(
          new_root.get_descendant_or_error(path.Path(["foo_doubled"])))
      [parent_index,
       values] = sess.run([leaf_node.parent_index, leaf_node.values])

      self.assertAllEqual(parent_index, [0, 1, 2])
      self.assertAllEqual(values, [18, 16, 14])

  def test_create_size_field(self):
    with self.test_session(use_gpu=False) as sess:
      expr = create_expression.create_expression_from_prensor(
          prensor_test_util.create_big_prensor())
      new_root = expr.create_size_field("doc.bar", "result")
      new_field = new_root.get_descendant_or_error(path.Path(["doc", "result"]))
      leaf_node = expression_test_util.calculate_value_slowly(new_field)
      [parent_index,
       values] = sess.run([leaf_node.parent_index, leaf_node.values])
      self.assertAllEqual(parent_index, [0, 1, 2])
      self.assertAllEqual(values, [1, 2, 1])

  def test_create_has_field(self):
    with self.test_session(use_gpu=False) as sess:
      expr = create_expression.create_expression_from_prensor(
          prensor_test_util.create_big_prensor())
      new_root = expr.create_has_field("doc.keep_me", "result")
      new_field = new_root.get_descendant_or_error(path.Path(["doc", "result"]))
      leaf_node = expression_test_util.calculate_value_slowly(new_field)
      [parent_index,
       values] = sess.run([leaf_node.parent_index, leaf_node.values])
      self.assertAllEqual(parent_index, [0, 1, 2])
      self.assertAllEqual(values, [True, True, False])

  def test_reroot_and_create_proto_index(self):
    expr = create_expression.create_expression_from_prensor(
        prensor_test_util.create_big_prensor()).reroot(
            "doc").create_proto_index("proto_index")
    proto_index = expr.get_child("proto_index")
    new_field = expr.get_child("bar")
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

    with self.session(use_gpu=False) as sess:
      [
          leaf_node_values, leaf_node_parent_index, proto_index_values,
          proto_index_parent_index
      ] = sess.run([
          leaf_node.values, leaf_node.parent_index, proto_index_node.values,
          proto_index_node.parent_index
      ])
      self.assertAllEqual([b"a", b"b", b"c", b"d"], leaf_node_values)
      self.assertAllEqual([0, 1, 1, 2], leaf_node_parent_index)
      self.assertAllEqual([0, 1, 1], proto_index_values)
      self.assertAllEqual([0, 1, 2], proto_index_parent_index)


if __name__ == "__main__":
  absltest.main()
