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
"""Tests for struct2tensor.proto."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from absl.testing import absltest
from struct2tensor.test import test_any_pb2
from struct2tensor.test import test_extension_pb2
from struct2tensor.test import test_pb2

from struct2tensor.test import expression_test_util
from struct2tensor import path
from struct2tensor.expression_impl import proto
from struct2tensor.expression_impl import proto_test_util


def _get_expression_with_any():
  my_any_0 = test_any_pb2.MessageWithAny()
  my_value_0 = test_pb2.AllSimple()
  my_value_0.optional_int32 = 0
  my_any_0.my_any.Pack(my_value_0)
  my_any_1 = test_any_pb2.MessageWithAny()
  my_value_1 = test_pb2.UserInfo()
  my_any_1.my_any.Pack(my_value_1)
  my_any_2 = test_any_pb2.MessageWithAny()
  my_value_2 = test_pb2.AllSimple()
  my_value_2.optional_int32 = 20
  my_any_2.my_any.Pack(my_value_2)
  serialized = [x.SerializeToString() for x in [my_any_0, my_any_1, my_any_2]]
  return proto.create_expression_from_proto(
      serialized, test_any_pb2.MessageWithAny.DESCRIPTOR)


def _get_user_info_with_extension():
  my_user_info = test_pb2.UserInfo()
  my_user_info.Extensions[
      test_extension_pb2.MyExternalExtension.ext].special = "shhh"

  serialized = [my_user_info.SerializeToString()]
  return proto.create_expression_from_proto(
      serialized, test_any_pb2.MessageWithAny.DESCRIPTOR)


class ProtoTest(absltest.TestCase):

  def test_create_expression_from_proto_with_event(self):
    expr = proto_test_util._get_expression_from_session_empty_user_info()
    event_expr = expr.get_child_or_error("event")
    self.assertTrue(event_expr.is_repeated)
    self.assertIsNone(event_expr.type)
    self.assertFalse(event_expr.is_leaf)
    self.assertFalse(event_expr.calculation_is_identity())
    self.assertTrue(event_expr.calculation_equal(event_expr))
    self.assertFalse(event_expr.calculation_equal(expr))
    child_node = expression_test_util.calculate_value_slowly(event_expr)
    self.assertEqual(child_node.parent_index.dtype, tf.int64)
    self.assertEqual(
        event_expr.known_field_names(),
        frozenset({
            "event_id", "query", "query_token", "action", "user_info",
            "action_mask"
        }))

    sources = event_expr.get_source_expressions()
    self.assertLen(sources, 1)
    self.assertIs(expr, sources[0])

  def test_create_expression_from_proto_with_root(self):
    expr = proto_test_util._get_expression_from_session_empty_user_info()
    self.assertTrue(expr.is_repeated)
    self.assertIsNone(expr.type)
    self.assertFalse(expr.is_leaf)
    self.assertFalse(expr.calculation_is_identity())
    self.assertTrue(expr.calculation_equal(expr))
    self.assertFalse(expr.calculation_equal(expr.get_child_or_error("event")))
    root_node = expression_test_util.calculate_value_slowly(expr)
    self.assertEqual(root_node.size.dtype, tf.int64)
    self.assertEqual(expr.known_field_names(),
                     frozenset({"event", "session_id", "session_info"}))

    sources = expr.get_source_expressions()
    self.assertEmpty(sources)

  def test_user_info_with_extension(self):
    expr = _get_user_info_with_extension()
    ext_expr = expr.get_child_or_error("(struct2tensor.test.MyExternalExtension.ext)")
    self.assertFalse(ext_expr.is_repeated)
    self.assertIsNone(ext_expr.type)
    self.assertFalse(ext_expr.is_leaf)
    self.assertFalse(ext_expr.calculation_is_identity())
    self.assertTrue(ext_expr.calculation_equal(ext_expr))
    self.assertFalse(ext_expr.calculation_equal(expr))
    child_node = expression_test_util.calculate_value_slowly(ext_expr)
    self.assertEqual(child_node.parent_index.dtype, tf.int64)
    self.assertEqual(ext_expr.known_field_names(), frozenset({"special"}))

  def test_missing_extension(self):
    """Tests a missing extension on a deep tree."""
    expr = proto_test_util._get_expression_from_session_empty_user_info()
    missing_expr = expr.get_child("(ext.NotPresent)")
    self.assertIsNone(missing_expr)

  def test_create_expression_from_proto_with_any(self):
    """Test an any field."""
    expr = _get_expression_with_any()
    any_expr = expr.get_child_or_error("my_any")
    simple_expr = expr.get_descendant_or_error(
        path.Path(["my_any", "(type.googleapis.com/struct2tensor.test.AllSimple)"]))
    self.assertFalse(simple_expr.is_repeated)
    self.assertIsNone(simple_expr.type)
    self.assertFalse(simple_expr.is_leaf)
    self.assertFalse(simple_expr.calculation_is_identity())
    self.assertTrue(simple_expr.calculation_equal(simple_expr))
    self.assertFalse(simple_expr.calculation_equal(expr))
    child_node = expression_test_util.calculate_value_slowly(simple_expr)
    self.assertEqual(child_node.parent_index.dtype, tf.int64)
    self.assertEqual(
        simple_expr.known_field_names(),
        frozenset({
            "optional_string", "optional_uint64", "repeated_uint64",
            "repeated_int32", "repeated_string", "optional_int32",
            "optional_float", "repeated_int64", "optional_uint32",
            "repeated_float", "repeated_uint32", "optional_double",
            "optional_int64", "repeated_double"
        }))

    sources = simple_expr.get_source_expressions()
    self.assertLen(sources, 1)
    self.assertIs(any_expr, sources[0])

  def test_create_expression_from_proto_with_any_missing_message(self):
    """Test an any field."""
    expr = _get_expression_with_any()
    any_expr = expr.get_child_or_error("my_any")
    simple_expr = expr.get_descendant_or_error(
        path.Path(
            ["my_any", "(type.googleapis.com/struct2tensor.test.SpecialUserInfo)"]))
    self.assertFalse(simple_expr.is_repeated)
    self.assertIsNone(simple_expr.type)
    self.assertFalse(simple_expr.is_leaf)
    self.assertFalse(simple_expr.calculation_is_identity())
    self.assertTrue(simple_expr.calculation_equal(simple_expr))
    self.assertFalse(simple_expr.calculation_equal(expr))
    child_node = expression_test_util.calculate_value_slowly(simple_expr)
    self.assertEqual(child_node.parent_index.dtype, tf.int64)
    self.assertEqual(simple_expr.known_field_names(), frozenset({"secret"}))

    sources = simple_expr.get_source_expressions()
    self.assertLen(sources, 1)
    self.assertIs(any_expr, sources[0])

  def test_create_expression_from_proto_with_any_type_url(self):
    """Test an any with type_url."""
    expr = _get_expression_with_any()
    any_expr = expr.get_child_or_error("my_any")
    simple_expr = expr.get_descendant_or_error(
        path.Path(["my_any", "type_url"]))
    self.assertFalse(simple_expr.is_repeated)
    self.assertEqual(simple_expr.type, tf.string)
    self.assertTrue(simple_expr.is_leaf)
    self.assertFalse(simple_expr.calculation_is_identity())
    self.assertTrue(simple_expr.calculation_equal(simple_expr))
    self.assertFalse(simple_expr.calculation_equal(expr))
    leaf_node = expression_test_util.calculate_value_slowly(simple_expr)
    self.assertEqual(leaf_node.parent_index.dtype, tf.int64)
    self.assertEqual(leaf_node.values.dtype, tf.string)
    self.assertEqual(simple_expr.known_field_names(), frozenset({}))

    sources = simple_expr.get_source_expressions()
    self.assertLen(sources, 1)
    self.assertIs(any_expr, sources[0])

  def test_create_expression_from_proto_with_any_value(self):
    """Test an any with value."""
    expr = _get_expression_with_any()
    any_expr = expr.get_child_or_error("my_any")
    simple_expr = expr.get_descendant_or_error(path.Path(["my_any", "value"]))
    self.assertFalse(simple_expr.is_repeated)
    self.assertEqual(simple_expr.type, tf.string)
    self.assertTrue(simple_expr.is_leaf)
    self.assertFalse(simple_expr.calculation_is_identity())
    self.assertTrue(simple_expr.calculation_equal(simple_expr))
    self.assertFalse(simple_expr.calculation_equal(expr))
    leaf_node = expression_test_util.calculate_value_slowly(simple_expr)
    self.assertEqual(leaf_node.parent_index.dtype, tf.int64)
    self.assertEqual(leaf_node.values.dtype, tf.string)
    self.assertEqual(simple_expr.known_field_names(), frozenset({}))

    sources = simple_expr.get_source_expressions()
    self.assertLen(sources, 1)
    self.assertIs(any_expr, sources[0])


class ProtoValuesTest(tf.test.TestCase):

  def test_create_expression_from_proto_and_calculate_root_value(self):
    """Tests get_sparse_tensors on a deep tree."""
    with self.session(use_gpu=False) as sess:
      expr = proto_test_util._get_expression_from_session_empty_user_info()
      root_value = expression_test_util.calculate_value_slowly(expr)
      size = sess.run(root_value.size)
      self.assertEqual(size, 2)

  def test_create_expression_from_proto_and_calculate_event_value(self):
    """Tests get_sparse_tensors on a deep tree."""
    with self.session(use_gpu=False) as sess:
      expr = proto_test_util._get_expression_from_session_empty_user_info()
      event_value = expression_test_util.calculate_value_slowly(
          expr.get_child_or_error("event"))
      parent_index = sess.run(event_value.parent_index)
      self.assertAllEqual(parent_index, [0, 0, 0, 1, 1])

  def test_create_expression_from_proto_and_calculate_event_id_value(self):
    """Tests get_sparse_tensors on a deep tree."""
    with self.session(use_gpu=False) as sess:
      expr = proto_test_util._get_expression_from_session_empty_user_info()
      event_id_value = expression_test_util.calculate_value_slowly(
          expr.get_descendant_or_error(path.Path(["event", "event_id"])))
      [parent_index,
       values] = sess.run([event_id_value.parent_index, event_id_value.values])
      self.assertAllEqual(parent_index, [0, 1, 2, 4])
      self.assertAllEqual(values, [b"A", b"B", b"C", b"D"])

  def test_create_expression_from_proto_with_any(self):
    """Test an any field."""
    with self.session(use_gpu=False) as sess:
      expr = _get_expression_with_any()
      simple_expr = expr.get_descendant_or_error(
          path.Path(["my_any", "(type.googleapis.com/struct2tensor.test.AllSimple)"]))
      child_node = sess.run(
          expression_test_util.calculate_value_slowly(simple_expr).parent_index)
      self.assertAllEqual(child_node, [0, 2])

  def test_create_expression_from_proto_with_any_missing_message(self):
    """Test an any field with a message that is absent."""
    with self.session(use_gpu=False) as sess:
      expr = _get_expression_with_any()
      simple_expr = expr.get_descendant_or_error(
          path.Path(
              ["my_any", "(type.googleapis.com/struct2tensor.test.SpecialUserInfo)"]))
      child_node = sess.run(
          expression_test_util.calculate_value_slowly(simple_expr).parent_index)
      self.assertAllEqual(child_node, [])

  def test_project_proto_map(self):
    examples = [
        """
        features {
          feature {
            key: "feature1"
            value { bytes_list { value: ["hello", "world"] } }
          }
          feature {
            key: "feature2"
            value { float_list { value: 8.0 } }
          }
        }
        """, """
        features {
          feature {
            key: "feature1"
            value { bytes_list { value: "deadbeef" } }
          }
          feature {
            key: "feature3"
            value { int64_list { value: [123, 456] } }
          }
        }
        """
    ]
    with tf.Session() as sess:
      expr = proto_test_util.text_to_expression(examples, tf.train.Example)
      result = expression_test_util.calculate_list_map(
          expr.project([
              "features.feature[feature1].bytes_list.value",
              "features.feature[feature2].float_list.value",
              "features.feature[feature3].int64_list.value",
          ]), sess)

      feature1 = result["features.feature[feature1].bytes_list.value"]
      feature2 = result["features.feature[feature2].float_list.value"]
      feature3 = result["features.feature[feature3].int64_list.value"]
      self.assertAllEqual(feature1,
                          [[[[[b"hello", b"world"]]]], [[[[b"deadbeef"]]]]])
      self.assertAllEqual(feature2, [[[[[8.0]]]], [[]]])
      self.assertAllEqual(feature3, [[[]], [[[[123, 456]]]]])


if __name__ == "__main__":
  absltest.main()
