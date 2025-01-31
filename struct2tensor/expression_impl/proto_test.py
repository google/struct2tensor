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

import tensorflow as tf
from absl.testing import absltest, parameterized
from tensorflow.python.framework import (
  test_util,  # pylint: disable=g-direct-tensorflow-import
)

from struct2tensor import calculate_options, path
from struct2tensor.expression_impl import proto, proto_test_util
from struct2tensor.test import (
  expression_test_util,
  test_any_pb2,
  test_extension_pb2,
  test_map_pb2,
  test_pb2,
)


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
    ext_expr = expr.get_child_or_error(
        "(struct2tensor.test.MyExternalExtension.ext)")
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
        path.Path(
            ["my_any", "(type.googleapis.com/struct2tensor.test.AllSimple)"]))
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
        path.Path([
            "my_any", "(type.googleapis.com/struct2tensor.test.SpecialUserInfo)"
        ]))
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

  def test_create_transformed_field(self):
    expr = proto_test_util._get_expression_from_session_empty_user_info()
    reversed_events_expr = proto.create_transformed_field(
        expr, path.Path(["event"]), "reversed_event", _reverse_values)
    source_events = expr.get_child_or_error("event")
    dest_events = reversed_events_expr.get_child_or_error("reversed_event")
    self.assertTrue(dest_events.is_repeated)
    self.assertFalse(dest_events.is_leaf)
    self.assertEqual(source_events.type, dest_events.type)
    leaf_expr = reversed_events_expr.get_descendant_or_error(
        path.Path(["reversed_event", "action", "doc_id"]))
    leaf_tensor = expression_test_util.calculate_value_slowly(leaf_expr)
    self.assertEqual(leaf_tensor.parent_index.dtype, tf.int64)
    self.assertEqual(leaf_tensor.values.dtype, tf.string)

  def test_create_reversed_field_nested(self):
    expr = proto_test_util._get_expression_from_session_empty_user_info()
    first_reverse = proto.create_transformed_field(expr, path.Path(["event"]),
                                                   "reversed_event",
                                                   _reverse_values)
    second_reverse = proto.create_transformed_field(
        first_reverse, path.Path(["reversed_event", "action"]),
        "reversed_action", _reverse_values)
    leaf_expr = second_reverse.get_descendant_or_error(
        path.Path(["reversed_event", "reversed_action", "doc_id"]))
    leaf_tensor = expression_test_util.calculate_value_slowly(leaf_expr)
    self.assertEqual(leaf_tensor.parent_index.dtype, tf.int64)
    self.assertEqual(leaf_tensor.values.dtype, tf.string)


@test_util.run_all_in_graph_and_eager_modes
class ProtoValuesTest(tf.test.TestCase, parameterized.TestCase):

  def _get_calculate_options(self, use_string_view):
    options = calculate_options.get_default_options()
    options.use_string_view = use_string_view
    return options

  def _check_string_view(self):
    for op in tf.compat.v1.get_default_graph().get_operations():
      if op.type.startswith("DecodeProtoSparse"):
        self.assertLen(op.inputs, 2)
      if op.type.startswith("DecodeProtoMap"):
        self.assertLen(op.inputs, 3)

  @parameterized.named_parameters(("string_view", True),
                                  ("no_string_view", False))
  def test_create_expression_from_proto_and_calculate_root_value(
      self, use_string_view):
    """Tests get_sparse_tensors on a deep tree."""
    expr = proto_test_util._get_expression_from_session_empty_user_info()
    root_value = expression_test_util.calculate_value_slowly(
        expr, options=self._get_calculate_options(use_string_view))
    # For some reason, this fails on tf.eager. It could be because it is
    # a scalar, I don't know.
    self.assertEqual(self.evaluate(root_value.size), 2)
    if use_string_view:
      self._check_string_view()

  @parameterized.named_parameters(("string_view", True),
                                  ("no_string_view", False))
  def test_create_expression_from_proto_and_calculate_event_value(
      self, use_string_view):
    """Tests get_sparse_tensors on a deep tree."""
    expr = proto_test_util._get_expression_from_session_empty_user_info()
    event_value = expression_test_util.calculate_value_slowly(
        expr.get_child_or_error("event"),
        options=self._get_calculate_options(use_string_view))
    self.assertAllEqual(event_value.parent_index, [0, 0, 0, 1, 1])
    if use_string_view:
      self._check_string_view()

  @parameterized.named_parameters(("string_view", True),
                                  ("no_string_view", False))
  def test_create_expression_from_proto_and_calculate_event_id_value(
      self, use_string_view):
    """Tests get_sparse_tensors on a deep tree."""
    expr = proto_test_util._get_expression_from_session_empty_user_info()
    event_id_value = expression_test_util.calculate_value_slowly(
        expr.get_descendant_or_error(path.Path(["event", "event_id"])),
        options=self._get_calculate_options(use_string_view))
    self.assertAllEqual(event_id_value.parent_index, [0, 1, 2, 4])
    self.assertAllEqual(event_id_value.values, [b"A", b"B", b"C", b"D"])
    if use_string_view:
      self._check_string_view()

  @parameterized.named_parameters(("string_view", True),
                                  ("no_string_view", False))
  def test_create_expression_from_proto_with_any(self, use_string_view):
    """Test an any field."""
    expr = _get_expression_with_any()
    simple_expr = expr.get_descendant_or_error(
        path.Path(
            ["my_any", "(type.googleapis.com/struct2tensor.test.AllSimple)"]))
    child_node = expression_test_util.calculate_value_slowly(
        simple_expr,
        options=self._get_calculate_options(use_string_view)).parent_index
    self.assertAllEqual(child_node, [0, 2])

  @parameterized.named_parameters(("string_view", True),
                                  ("no_string_view", False))
  def test_create_expression_from_proto_with_any_missing_message(
      self, use_string_view):
    """Test an any field with a message that is absent."""
    expr = _get_expression_with_any()
    simple_expr = expr.get_descendant_or_error(
        path.Path([
            "my_any", "(type.googleapis.com/struct2tensor.test.SpecialUserInfo)"
        ]))
    child_node = expression_test_util.calculate_value_slowly(
        simple_expr,
        options=self._get_calculate_options(use_string_view)).parent_index
    self.assertAllEqual(child_node, [])

  @parameterized.named_parameters(("string_view", True),
                                  ("no_string_view", False))
  def test_project_proto_map(self, use_string_view):
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
    expr = proto_test_util.text_to_expression(examples, tf.train.Example)
    result = expression_test_util.calculate_list_map(
        expr.project([
            "features.feature[feature1].bytes_list.value",
            "features.feature[feature2].float_list.value",
            "features.feature[feature3].int64_list.value",
        ]),
        self,
        options=self._get_calculate_options(use_string_view))

    feature1 = result["features.feature[feature1].bytes_list.value"]
    feature2 = result["features.feature[feature2].float_list.value"]
    feature3 = result["features.feature[feature3].int64_list.value"]
    self.assertAllEqual(feature1,
                        [[[[[b"hello", b"world"]]]], [[[[b"deadbeef"]]]]])
    self.assertAllEqual(feature2, [[[[[8.0]]]], [[]]])
    self.assertAllEqual(feature3, [[[]], [[[[123, 456]]]]])
    if use_string_view:
      self._check_string_view()

  @parameterized.named_parameters(("string_view", True),
                                  ("no_string_view", False))
  def test_project_proto_map_leaf_value(self, use_string_view):
    protos = [
        """
            int32_string_map {
              key: 222
              value: "2"
            }
            """
    ]

    expr = proto_test_util.text_to_expression(protos,
                                              test_map_pb2.MessageWithMap)
    result = expression_test_util.calculate_list_map(
        expr.project([
            "int32_string_map[222]",
            "int32_string_map[223]",
        ]),
        self,
        options=self._get_calculate_options(use_string_view))
    self.assertLen(result, 2)
    self.assertAllEqual(result["int32_string_map[222]"], [[b"2"]])
    self.assertAllEqual(result["int32_string_map[223]"], [[]])
    if use_string_view:
      self._check_string_view()

  @parameterized.named_parameters(("string_view", True),
                                  ("no_string_view", False))
  def test_transformed_field_values(self, use_string_view):
    expr = proto_test_util._get_expression_from_session_empty_user_info()
    reversed_events_expr = proto.create_transformed_field(
        expr, path.Path(["event"]), "reversed_event", _reverse_values)
    result = expression_test_util.calculate_list_map(
        reversed_events_expr.project(["reversed_event.action.doc_id"]),
        self,
        options=self._get_calculate_options(use_string_view))
    self.assertAllEqual(result["reversed_event.action.doc_id"],
                        [[[[b"h"], [b"i"], [b"j"]], [[b"g"]], [[b"e"], [b"f"]]],
                         [[[b"c"], []], [[b"a"], [b"b"]]]])
    if use_string_view:
      self._check_string_view()

  @parameterized.named_parameters(("string_view", True),
                                  ("no_string_view", False))
  def test_transformed_field_values_with_transformed_parent(
      self, use_string_view):
    expr = proto_test_util._get_expression_from_session_empty_user_info()
    first_reversed_expr = proto.create_transformed_field(
        expr, path.Path(["event"]), "reversed_event", _reverse_values)
    second_reversed_expr = proto.create_transformed_field(
        first_reversed_expr, path.Path(["reversed_event", "action"]),
        "reversed_action", _reverse_values)
    result = expression_test_util.calculate_list_map(
        second_reversed_expr.project(["reversed_event.reversed_action.doc_id"]),
        self,
        options=self._get_calculate_options(use_string_view))
    self.assertAllEqual(result["reversed_event.reversed_action.doc_id"],
                        [[[[b"b"], [b"a"], []], [[b"c"]], [[b"f"], [b"e"]]],
                         [[[b"g"], [b"j"]], [[b"i"], [b"h"]]]])
    if use_string_view:
      self._check_string_view()

  @parameterized.named_parameters(("string_view", True),
                                  ("no_string_view", False))
  def test_transformed_field_values_with_multiple_transforms(
      self, use_string_view):
    expr = proto_test_util._get_expression_from_session_empty_user_info()
    reversed_events_expr = proto.create_transformed_field(
        expr, path.Path(["event"]), "reversed_event", _reverse_values)
    reversed_events_again_expr = proto.create_transformed_field(
        reversed_events_expr, path.Path(["reversed_event"]),
        "reversed_reversed_event", _reverse_values)

    result = expression_test_util.calculate_list_map(
        reversed_events_again_expr.project(
            ["reversed_reversed_event.action.doc_id"]),
        self,
        options=self._get_calculate_options(use_string_view))
    self.assertAllEqual(result["reversed_reversed_event.action.doc_id"],
                        [[[[b"a"], [b"b"]], [[b"c"], []], [[b"e"], [b"f"]]],
                         [[[b"g"]], [[b"h"], [b"i"], [b"j"]]]])
    if use_string_view:
      self._check_string_view()



def _reverse_values(parent_indices, values):
  """A simple function for testing create_transformed_field."""
  return parent_indices, tf.reverse(values, axis=[-1])


if __name__ == "__main__":
  absltest.main()
