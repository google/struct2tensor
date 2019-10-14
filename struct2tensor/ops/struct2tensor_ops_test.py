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
"""Tests for struct2tensor.ops.struct2tensor_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from absl.testing import parameterized

from struct2tensor.ops import struct2tensor_ops
from struct2tensor.test import test_extension_pb2
from struct2tensor.test import test_map_pb2
from struct2tensor.test import test_pb2
import tensorflow as tf


from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import

INDEX = "index"
VALUE = "value"


def _parse_full_message_level_as_dict(proto_list):
  serialized = [proto.SerializeToString() for proto in proto_list]
  parsed_field_list = struct2tensor_ops.parse_full_message_level(
      tf.constant(serialized), proto_list[0].DESCRIPTOR)
  parsed_field_dict = {}
  for parsed_field in parsed_field_list:
    parsed_field_dict[parsed_field.field_name] = parsed_field
  return parsed_field_dict


def _make_dict_runnable(level_as_dict):
  """Prepares output of parse_full_message_level_as_dict for evaluate."""
  result = {}
  for key, value in level_as_dict.items():
    local_dict = {}
    local_dict[INDEX] = value.index
    local_dict[VALUE] = value.value
    result[key] = local_dict
  return result


def _get_full_message_level_runnable(proto_list):
  return _make_dict_runnable(_parse_full_message_level_as_dict(proto_list))


# TODO(martinz): test empty tensors for decode_proto_sparse more thoroughly.
@test_util.run_all_in_graph_and_eager_modes
class PrensorOpsTest(tf.test.TestCase):

  def test_parse_full_message_level_for_event(self):
    event = test_pb2.Event()
    event.event_id = "foo"
    event.query = "query"
    event.query_token.append("a")
    event.query_token.append("b")
    action0 = event.action.add()
    action0.doc_id = "abc"
    action1 = event.action.add()
    event.user_info.age_in_years = 38
    event2 = test_pb2.Event()
    action2 = event2.action.add()
    action2.doc_id = "def"
    parsed_field_dict = _parse_full_message_level_as_dict([event, event2])
    doc_id = parsed_field_dict["action"]
    serialized_actions = [
        proto.SerializeToString() for proto in [action0, action1, action2]
    ]

    self.assertAllEqual(doc_id.index, [0, 0, 1])
    self.assertAllEqual(doc_id.value, serialized_actions)

  def test_parse_full_message_level_for_simple_action_multiple(self):
    """Test multiple messages."""
    as1 = test_pb2.AllSimple()
    as1.optional_string = "a"
    as1.repeated_string.append("b")
    as1.repeated_string.append("c")
    as2 = test_pb2.AllSimple()
    as2.optional_string = "d"
    as2.optional_int32 = 123
    as3 = test_pb2.AllSimple()
    as3.repeated_string.append("d")
    as3.repeated_string.append("e")
    as3.optional_int32 = 123

    parsed_field_dict = _parse_full_message_level_as_dict([as1, as2, as3])

    doc_id = parsed_field_dict["repeated_string"]
    self.assertAllEqual(doc_id.index, [0, 0, 2, 2])
    self.assertAllEqual(doc_id.value, [b"b", b"c", b"d", b"e"])

  def test_parse_full_message_level_for_all_simple_repeated_repeated(self):
    """Test five messages with every possible repeated field repeated."""
    all_simple = test_pb2.AllSimple()
    all_simple.repeated_string.append("foo")
    all_simple.repeated_string.append("foo2")
    all_simple.repeated_int32.append(32)
    all_simple.repeated_int32.append(322)
    all_simple.repeated_uint32.append(123)
    all_simple.repeated_uint32.append(1232)
    all_simple.repeated_int64.append(123456)
    all_simple.repeated_int64.append(1234562)
    all_simple.repeated_uint64.append(123)
    all_simple.repeated_uint64.append(1232)
    all_simple.repeated_float.append(1.0)
    all_simple.repeated_float.append(2.0)
    all_simple.repeated_double.append(1.5)
    all_simple.repeated_double.append(2.5)
    result = _get_full_message_level_runnable([
        all_simple,
        test_pb2.AllSimple(),
        test_pb2.AllSimple(), all_simple,
        test_pb2.AllSimple(), all_simple,
        test_pb2.AllSimple()
    ])
    self.assertAllEqual(result["repeated_string"][INDEX], [0, 0, 3, 3, 5, 5])
    self.assertAllEqual(result["repeated_string"][VALUE],
                        [b"foo", b"foo2", b"foo", b"foo2", b"foo", b"foo2"])
    self.assertAllEqual(result["repeated_int32"][INDEX], [0, 0, 3, 3, 5, 5])
    self.assertAllEqual(result["repeated_int32"][VALUE],
                        [32, 322, 32, 322, 32, 322])
    self.assertAllEqual(result["repeated_uint32"][INDEX], [0, 0, 3, 3, 5, 5])
    self.assertAllEqual(result["repeated_uint32"][VALUE],
                        [123, 1232, 123, 1232, 123, 1232])
    self.assertAllEqual(result["repeated_int64"][INDEX], [0, 0, 3, 3, 5, 5])
    self.assertAllEqual(result["repeated_int64"][VALUE],
                        [123456, 1234562, 123456, 1234562, 123456, 1234562])
    self.assertAllEqual(result["repeated_uint64"][INDEX], [0, 0, 3, 3, 5, 5])
    self.assertAllEqual(result["repeated_uint64"][VALUE],
                        [123, 1232, 123, 1232, 123, 1232])
    self.assertAllEqual(result["repeated_float"][INDEX], [0, 0, 3, 3, 5, 5])
    self.assertAllEqual(result["repeated_float"][VALUE],
                        [1.0, 2.0, 1.0, 2.0, 1.0, 2.0])
    self.assertAllEqual(result["repeated_double"][INDEX], [0, 0, 3, 3, 5, 5])
    self.assertAllEqual(result["repeated_double"][VALUE],
                        [1.5, 2.5, 1.5, 2.5, 1.5, 2.5])

  def test_parse_full_message_level_for_all_simple_repeated(self):
    """Test a single message with every possible repeated field repeated."""
    all_simple = test_pb2.AllSimple()
    all_simple.repeated_string.append("foo")
    all_simple.repeated_string.append("foo2")
    all_simple.repeated_int32.append(32)
    all_simple.repeated_int32.append(322)
    all_simple.repeated_uint32.append(123)
    all_simple.repeated_uint32.append(1232)
    all_simple.repeated_int64.append(123456)
    all_simple.repeated_int64.append(1234562)
    all_simple.repeated_uint64.append(123)
    all_simple.repeated_uint64.append(1232)
    all_simple.repeated_float.append(1.0)
    all_simple.repeated_float.append(2.0)
    all_simple.repeated_double.append(1.5)
    all_simple.repeated_double.append(2.5)
    result = _get_full_message_level_runnable([all_simple])
    self.assertAllEqual(result["repeated_string"][INDEX], [0, 0])
    self.assertAllEqual(result["repeated_string"][VALUE], [b"foo", b"foo2"])
    self.assertAllEqual(result["repeated_int32"][INDEX], [0, 0])
    self.assertAllEqual(result["repeated_int32"][VALUE], [32, 322])
    self.assertAllEqual(result["repeated_uint32"][INDEX], [0, 0])
    self.assertAllEqual(result["repeated_uint32"][VALUE], [123, 1232])
    self.assertAllEqual(result["repeated_int64"][INDEX], [0, 0])
    self.assertAllEqual(result["repeated_int64"][VALUE], [123456, 1234562])
    self.assertAllEqual(result["repeated_uint64"][INDEX], [0, 0])
    self.assertAllEqual(result["repeated_uint64"][VALUE], [123, 1232])
    self.assertAllEqual(result["repeated_float"][INDEX], [0, 0])
    self.assertAllEqual(result["repeated_float"][VALUE], [1.0, 2.0])
    self.assertAllEqual(result["repeated_double"][INDEX], [0, 0])
    self.assertAllEqual(result["repeated_double"][VALUE], [1.5, 2.5])

  def test_parse_full_message_level_for_all_simple(self):
    """Test a single message with every possible primitive field."""
    all_simple = test_pb2.AllSimple()
    all_simple.optional_string = "foo"
    all_simple.optional_int32 = -5
    all_simple.optional_uint32 = 2**31
    all_simple.optional_int64 = 100123
    all_simple.optional_uint64 = 2**63
    all_simple.optional_float = 6.5
    all_simple.optional_double = -7.0
    all_simple.repeated_string.append("foo")
    all_simple.repeated_int32.append(32)
    all_simple.repeated_uint32.append(123)
    all_simple.repeated_int64.append(123456)
    all_simple.repeated_uint64.append(123)
    all_simple.repeated_float.append(1.0)
    all_simple.repeated_double.append(1.5)
    runnable = _get_full_message_level_runnable([all_simple])
    self.assertEqual(len(runnable["optional_string"][INDEX].shape.dims), 1)
    self.assertEqual(len(runnable["optional_string"][VALUE].shape.dims), 1)
    self.assertEqual(len(runnable["repeated_string"][INDEX].shape.dims), 1)
    self.assertEqual(len(runnable["repeated_string"][VALUE].shape.dims), 1)

    result = runnable
    self.assertAllEqual(result["optional_string"][INDEX], [0])
    self.assertAllEqual(result["optional_string"][VALUE], [b"foo"])
    self.assertAllEqual(result["optional_int32"][INDEX], [0])
    self.assertAllEqual(result["optional_int32"][VALUE], [-5])
    self.assertAllEqual(result["optional_uint32"][INDEX], [0])
    self.assertAllEqual(result["optional_uint32"][VALUE], [2**31])
    self.assertAllEqual(result["optional_int64"][INDEX], [0])
    self.assertAllEqual(result["optional_int64"][VALUE], [100123])
    self.assertAllEqual(result["optional_uint64"][INDEX], [0])
    self.assertAllEqual(result["optional_uint64"][VALUE], [2**63])
    self.assertAllEqual(result["optional_float"][INDEX], [0])
    self.assertAllEqual(result["optional_float"][VALUE], [6.5])
    self.assertAllEqual(result["optional_double"][INDEX], [0])
    self.assertAllEqual(result["optional_double"][VALUE], [-7.0])
    # TODO(martinz): test the repeated fields too.

  def test_parse_full_message_level_action(self):
    action = test_pb2.Action()
    action.doc_id = "3"
    action.number_of_views = 3
    result = _get_full_message_level_runnable([action])
    self.assertAllEqual(result["doc_id"][INDEX], [0])
    self.assertAllEqual(result["doc_id"][VALUE], [b"3"])
    self.assertAllEqual(result["number_of_views"][INDEX], [0])
    self.assertAllEqual(result["number_of_views"][VALUE], [3])

  def test_parse_message_level(self):
    action = test_pb2.Action()
    action.doc_id = "3"
    action.number_of_views = 3
    tensor_of_protos = tf.constant([action.SerializeToString()])
    [field_tuple
    ] = struct2tensor_ops.parse_message_level(tensor_of_protos,
                                              test_pb2.Action().DESCRIPTOR,
                                              ["number_of_views"])
    values = field_tuple.value
    indices = field_tuple.index
    self.assertAllEqual(indices, [0])
    self.assertAllEqual(values, [3])

  def test_parse_extension(self):
    user_info = test_pb2.UserInfo()
    user_info.Extensions[
        test_pb2.LocationOfExtension.special_user_info].secret = "shhh"
    expected_value = test_pb2.SpecialUserInfo()
    expected_value.secret = "shhh"
    tensor_of_protos = tf.constant([user_info.SerializeToString()])
    [field_tuple] = struct2tensor_ops.parse_message_level(
        tensor_of_protos,
        test_pb2.UserInfo().DESCRIPTOR,
        ["(struct2tensor.test.LocationOfExtension.special_user_info)"])
    self.assertAllEqual(field_tuple.index, [0])
    self.assertAllEqual(field_tuple.value, [expected_value.SerializeToString()])

  def test_parse_external_extension(self):
    user_info = test_pb2.UserInfo()
    user_info.Extensions[
        test_extension_pb2.MyExternalExtension.ext].special = "shhh"
    expected_value = test_extension_pb2.MyExternalExtension()
    expected_value.special = "shhh"
    tensor_of_protos = tf.constant([user_info.SerializeToString()])
    [field_tuple] = struct2tensor_ops.parse_message_level(
        tensor_of_protos,
        test_pb2.UserInfo().DESCRIPTOR,
        ["(struct2tensor.test.MyExternalExtension.ext)"])
    self.assertAllEqual(field_tuple.index, [0])
    self.assertAllEqual(field_tuple.value, [expected_value.SerializeToString()])


  def test_parse_packed_fields(self):
    message_with_packed_fields = test_pb2.HasPackedFields(
        packed_int32=[-1, -2, -3],
        packed_uint32=[100000, 200000, 300000],
        packed_int64=[-400000, -500000, -600000],
        packed_uint64=[4, 5, 6],
        packed_float=[7.0, 8.0, 9.0],
        packed_double=[10.0, 11.0, 12.0],
    )
    tensor_of_protos = tf.constant(
        [message_with_packed_fields.SerializeToString()] * 2)

    parsed_tuples = struct2tensor_ops.parse_message_level(
        tensor_of_protos, test_pb2.HasPackedFields.DESCRIPTOR, [
            "packed_int32",
            "packed_uint32",
            "packed_int64",
            "packed_uint64",
            "packed_float",
            "packed_double",
        ])
    indices = {
        parsed_tuple.field_name: parsed_tuple.index
        for parsed_tuple in parsed_tuples
    }
    values = {
        parsed_tuple.field_name: parsed_tuple.value
        for parsed_tuple in parsed_tuples
    }

    for index in indices.values():
      self.assertAllEqual(index, [0, 0, 0, 1, 1, 1])
    for field_name, value in values.items():
      self.assertAllEqual(
          value,
          list(getattr(message_with_packed_fields, field_name)) * 2)

  def test_make_repeated_basic(self):
    parent_index = tf.constant([0, 0, 4, 4, 4, 7, 8, 9], dtype=tf.int64)
    values = tf.constant(["a", "b", "c", "d", "e", "f", "g", "h"])
    sparse_tensor = struct2tensor_ops.create_sparse_tensor_for_repeated(
        parent_index, values, tf.constant([10, 3], dtype=tf.int64))
    self.assertAllEqual(
        sparse_tensor.indices,
        [[0, 0], [0, 1], [4, 0], [4, 1], [4, 2], [7, 0], [8, 0], [9, 0]])

  def test_make_repeated_empty(self):
    parent_index = tf.constant([], dtype=tf.int64)
    values = tf.constant([], dtype=tf.int32)
    sparse_tensor = struct2tensor_ops.create_sparse_tensor_for_repeated(
        parent_index, values, tf.constant([0, 0], dtype=tf.int64))
    self.assertAllEqual(sparse_tensor.indices.shape, [0, 2])

  def test_equi_join_indices(self):
    a = tf.constant([0, 0, 1, 1, 2, 3, 4], dtype=tf.int64)
    b = tf.constant([0, 0, 2, 2, 3], dtype=tf.int64)
    [index_a, index_b] = struct2tensor_ops.equi_join_indices(a, b)
    self.assertAllEqual(index_a, [0, 0, 1, 1, 4, 4, 5])
    self.assertAllEqual(index_b, [0, 1, 0, 1, 2, 3, 4])

  def test_equi_join_indices_2(self):
    a = tf.constant([0, 1, 1, 2], dtype=tf.int64)
    b = tf.constant([0, 1, 2], dtype=tf.int64)
    [index_a, index_b] = struct2tensor_ops.equi_join_indices(a, b)
    self.assertAllEqual(index_a, [0, 1, 2, 3])
    self.assertAllEqual(index_b, [0, 1, 1, 2])

  def test_equi_join_indices_empty_a(self):
    a = tf.constant([], dtype=tf.int64)
    b = tf.constant([0, 1, 2], dtype=tf.int64)
    [index_a, index_b] = struct2tensor_ops.equi_join_indices(a, b)
    self.assertAllEqual(index_a, [])
    self.assertAllEqual(index_b, [])

  def test_equi_join_indices_empty_b(self):
    a = tf.constant([0, 1, 1, 2], dtype=tf.int64)
    b = tf.constant([], dtype=tf.int64)
    [index_a, index_b] = struct2tensor_ops.equi_join_indices(a, b)
    self.assertAllEqual(index_a, [])
    self.assertAllEqual(index_b, [])

  def test_equi_join_indices_both_empty(self):
    a = tf.constant([], dtype=tf.int64)
    b = tf.constant([], dtype=tf.int64)
    [index_a, index_b] = struct2tensor_ops.equi_join_indices(a, b)
    self.assertAllEqual(index_a, [])
    self.assertAllEqual(index_b, [])

  def test_equi_join_indices_no_overlap(self):
    a = tf.constant([0, 1, 1, 2], dtype=tf.int64)
    b = tf.constant([3, 4, 5], dtype=tf.int64)
    [index_a, index_b] = struct2tensor_ops.equi_join_indices(a, b)
    self.assertAllEqual(index_a, [])
    self.assertAllEqual(index_b, [])

  def test_equi_join_indices_for_broadcast(self):
    """Breaking down the broadcast."""
    a = tf.constant([0, 1, 1], dtype=tf.int64)
    b = tf.constant([0, 1, 2], dtype=tf.int64)
    [index_a, index_b] = struct2tensor_ops.equi_join_indices(a, b)
    self.assertAllEqual(index_a, [0, 1, 2])
    self.assertAllEqual(index_b, [0, 1, 1])

  def test_run_length_before(self):
    """Breaking down the broadcast."""
    a = tf.constant([0, 1, 1, 7, 8, 8, 9], dtype=tf.int64)
    b = struct2tensor_ops.run_length_before(a)
    self.assertAllEqual(b, [0, 0, 1, 0, 0, 1, 0])

  def test_run_length_before_empty(self):
    """Breaking down the broadcast."""
    a = tf.constant([], dtype=tf.int64)
    b = struct2tensor_ops.run_length_before(a)
    self.assertAllEqual(b, [])

_SIGNED_INTEGER_TYPES = [
    "int32", "int64", "sfixed32", "sfixed64", "sint32", "sint64"
]

_UNSIGNED_INTEGER_TYPES = ["uint32", "uint64", "fixed32", "fixed64"]


@test_util.run_all_in_graph_and_eager_modes
class DecodeProtoMapOpTest(parameterized.TestCase, tf.test.TestCase):

  def _parse_map_entry(self, messages_with_map, map_field_name, keys_needed):
    parsed_map_submessage = struct2tensor_ops.parse_message_level(
        tf.constant([m.SerializeToString() for m in messages_with_map]),
        test_map_pb2.MessageWithMap.DESCRIPTOR, [map_field_name])[0]

    return struct2tensor_ops.parse_proto_map(
        parsed_map_submessage.value, parsed_map_submessage.index,
        parsed_map_submessage.field_descriptor.message_type, keys_needed)

  @parameterized.named_parameters(
      [dict(testcase_name=t, key_type=t) for t in _SIGNED_INTEGER_TYPES])
  def test_signed_integer_key_types(self, key_type):
    field_name = "{}_string_map".format(key_type)
    message_with_map = test_map_pb2.MessageWithMap()
    map_entry = getattr(message_with_map, "{}_string_map".format(key_type))
    map_entry[42] = "hello"
    map_entry[-42] = "world"

    [(values_42, indices_42), (values_n42, indices_n42),
     (values_0, indices_0)] = self._parse_map_entry([message_with_map],
                                                    field_name,
                                                    ["42", "-42", "0"])

    self.assertAllEqual(values_42, [b"hello"])
    self.assertAllEqual(values_n42, [b"world"])
    self.assertAllEqual(values_0, [])
    self.assertAllEqual(indices_42, [0])
    self.assertAllEqual(indices_n42, [0])
    self.assertAllEqual(indices_0, [])

  @parameterized.named_parameters(
      [dict(testcase_name=t, key_type=t) for t in _UNSIGNED_INTEGER_TYPES])
  def test_unsigned_integer_key_types(self, key_type):
    field_name = "{}_string_map".format(key_type)
    message_with_map = test_map_pb2.MessageWithMap()
    map_entry = getattr(message_with_map, "{}_string_map".format(key_type))
    map_entry[42] = "hello"

    [(values_42, indices_42),
     (values_0, indices_0)] = self._parse_map_entry([message_with_map],
                                                    field_name, ["42", "0"])
    self.assertAllEqual(values_42, [b"hello"])
    self.assertAllEqual(values_0, [])
    self.assertAllEqual(indices_42, [0])
    self.assertAllEqual(indices_0, [])

  def test_invalid_uint32_key(self):
    with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                 "Failed to parse .*string"):
      self.evaluate(
          self._parse_map_entry([test_map_pb2.MessageWithMap()],
                                "uint32_string_map", ["-42"]))

  def test_invalid_int32_key(self):
    with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                 "Failed to parse .*string"):
      self.evaluate(
          self._parse_map_entry([test_map_pb2.MessageWithMap()],
                                "int32_string_map", ["foo"]))

  def test_bool_key_type(self):
    message_with_map = test_map_pb2.MessageWithMap()
    message_with_map.bool_string_map[False] = "hello"
    [(values_false, indices_false), (values_true, indices_true)
    ] = self._parse_map_entry([message_with_map], "bool_string_map", ["0", "1"])
    self.assertAllEqual(values_true, [])
    self.assertAllEqual(values_false, [b"hello"])
    self.assertAllEqual(indices_true, [])
    self.assertAllEqual(indices_false, [0])

  def test_invalid_bool_key(self):
    message_with_map = test_map_pb2.MessageWithMap()
    with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                 "Failed to parse .*string"):
      self.evaluate(
          self._parse_map_entry([message_with_map], "bool_string_map", ["2"]))

  @parameterized.named_parameters(
      [dict(testcase_name=t, value_type=t) for t in _SIGNED_INTEGER_TYPES])
  def test_signed_integer_value_types(self, value_type):
    field_name = "string_{}_map".format(value_type)
    message_with_map = test_map_pb2.MessageWithMap()
    map_entry = getattr(message_with_map, "string_{}_map".format(value_type))
    map_entry["foo"] = 42
    map_entry["bar"] = -42
    [(values_foo, indices_foo), (values_bar, indices_bar),
     (values_null, indices_null)] = self._parse_map_entry([message_with_map],
                                                          field_name,
                                                          ["foo", "bar", ""])
    self.assertAllEqual(values_foo, [42])
    self.assertAllEqual(values_bar, [-42])
    self.assertAllEqual(values_null, [])
    self.assertAllEqual(indices_foo, [0])
    self.assertAllEqual(indices_bar, [0])
    self.assertAllEqual(indices_null, [])

  @parameterized.named_parameters(
      [dict(testcase_name=t, value_type=t) for t in _UNSIGNED_INTEGER_TYPES])
  def test_unsigned_integer_value_types(self, value_type):
    field_name = "string_{}_map".format(value_type)
    message_with_map = test_map_pb2.MessageWithMap()
    map_entry = getattr(message_with_map, "string_{}_map".format(value_type))
    map_entry["foo"] = 42
    [(values_foo, indices_foo), (values_null, indices_null)
    ] = self._parse_map_entry([message_with_map], field_name, ["foo", ""])
    self.assertAllEqual(values_foo, [42])
    self.assertAllEqual(values_null, [])
    self.assertAllEqual(indices_foo, [0])
    self.assertAllEqual(indices_null, [])

  @parameterized.named_parameters(
      [dict(testcase_name=t, value_type=t) for t in ["float", "double"]])
  def test_fp_value_types(self, value_type):
    field_name = "string_{}_map".format(value_type)
    message_with_map = test_map_pb2.MessageWithMap()
    map_entry = getattr(message_with_map, "string_{}_map".format(value_type))
    map_entry["foo"] = 0.5
    [(values_foo, indices_foo), (values_null, indices_null)
    ] = self._parse_map_entry([message_with_map], field_name, ["foo", ""])
    self.assertAllEqual(values_foo, [0.5])
    self.assertAllEqual(values_null, [])
    self.assertAllEqual(indices_foo, [0])
    self.assertAllEqual(indices_null, [])

  def test_enum_value_type(self):
    message_with_map = test_map_pb2.MessageWithMap()
    message_with_map.string_enum_map["foo"] = test_map_pb2.BAZ
    [(values_foo, indices_foo),
     (values_null, indices_null)] = self._parse_map_entry([message_with_map],
                                                          "string_enum_map",
                                                          ["foo", ""])
    self.assertAllEqual(values_foo, [int(test_map_pb2.BAZ)])
    self.assertAllEqual(values_null, [])
    self.assertAllEqual(indices_foo, [0])
    self.assertAllEqual(indices_null, [])

  def test_message_value_type(self):
    sub_message = test_map_pb2.SubMessage(repeated_int64=[1, 2, 3])
    message_with_map = test_map_pb2.MessageWithMap()
    message_with_map.string_message_map["foo"].MergeFrom(sub_message)
    [(values_foo, indices_foo),
     (values_null, indices_null)] = self._parse_map_entry([message_with_map],
                                                          "string_message_map",
                                                          ["foo", ""])
    self.assertAllEqual(values_foo, [sub_message.SerializeToString()])
    self.assertAllEqual(values_null, [])
    self.assertAllEqual(indices_foo, [0])
    self.assertAllEqual(indices_null, [])

  def test_multiple_messages(self):
    message_with_map1 = test_map_pb2.MessageWithMap(string_string_map={
        "key1": "foo",
        "key3": "bar"
    })
    message_with_map2 = test_map_pb2.MessageWithMap()
    message_with_map3 = test_map_pb2.MessageWithMap(string_string_map={
        "key2": "baz",
        "key1": "kaz"
    })
    [(values_key1, indices_key1), (values_key2, indices_key2),
     (values_key3, indices_key3)] = self._parse_map_entry(
         [message_with_map1, message_with_map2, message_with_map3],
         "string_string_map", ["key1", "key2", "key3"])
    self.assertAllEqual(values_key1, [b"foo", b"kaz"])
    self.assertAllEqual(values_key2, [b"baz"])
    self.assertAllEqual(values_key3, [b"bar"])
    self.assertAllEqual(indices_key1, [0, 2])
    self.assertAllEqual(indices_key2, [2])
    self.assertAllEqual(indices_key3, [0])

  def test_corrupted_message(self):
    with self.assertRaises(tf.errors.DataLossError):
      self.evaluate(
          struct2tensor_ops.parse_proto_map(
              tf.constant(["corrupted message"]),
              tf.constant([0], dtype=tf.int64), test_map_pb2.MessageWithMap
              .DESCRIPTOR.fields_by_name["int32_string_map"].message_type,
              ["0"]))


if __name__ == "__main__":
  absltest.main()
