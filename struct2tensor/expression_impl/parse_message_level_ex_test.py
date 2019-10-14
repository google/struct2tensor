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
"""Tests for struct2tensor.expression_impl.parse_message_level_ex.

Since parse_message_level_ex wraps functionality provided elsewhere, the
tests here are lightweight to make sure all the connections are present and
work correctly.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from struct2tensor.expression_impl import parse_message_level_ex
from struct2tensor.test import test_any_pb2
from struct2tensor.test import test_map_pb2
from struct2tensor.test import test_pb2
import tensorflow as tf

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import

_INDEX = "index"
_VALUE = "value"
_TYPE_URL = "type_url"
_ALLSIMPLE_NO_PARENS = b"type.googleapis.com/struct2tensor.test.AllSimple"
_ALLSIMPLE = "(type.googleapis.com/struct2tensor.test.AllSimple)"
_USERINFO_NO_PARENS = b"type.googleapis.com/struct2tensor.test.UserInfo"
_USERINFO = "(type.googleapis.com/struct2tensor.test.UserInfo)"


def _run_parse_message_level_ex(proto_list, fields):
  serialized = [x.SerializeToString() for x in proto_list]
  parsed_field_dict = parse_message_level_ex.parse_message_level_ex(
      tf.constant(serialized), proto_list[0].DESCRIPTOR, fields)
  sess_input = {}
  for key, value in parsed_field_dict.items():
    local_dict = {}
    local_dict[_INDEX] = value.index
    local_dict[_VALUE] = value.value
    sess_input[key] = local_dict
  return sess_input


def _create_any(x):
  """Create an any object from a protobuf."""
  new_any = test_any_pb2.MessageWithAny().my_any
  new_any.Pack(x)
  return new_any


def _create_any_protos():
  my_value_0 = test_pb2.AllSimple()
  my_value_0.optional_int32 = 0
  my_value_1 = test_pb2.UserInfo()
  my_value_2 = test_pb2.AllSimple()
  my_value_2.optional_int32 = 20
  return [_create_any(x) for x in [my_value_0, my_value_1, my_value_2]]


def _get_optional_int32(serialized_all_simple):
  """Take a serialized test_pb2.AllSimple object and extract optional_int32."""
  holder = test_pb2.AllSimple()
  holder.ParseFromString(serialized_all_simple)
  return holder.optional_int32


def _get_empty_all_simple():
  """Take a serialized test_pb2.AllSimple object and extract optional_int32."""
  return test_pb2.AllSimple().SerializeToString()


@test_util.run_all_in_graph_and_eager_modes
class ParseMessageLevelExTest(tf.test.TestCase):

  def test_any_field(self):
    original_protos = _create_any_protos()
    result = _run_parse_message_level_ex(original_protos, {_ALLSIMPLE})
    self.assertIn(_ALLSIMPLE, result)
    self.assertIn(_INDEX, result[_ALLSIMPLE])
    self.assertIn(_VALUE, result[_ALLSIMPLE])

    self.assertAllEqual(result[_ALLSIMPLE][_INDEX], [0, 2])
    result_optional_int32 = [
        _get_optional_int32(x)
        for x in self.evaluate(result[_ALLSIMPLE][_VALUE])
    ]
    self.assertAllEqual(result_optional_int32, [0, 20])

  def test_any_field_no_special(self):
    result = _run_parse_message_level_ex(_create_any_protos(), {_TYPE_URL})
    self.assertIn(_TYPE_URL, result)
    self.assertAllEqual(result[_TYPE_URL][_INDEX], [0, 1, 2])

    self.assertAllEqual(
        result[_TYPE_URL][_VALUE],
        [_ALLSIMPLE_NO_PARENS, _USERINFO_NO_PARENS, _ALLSIMPLE_NO_PARENS])

  def test_any_field_special_and_type_url(self):
    result = _run_parse_message_level_ex(_create_any_protos(),
                                         {_TYPE_URL, _ALLSIMPLE})
    self.assertIn(_TYPE_URL, result)
    self.assertAllEqual(result[_TYPE_URL][_INDEX], [0, 1, 2])

    self.assertAllEqual(
        result[_TYPE_URL][_VALUE],
        [_ALLSIMPLE_NO_PARENS, _USERINFO_NO_PARENS, _ALLSIMPLE_NO_PARENS])

    self.assertAllEqual(result[_ALLSIMPLE][_INDEX], [0, 2])
    actual_values = [
        _get_optional_int32(x)
        for x in self.evaluate(result[_ALLSIMPLE][_VALUE])
    ]
    self.assertAllEqual(actual_values, [0, 20])

  def test_full_name_from_any_step(self):
    self.assertEqual(
        parse_message_level_ex.get_full_name_from_any_step(_ALLSIMPLE),
        "struct2tensor.test.AllSimple")
    self.assertEqual(
        parse_message_level_ex.get_full_name_from_any_step(_USERINFO),
        "struct2tensor.test.UserInfo")
    self.assertIsNone(
        parse_message_level_ex.get_full_name_from_any_step("broken"))
    self.assertIsNone(
        parse_message_level_ex.get_full_name_from_any_step("(broken"))
    self.assertIsNone(
        parse_message_level_ex.get_full_name_from_any_step("broken)"))

  def test_normal_field(self):
    """Test three messages with a repeated string."""
    all_simple = test_pb2.AllSimple()
    all_simple.repeated_string.append("foo")
    all_simple.repeated_string.append("foo2")
    all_simple_empty = test_pb2.AllSimple()

    result = _run_parse_message_level_ex(
        [all_simple, all_simple_empty, all_simple, all_simple],
        {"repeated_string"})
    self.assertNotIn("repeated_bool", result)

    self.assertAllEqual(result["repeated_string"][_INDEX], [0, 0, 2, 2, 3, 3])
    self.assertAllEqual(result["repeated_string"][_VALUE],
                        [b"foo", b"foo2", b"foo", b"foo2", b"foo", b"foo2"])

  def test_bool_key_type(self):
    map_field = "bool_string_map[1]"
    message_with_map_0 = test_map_pb2.MessageWithMap()
    message_with_map_0.bool_string_map[False] = "hello"
    message_with_map_1 = test_map_pb2.MessageWithMap()
    message_with_map_1.bool_string_map[True] = "goodbye"
    result = _run_parse_message_level_ex(
        [message_with_map_0, message_with_map_1], {map_field})
    self.assertIn(map_field, result)
    self.assertAllEqual(result[map_field][_VALUE], [b"goodbye"])
    self.assertAllEqual(result[map_field][_INDEX], [1])


if __name__ == "__main__":
  absltest.main()
