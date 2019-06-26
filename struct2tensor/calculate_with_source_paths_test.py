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
"""Tests for calculate_with_source_paths."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from struct2tensor import calculate_with_source_paths
from struct2tensor import path
from struct2tensor.expression_impl import promote
from struct2tensor.expression_impl import proto
from struct2tensor.expression_impl import proto_test_util
from struct2tensor.proto import query_metadata_pb2
from struct2tensor.test import test_any_pb2
from struct2tensor.test import test_pb2
import tensorflow as tf

from google.protobuf import text_format


class CalculateWithSourcePathsTest(tf.test.TestCase):

  def assertLen(self, arr, expected_len):
    self.assertEqual(len(arr), expected_len)

  # Doesn't check number of occurrences.
  # E.g.: ["a", "a", "b"] == ["a", "b"]
  def equal_ignore_order(self, a, b):
    debug_string = "[{}] vs [{}]".format(
        ",".join(['"{}"'.format(str(ae)) for ae in a]),
        ",".join(['"{}"'.format(str(be)) for be in b]))
    for ae in a:
      self.assertIn(ae, b, debug_string)
    for be in b:
      self.assertIn(be, a, debug_string)

  def test_dedup_paths(self):
    paths = [path.Path(["a", "b"]), path.Path(["a"])]
    result = calculate_with_source_paths._dedup_paths(paths)
    self.equal_ignore_order([path.Path(["a", "b"])], result)

  def test_calculate_prensors_with_source_paths(self):
    """Tests get_sparse_tensors on a deep tree."""

    with self.session(use_gpu=False) as sess:
      expr = proto_test_util._get_expression_from_session_empty_user_info()
      # Let's make it non-trivial by transforming the data.
      new_root = promote.promote(expr, path.Path(["event", "action", "doc_id"]),
                                 "action_doc_ids")
      # A poor-man's reroot.
      new_field = new_root.get_descendant_or_error(
          path.Path(["event", "action_doc_ids"]))
      result = calculate_with_source_paths.calculate_prensors_with_source_paths(
          [new_field])
      prensor_result, proto_summary_result = result
      self.assertLen(prensor_result, 1)
      self.assertLen(proto_summary_result, 1)
      leaf_node = prensor_result[0].node
      [parent_index,
       values] = sess.run([leaf_node.parent_index, leaf_node.values])
      self.assertAllEqual(parent_index, [0, 0, 1, 2, 2, 3, 4, 4, 4])
      self.assertAllEqual(
          values, [b"a", b"b", b"c", b"e", b"f", b"g", b"h", b"i", b"j"])
      list_of_paths = proto_summary_result[0].paths
      expected = [path.Path(["event", "action", "doc_id"])]
      self.equal_ignore_order(list_of_paths, expected)

  def test_requirements_to_metadata_proto(self):
    proto_summary_result_0 = calculate_with_source_paths.ProtoRequirements(
        None, test_pb2.Session.DESCRIPTOR, [
            path.Path(["event", "action", "doc_id"]),
            path.Path(["event", "event_id"])
        ])
    proto_summary_result_1 = calculate_with_source_paths.ProtoRequirements(
        None, test_pb2.Event.DESCRIPTOR,
        [path.Path(["action", "doc_id"]),
         path.Path(["event_id"])])

    result = query_metadata_pb2.QueryMetadata()
    calculate_with_source_paths.requirements_to_metadata_proto(
        [proto_summary_result_0, proto_summary_result_1], result)
    self.assertLen(result.parsed_proto_info, 2)
    expected_result = text_format.Parse(
        """
        parsed_proto_info {
          message_name: "struct2tensor.test.Session"
          field_paths {
            step: "event"
            step: "action"
            step: "doc_id"
          }
          field_paths {
            step: "event"
            step: "event_id"
          }
        }
        parsed_proto_info {
          message_name: "struct2tensor.test.Event"
          field_paths {
            step: "action"
            step: "doc_id"
          }
          field_paths {
            step: "event_id"
          }
        }""", query_metadata_pb2.QueryMetadata())
    self.assertEqual(result, expected_result)

  def test_any_path(self):
    with self.session(use_gpu=False) as sess:

      my_any_0 = test_any_pb2.MessageWithAny()
      my_value_0 = test_pb2.AllSimple()
      my_value_0.optional_int32 = 17
      my_any_0.my_any.Pack(my_value_0)
      expr = proto.create_expression_from_proto(
          [my_any_0.SerializeToString()],
          test_any_pb2.MessageWithAny.DESCRIPTOR)
      new_root = promote.promote(
          expr,
          path.Path(
              ["my_any", "(struct2tensor.test.AllSimple)", "optional_int32"]),
          "new_int32")
      new_field = new_root.get_descendant_or_error(
          path.Path(["my_any", "new_int32"]))
      result = calculate_with_source_paths.calculate_prensors_with_source_paths(
          [new_field])
      prensor_result, proto_summary_result = result
      self.assertLen(prensor_result, 1)
      self.assertLen(proto_summary_result, 1)
      leaf_node = prensor_result[0].node
      [parent_index,
       values] = sess.run([leaf_node.parent_index, leaf_node.values])
      self.assertAllEqual(parent_index, [0])
      self.assertAllEqual(values, [17])
      list_of_paths = proto_summary_result[0].paths
      expected = [
          path.Path(
              ["my_any", "(struct2tensor.test.AllSimple)", "optional_int32"])
      ]
      self.equal_ignore_order(list_of_paths, expected)


if __name__ == "__main__":
  absltest.main()
