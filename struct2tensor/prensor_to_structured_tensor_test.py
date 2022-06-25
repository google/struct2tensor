# Copyright 2020 Google LLC
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
"""Tests for StructuredTensor."""

from google.protobuf import text_format
from struct2tensor import calculate
from struct2tensor import prensor
from struct2tensor import prensor_to_structured_tensor
from struct2tensor.expression_impl import proto
from struct2tensor.test import prensor_test_util
from struct2tensor.test import test_pb2
import tensorflow as tf


# @test_util.run_all_in_graph_and_eager_modes
class PrensorToStructuredTensorTest(tf.test.TestCase):

  def test_simple_prensor(self):
    pren = prensor_test_util.create_simple_prensor()
    st = prensor_to_structured_tensor.prensor_to_structured_tensor(pren)
    self.assertAllEqual(st.field_value("foo"), [[9], [8], [7]])
    self.assertAllEqual(st.field_value("foorepeated"), [[9], [8, 7], [6]])

  def test_nested_prensor(self):
    """Tests on a deep expression."""
    pren = prensor_test_util.create_nested_prensor()
    st = prensor_to_structured_tensor.prensor_to_structured_tensor(pren)
    self.assertAllEqual(
        st.field_value(["doc", "bar"]), [[[b"a"]], [[b"b", b"c"], [b"d"]], []])
    self.assertAllEqual(
        st.field_value(["doc", "keep_me"]), [[[False]], [[True], []], []])
    self.assertAllEqual(
        st.field_value(["user", "friends"]),
        [[[b"a"]], [[b"b", b"c"], [b"d"]], [[b"e"]]])

  def test_big_prensor(self):
    """Test the big prensor.

      a prensor expression representing:
      {foo:9, foorepeated:[9], doc:[{bar:["a"], keep_me:False}],
      user:[{friends:["a"]}]}
      {foo:8, foorepeated:[8,7],
      doc:[{bar:["b","c"],keep_me:True},{bar:["d"]}],
      user:[{friends:["b", "c"]},{friends:["d"]}],}
      {foo:7, foorepeated:[6], user:[friends:["e"]]}
    """
    pren = prensor_test_util.create_big_prensor()
    st = prensor_to_structured_tensor.prensor_to_structured_tensor(pren)
    self.assertAllEqual(st.field_value("foo"), [[9], [8], [7]])
    self.assertAllEqual(st.field_value("foorepeated"), [[9], [8, 7], [6]])
    self.assertAllEqual(
        st.field_value(["doc", "keep_me"]), [[[False]], [[True], []], []])
    self.assertAllEqual(
        st.field_value(["user", "friends"]),
        [[[b"a"]], [[b"b", b"c"], [b"d"]], [[b"e"]]])
    self.assertAllEqual(
        st.field_value(["doc", "bar"]), [[[b"a"]], [[b"b", b"c"], [b"d"]], []])

  def test_deep_prensor(self):
    """Test a prensor with three layers: root, event, and doc.

      a prensor expression representing:
      {foo:9, foorepeated:[9], user:[{friends:["a"]}],
       event:{doc:[{bar:["a"], keep_me:False}]}}
      {foo:8, foorepeated:[8,7],
       event:{doc:[{bar:["b","c"], keep_me:True},{bar:["d"]}]},
       user:[{friends:["b", "c"]}, {friends:["d"]}]}
      {foo:7, foorepeated:[6], user:[friends:["e"]], event:{}}
    """
    pren = prensor_test_util.create_deep_prensor()
    st = prensor_to_structured_tensor.prensor_to_structured_tensor(pren)
    self.assertAllEqual(st.field_value("foo"), [[9], [8], [7]])
    self.assertAllEqual(st.field_value(["foorepeated"]), [[9], [8, 7], [6]])
    self.assertAllEqual(
        st.field_value(["user", "friends"]),
        [[[b"a"]], [[b"b", b"c"], [b"d"]], [[b"e"]]])
    self.assertAllEqual(
        st.field_value(["event", "doc", "bar"]),
        [[[[b"a"]]], [[[b"b", b"c"], [b"d"]]], [[]]])
    self.assertAllEqual(
        st.field_value(["event", "doc", "keep_me"]),
        [[[[False]]], [[[True], []]], [[]]])

  def test_non_root_prensor(self):
    child_prensor = prensor.create_prensor_from_root_and_children(
        prensor_test_util.create_child_node([0, 0, 1, 3, 7], True), {})
    with self.assertRaisesRegexp(ValueError, "Must be a root prensor"):
      prensor_to_structured_tensor.prensor_to_structured_tensor(child_prensor)

  def test_e2e_proto(self):
    """Integration test for parsing protobufs."""
    serialized = tf.constant([
        text_format.Merge(
            """
        session_info {
          session_duration_sec: 1.0
          session_feature: "foo"
        }
        event {
          query: "Hello"
          action {
            number_of_views: 1
          }
          action {
          }
        }
        event {
          query: "world"
          action {
            number_of_views: 2
          }
          action {
            number_of_views: 3
          }
        }
        """, test_pb2.Session()).SerializeToString()
    ])
    expr = proto.create_expression_from_proto(serialized,
                                              test_pb2.Session().DESCRIPTOR)
    [p] = calculate.calculate_prensors([expr])
    print(p)
    st = prensor_to_structured_tensor.prensor_to_structured_tensor(p)
    print(st)


if __name__ == "__main__":
  tf.test.main()
