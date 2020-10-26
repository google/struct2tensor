# Lint as: python3
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

from struct2tensor import prensor
from struct2tensor import prensor_to_structured_tensor
from struct2tensor.test import prensor_test_util
import tensorflow as tf

from tensorflow.python.ops.structured import structured_tensor  # pylint: disable=g-direct-tensorflow-import


def _make_structured_tensor(shape, fields):
  return structured_tensor.StructuredTensor.from_fields(
      fields=fields, shape=shape)


# @test_util.run_all_in_graph_and_eager_modes
class PrensorToStructuredTensorTest(tf.test.TestCase):

  def test_simple_prensor(self):
    pren = prensor_test_util.create_simple_prensor()
    st = prensor_to_structured_tensor.prensor_to_structured_tensor(pren)
    self.assertAllEqual(st._fields["foo"], [[9], [8], [7]])
    self.assertAllEqual(st._fields["foorepeated"], [[9], [8, 7], [6]])

  def test_nested_prensor(self):
    """Tests on a deep expression."""
    pren = prensor_test_util.create_nested_prensor()
    st = prensor_to_structured_tensor.prensor_to_structured_tensor(pren)
    self.assertAllEqual(st._fields["doc"]._fields["bar"],
                        [[[b"a"]], [[b"b", b"c"], [b"d"]], []])
    self.assertAllEqual(st._fields["doc"]._fields["keep_me"],
                        [[[False]], [[True], []], []])
    self.assertAllEqual(st._fields["user"]._fields["friends"],
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
    self.assertAllEqual(st._fields["foo"], [[9], [8], [7]])
    self.assertAllEqual(st._fields["foorepeated"], [[9], [8, 7], [6]])
    self.assertAllEqual(st._fields["doc"]._fields["keep_me"],
                        [[[False]], [[True], []], []])
    self.assertAllEqual(st._fields["user"]._fields["friends"],
                        [[[b"a"]], [[b"b", b"c"], [b"d"]], [[b"e"]]])
    self.assertAllEqual(st._fields["doc"]._fields["bar"],
                        [[[b"a"]], [[b"b", b"c"], [b"d"]], []])

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
    self.assertAllEqual(st._fields["foo"], [[9], [8], [7]])
    self.assertAllEqual(st._fields["foorepeated"], [[9], [8, 7], [6]])
    self.assertAllEqual(st._fields["user"]._fields["friends"],
                        [[[b"a"]], [[b"b", b"c"], [b"d"]], [[b"e"]]])
    self.assertAllEqual(st._fields["event"]._fields["doc"]._fields["bar"],
                        [[[[b"a"]]], [[[b"b", b"c"], [b"d"]]], [[]]])
    self.assertAllEqual(st._fields["event"]._fields["doc"]._fields["keep_me"],
                        [[[[False]]], [[[True], []]], [[]]])

  def test_non_root_prensor(self):
    child_prensor = prensor.create_prensor_from_root_and_children(
        prensor_test_util.create_child_node([0, 0, 1, 3, 7], True), {})
    with self.assertRaisesRegexp(ValueError, "Must be a root prensor"):
      prensor_to_structured_tensor.prensor_to_structured_tensor(child_prensor)


if __name__ == "__main__":
  tf.test.main()
