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

"""Tests for struct2tensor.prensor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
from struct2tensor import path
from struct2tensor import prensor
from struct2tensor.test import prensor_test_util
import tensorflow as tf

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


@test_util.run_all_in_graph_and_eager_modes
class PrensorTest(tf.test.TestCase):

  def _assert_prensor_equals(self, lhs, rhs):
    if isinstance(lhs.node, prensor.RootNodeTensor):
      self.assertIsInstance(rhs.node, prensor.RootNodeTensor)
      self.assertIs(lhs.node.size, rhs.node.size)
    elif isinstance(lhs.node, prensor.ChildNodeTensor):
      self.assertIsInstance(rhs.node, prensor.ChildNodeTensor)
      self.assertIs(lhs.node.parent_index, rhs.node.parent_index)
      self.assertEqual(lhs.node.is_repeated, rhs.node.is_repeated)
    else:
      self.assertIsInstance(rhs.node, prensor.LeafNodeTensor)
      self.assertIs(lhs.node.parent_index, rhs.node.parent_index)
      self.assertIs(lhs.node.values, rhs.node.values)
      self.assertEqual(lhs.node.is_repeated, rhs.node.is_repeated)

    self.assertEqual(len(lhs.get_children()), len(rhs.get_children()))
    for (l_child_step, l_child), (r_child_step, r_child) in six.moves.zip(
        six.iteritems(lhs.get_children()), six.iteritems(rhs.get_children())):
      self.assertEqual(l_child_step, r_child_step)
      self._assert_prensor_equals(l_child, r_child)

  def test_prensor_children_ordered(self):
    def _recursively_check_sorted(p):
      self.assertEqual(list(p.get_children().keys()),
                       sorted(p.get_children().keys()))
      for c in p.get_children().values():
        _recursively_check_sorted(c)

    for pren in [
        prensor_test_util.create_nested_prensor(),
        prensor_test_util.create_big_prensor(),
        prensor_test_util.create_deep_prensor()
    ]:
      _recursively_check_sorted(pren)

    p = prensor.create_prensor_from_descendant_nodes({
        path.Path([]):
            prensor_test_util.create_root_node(1),
        path.Path(["d"]):
            prensor_test_util.create_optional_leaf_node([0], [True]),
        path.Path(["c"]):
            prensor_test_util.create_optional_leaf_node([0], [True]),
        path.Path(["b"]):
            prensor_test_util.create_optional_leaf_node([0], [True]),
        path.Path(["a"]):
            prensor_test_util.create_optional_leaf_node([0], [True]),
    })
    self.assertEqual(["a", "b", "c", "d"], list(p.get_children().keys()))

  def test_prensor_is_composite_tensor(self):
    for pren in [
        prensor_test_util.create_nested_prensor(),
        prensor_test_util.create_big_prensor(),
        prensor_test_util.create_deep_prensor()
    ]:
      flattened_tensors = tf.nest.flatten(pren, expand_composites=True)
      self.assertIsInstance(flattened_tensors, list)
      for t in flattened_tensors:
        self.assertIsInstance(t, tf.Tensor)
      packed_pren = tf.nest.pack_sequence_as(
          pren, flattened_tensors, expand_composites=True)
      self._assert_prensor_equals(pren, packed_pren)

# The following are only available post TF 1.14.

if __name__ == "__main__":
  tf.test.main()
