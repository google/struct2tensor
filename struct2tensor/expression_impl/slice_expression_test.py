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
"""Tests for slice_expression."""

from absl.testing import absltest
from struct2tensor import calculate
from struct2tensor import create_expression
from struct2tensor import path
# For tf.Session.Run against a Prensor
from struct2tensor import prensor_value  # pylint: disable=unused-import
from struct2tensor.expression_impl import slice_expression
from struct2tensor.test import prensor_test_util
import tensorflow as tf

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


@test_util.run_all_in_graph_and_eager_modes
class SliceExpressionTest(tf.test.TestCase):

  def test_slice_end(self):
    root = create_expression.create_expression_from_prensor(
        prensor_test_util.create_big_prensor())
    root_2 = slice_expression.slice_expression(root, path.Path(["doc"]),
                                               "new_doc", None, 1)
    result = calculate.calculate_prensors([root_2])[0]

    self.assertAllEqual(
        result.get_descendant_or_error(path.Path(["new_doc"
                                                 ])).node.parent_index, [0, 1])
    self.assertAllEqual(
        result.get_descendant_or_error(path.Path(["new_doc", "keep_me"
                                                 ])).node.parent_index, [0, 1])
    self.assertAllEqual(
        result.get_descendant_or_error(path.Path(["new_doc",
                                                  "keep_me"])).node.values,
        [False, True])
    self.assertAllEqual(
        result.get_descendant_or_error(path.Path(["new_doc",
                                                  "bar"])).node.parent_index,
        [0, 1, 1])
    self.assertAllEqual(
        result.get_descendant_or_error(path.Path(["new_doc",
                                                  "bar"])).node.values,
        [b"a", b"b", b"c"])

  def test_slice_begin(self):
    """Test slice with only begin specified.

    Starts with:
    {
      foo:9,
      foorepeated:[9],
      doc:[{
         bar:["a"],
         keep_me:False
        }],
      user:[
        {
          friends:["a"]
        }]
    }
    {foo:8,
     foorepeated:[8,7],
     doc:[{
       bar:["b","c"],
       keep_me:True
     },{
       bar:["d"]
     }],
     user:[{
       friends:["b", "c"]
     },{
       friends:["d"]
     }],
     }
     {foo:7,
      foorepeated:[6],
      user:[{friends:["e"]}]}

    Creates new_doc by slicing doc[1:]:
    {foo:9,
     foorepeated:[9],
     doc:[{
       bar:["a"],
       keep_me:False
     }],
     user:[{
       friends:["a"]
     }]}
    {foo:8,
     foorepeated:[8,7],
     doc:[{
       bar:["b","c"],
       keep_me:True
     },{
       bar:["d"]
     }],
     new_doc[{
       bar:["d"]
     }],
     user:[{
       friends:["b", "c"]
     },{
       friends:["d"]}],}
    {foo:7,
     foorepeated:[6],
     user:[{
       friends:["e"]
     }]}
    """
    root = create_expression.create_expression_from_prensor(
        prensor_test_util.create_big_prensor())
    root_2 = slice_expression.slice_expression(root, path.Path(["doc"]),
                                               "new_doc", 1, None)
    result = calculate.calculate_prensors([root_2])[0]
    self.assertAllEqual(
        result.get_descendant_or_error(path.Path(["new_doc"
                                                 ])).node.parent_index, [1])
    self.assertAllEqual(
        result.get_descendant_or_error(path.Path(["new_doc", "keep_me"
                                                 ])).node.parent_index, [])
    self.assertAllEqual(
        result.get_descendant_or_error(path.Path(["new_doc",
                                                  "keep_me"])).node.values, [])
    self.assertAllEqual(
        result.get_descendant_or_error(path.Path(["new_doc",
                                                  "bar"])).node.parent_index,
        [0])
    self.assertAllEqual(
        result.get_descendant_or_error(path.Path(["new_doc",
                                                  "bar"])).node.values, [b"d"])

  def test_slice_mask(self):
    root = create_expression.create_expression_from_prensor(
        prensor_test_util.create_big_prensor())
    root_2, new_path = slice_expression._get_slice_mask(root,
                                                        path.Path(["doc"]),
                                                        None, 1)
    result = calculate.calculate_prensors([root_2])[0]
    self.assertAllEqual(
        result.get_descendant_or_error(new_path).node.parent_index, [0, 1, 1])
    self.assertAllEqual(
        result.get_descendant_or_error(new_path).node.values,
        [True, True, False])

  def test_slice_mask_end_negative(self):
    root = create_expression.create_expression_from_prensor(
        prensor_test_util.create_big_prensor())
    root_2, new_path = slice_expression._get_slice_mask(root,
                                                        path.Path(["doc"]),
                                                        None, -1)
    result = calculate.calculate_prensors([root_2])[0]
    self.assertAllEqual(
        result.get_descendant_or_error(new_path).node.parent_index, [0, 1, 1])
    self.assertAllEqual(
        result.get_descendant_or_error(new_path).node.values,
        [False, True, False])

  def test_slice_mask_begin_positive(self):
    root = create_expression.create_expression_from_prensor(
        prensor_test_util.create_big_prensor())
    root_2, new_path = slice_expression._get_slice_mask(root,
                                                        path.Path(["doc"]), 1,
                                                        None)
    [result] = calculate.calculate_prensors([root_2])
    self.assertAllEqual(
        result.get_descendant_or_error(new_path).node.parent_index, [0, 1, 1])
    self.assertAllEqual(
        result.get_descendant_or_error(new_path).node.values,
        [False, False, True])

  def test_slice_mask_begin_negative(self):
    root = create_expression.create_expression_from_prensor(
        prensor_test_util.create_big_prensor())
    root_2, new_path = slice_expression._get_slice_mask(root,
                                                        path.Path(["doc"]), -1,
                                                        None)
    result = calculate.calculate_prensors([root_2])[0]
    self.assertAllEqual(
        result.get_descendant_or_error(new_path).node.parent_index, [0, 1, 1])
    self.assertAllEqual(
        result.get_descendant_or_error(new_path).node.values,
        [True, False, True])


if __name__ == "__main__":
  absltest.main()
