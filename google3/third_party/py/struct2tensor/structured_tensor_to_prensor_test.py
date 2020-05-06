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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

import numpy as np
from struct2tensor import path
from struct2tensor import structured_tensor_to_prensor
import tensorflow.compat.v2 as tf

import unittest  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.ops.ragged.row_partition import RowPartition  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.ops.structured import structured_tensor  # pylint: disable=g-direct-tensorflow-import


def _make_structured_tensor(shape, fields):
  return structured_tensor.StructuredTensor.from_fields(
      fields=fields, shape=shape)


# pylint: disable=g-long-lambda
class StructuredTensorToPrensorTest(test_util.TensorFlowTestCase,
                                    parameterized.TestCase):

  @parameterized.named_parameters([
      {
          # Structure:{},{}
          "testcase_name": "no_fields",
          "shape": [2],
          "fields": {
          },
      },
      {
          # Structure:[]
          "testcase_name": "field_of_zeroes",
          "shape": [0],
          "fields": {
              "t": np.zeros([0, 5]),
          },
          "path_to_check": "t",
          "values": [],
          "parent_indices": [],
      },
      {
          # Structure:[][unknown ragged]
          "testcase_name": "empty_2d",
          "shape": [2, 2],
          "fields": {
          },
      },
      {
          # Structure:{"r":[1,2]},{"r":[3]}
          "testcase_name": "one_normal_field",
          "shape": [2],
          "fields": {
              "r": tf.ragged.constant([[1, 2], [3]]),
          },
          "path_to_check": "r",
          "parent_indices": [0, 0, 1],
          "values": [1, 2, 3],
      },
      {
          # Structure:{"r":[1, 2]},{"r":[3, 4]}
          "testcase_name": "one_dense_field",
          "shape": [2],
          "fields": {
              "r": tf.constant([[1, 2], [3, 4]]),
          },
          "path_to_check": "r",
          "parent_indices": [0, 0, 1, 1],
          "values": [1, 2, 3, 4],
      },
      {
          # Structure:{"zzz":[{"data":[1, 2]}, {"data":[3]}]},
          #           {"zzz":[{"data":[4, 5, 6]}, {"data":[7]},
          #                   {"data":[8, 9]}]},
          "testcase_name": "field_with_two_dimensions_first_dimension",
          "shape": [2],
          "fields": {
              "zzz": tf.ragged.constant(
                  [[[1, 2], [3]], [[4, 5, 6], [7], [8, 9]]]),

          },
          "path_to_check": "zzz",
          "root_size": 2,
          "parent_indices": [0, 0, 1, 1, 1],
      },
      {
          # Structure:{"r":[{"data":[1, 2]}, {"data":[3]}]},
          #           {"r":[{"data":[4, 5, 6]}, {"data":[7]}, {"data":[8, 9]}]},
          "testcase_name": "test_field_with_two_dimensions_second_dimension",
          "shape": [2],
          "fields": {
              "r": tf.ragged.constant(
                  [[[1, 2], [3]], [[4, 5, 6], [7], [8, 9]]]),

          },
          "path_to_check": "r.data",
          "parent_indices": [0, 0, 1, 2, 2, 2, 3, 4, 4],
      },
      {
          # Structure:{"foo":[{"bar":[1, 2]}, {"bar":[3]}]},
          #           {"foo":[{"bar":[4, 5, 6]}, {"bar":[7]}, {"bar":[8, 9]}]},
          "testcase_name": "StructTensor_within_StructTensor",
          "shape": [2],
          "fields": {
              "foo":
                  _make_structured_tensor([2, None],
                                          {"bar":
                                           tf.ragged.constant(
                                               [[[1, 2], [3]],
                                                [[4, 5, 6], [7], [8, 9]]])}),
          },
          "path_to_check": "foo",
          "root_size": 2,
          "parent_indices": [0, 0, 1, 1, 1],
      },
      {
          "testcase_name": "multiple_normal_fields",
          "shape": [2],
          "fields": {
              "a": tf.ragged.constant([[1, 2], [3]]),
              "b": tf.ragged.constant([[4, 5], [6]]),
          },
          "path_to_check": "a",
          "parent_indices": [0, 0, 1],
      },
      {
          "testcase_name": "multiple_fields_with_two_dimensions",
          "shape": [2],
          "fields": {
              "a": tf.ragged.constant(
                  [[[1, 2], [3]], [[4, 5, 6], [7], [8, 9]]]),
              "b": tf.ragged.constant(
                  [[[1], []], [[2, 3], [4, 5, 6], [7, 8]]]),
          },
          "path_to_check": "b.data",
          "values": [1, 2, 3, 4, 5, 6, 7, 8],
      },
      {
          "testcase_name": "one_empty_field",
          "shape": [0],
          "fields": {
              "t": [],
          },
          "path_to_check": "t",
          "values": [],
          "parent_indices": [],
      },
      {
          "testcase_name": "shallow_field",
          "shape": [2],
          "fields": {"t": [5, 8]},
          "path_to_check": "t",
          "values": [5, 8],
          "parent_indices": [0, 1],
      },
      {
          "testcase_name": "shallow_message",
          "shape": [2],
          "fields": {"t": structured_tensor.StructuredTensor.from_pyval([{"a": [7]}, {"a": [8]}])},
          "path_to_check": "t",
          "parent_indices": [0, 1],
      },
      {
          "testcase_name": "zero_list_tensor",
          "shape": [],
          "fields": {
          },
      },
      {
          # A scalar value for a scalar structured tensor.
          "testcase_name": "one_scalar",
          "shape": [],
          "fields": {
              "t": 3
          },
          "path_to_check": "t",
          "parent_indices": [0],
          "values": [3],
      },
      {
          # A repeated value for a scalar structured tensor.
          "testcase_name": "scalar_repeated",
          "shape": [],
          "fields": {
              "t": [3, 4, 5]
          },
          "path_to_check": "t",
          "parent_indices": [0, 0, 0],
          "values": [3, 4, 5],
      },

  ])  # pyformat: disable
  def testField(self,
                shape,
                fields,
                path_to_check=None,
                parent_indices=None,
                root_size=None,
                values=None):
    struct = _make_structured_tensor(shape, fields)
    prensor = structured_tensor_to_prensor.structured_tensor_to_prensor(struct)
    if root_size is not None:
      self.assertAllEqual(root_size, prensor.node.size)
    if path_to_check is not None:
      my_path = path.create_path(path_to_check)
      descendant = prensor.get_descendant(my_path)
      self.assertIsNotNone(descendant)
      my_node = descendant.node
      if parent_indices is not None:
        self.assertAllEqual(my_node.parent_index, parent_indices)
      if values is not None:
        self.assertAllEqual(my_node.values, values)

  def testEmpty2DRagged(self):
    struct = structured_tensor.StructuredTensor.from_fields(
        fields={},
        shape=[2, None],
        row_partitions=[RowPartition.from_row_splits([0, 3, 5])])
    p = structured_tensor_to_prensor.structured_tensor_to_prensor(struct)
    child_node = p.get_child("data").node
    self.assertAllEqual(child_node.parent_index, [0, 0, 0, 1, 1])

  def testStructuredTensorCreation(self):
    rt = tf.RaggedTensor.from_value_rowids(
        tf.constant([[1, 2], [3, 4], [5, 6]]), [0, 0, 1])

    struct = _make_structured_tensor([2], {"r": rt})
    p = structured_tensor_to_prensor.structured_tensor_to_prensor(struct)
    rt_value = p.get_descendant(path.create_path("r.data"))
    self.assertAllEqual(rt_value.node.parent_index, [0, 0, 1, 1, 2, 2])
    self.assertAllEqual(rt_value.node.values, [1, 2, 3, 4, 5, 6])

  def testDeepStructuredTensor(self):
    rt = tf.RaggedTensor.from_value_rowids(
        tf.constant([[1, 2], [3, 4], [5, 6]]), [0, 0, 1])

    struct = _make_structured_tensor([2], {"r": rt})
    struct_2 = struct.partition_outer_dimension(
        RowPartition.from_row_splits([0, 1, 2]))

    p = structured_tensor_to_prensor.structured_tensor_to_prensor(struct_2)
    rt_value = p.get_descendant(path.create_path("data.r.data"))
    self.assertAllEqual(rt_value.node.parent_index, [0, 0, 1, 1, 2, 2])
    self.assertAllEqual(rt_value.node.values, [1, 2, 3, 4, 5, 6])
    p_data = p.get_descendant(path.create_path("data"))
    self.assertAllEqual(p_data.node.parent_index, [0, 1])
    p_data_r = p.get_descendant(path.create_path("data.r"))
    self.assertAllEqual(p_data_r.node.parent_index, [0, 0, 1])

  def testTwoDeepStructuredTensor(self):
    rt = tf.RaggedTensor.from_value_rowids(
        tf.constant([[1, 2], [3, 4], [5, 6]]), [0, 0, 1])

    struct = _make_structured_tensor([2], {"r": rt})
    struct_2 = struct.partition_outer_dimension(
        RowPartition.from_row_splits([0, 1, 2]))
    struct_3 = struct_2.partition_outer_dimension(
        RowPartition.from_row_splits([0, 1, 2]))
    p = structured_tensor_to_prensor.structured_tensor_to_prensor(struct_3)
    rt_value = p.get_descendant(path.create_path("data.data.r.data"))
    self.assertAllEqual(rt_value.node.parent_index, [0, 0, 1, 1, 2, 2])
    self.assertAllEqual(rt_value.node.values, [1, 2, 3, 4, 5, 6])

  def testFullyPartitionedRaggedTensor2D(self):
    values = [3, 1, 4, 1, 5, 9, 2, 6]
    row_splits = [0, 4, 4, 7, 8, 8]
    rt = tf.RaggedTensor.from_row_splits(values, row_splits=row_splits)
    rt_result = structured_tensor_to_prensor._fully_partitioned_ragged_tensor(
        rt)
    self.assertAllEqual(rt_result.values, values)
    self.assertAllEqual(rt_result.row_splits, row_splits)

  def testFullyPartitionedRaggedTensor3D(self):
    values = [3, 1, 4, 1, 5, 9, 2, 6]
    row_splits1 = [0, 2, 4]
    row_splits2 = [0, 2, 5, 7, 8]
    rt = tf.RaggedTensor.from_row_splits(
        tf.RaggedTensor.from_row_splits(values=values, row_splits=row_splits2),
        row_splits=row_splits1)
    rt_result = structured_tensor_to_prensor._fully_partitioned_ragged_tensor(
        rt)
    self.assertAllEqual(rt_result.values.row_splits, row_splits2)
    self.assertAllEqual(rt_result.row_splits, row_splits1)
    self.assertAllEqual(rt_result.values.values, values)

  def testFullyPartitionedRaggedTensor2DTensor(self):
    rt_result = structured_tensor_to_prensor._fully_partitioned_ragged_tensor(
        tf.constant([[1, 2], [3, 4]]))
    self.assertAllEqual(rt_result.values, [1, 2, 3, 4])
    self.assertAllEqual(rt_result.row_splits, [0, 2, 4])

  def testFullyPartitionedRaggedTensor3DTensor(self):
    rt_result = structured_tensor_to_prensor._fully_partitioned_ragged_tensor(
        tf.constant([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]))
    self.assertAllEqual(rt_result.values.values,
                        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    self.assertAllEqual(rt_result.row_splits, [0, 2, 4])
    self.assertAllEqual(rt_result.values.row_splits, [0, 3, 6, 9, 12])

  def testFullyPartitionedRaggedTensor3DRaggedRank1(self):
    row_splits = [0, 2, 4]
    rt_result = structured_tensor_to_prensor._fully_partitioned_ragged_tensor(
        tf.RaggedTensor.from_row_splits(
            tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]),
            row_splits=row_splits))
    self.assertAllEqual(rt_result.values.values,
                        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    self.assertAllEqual(rt_result.row_splits, [0, 2, 4])
    self.assertAllEqual(rt_result.values.row_splits, [0, 3, 6, 9, 12])

  def testFullyPartitionedRaggedTensor4DRaggedRank2(self):
    row_splits1 = [0, 2]
    row_splits2 = [0, 1, 2]
    rt_result = structured_tensor_to_prensor._fully_partitioned_ragged_tensor(
        tf.RaggedTensor.from_row_splits(
            tf.RaggedTensor.from_row_splits(
                tf.constant([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11,
                                                                  12]]]),
                row_splits=row_splits2),
            row_splits=row_splits1))
    self.assertAllEqual(rt_result.row_splits, [0, 2])
    self.assertAllEqual(rt_result.values.row_splits, [0, 1, 2])
    self.assertAllEqual(rt_result.values.values.row_splits, [0, 2, 4])
    self.assertAllEqual(rt_result.values.values.values.row_splits,
                        [0, 3, 6, 9, 12])
    self.assertAllEqual(rt_result.values.values.values.values,
                        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

  def testPartitionIfNotVector3D(self):
    t = tf.constant([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    rt_result = structured_tensor_to_prensor._partition_if_not_vector(
        t, tf.int64)
    self.assertAllEqual(rt_result.row_splits, [0, 2, 4])
    self.assertAllEqual(rt_result.values.row_splits, [0, 3, 6, 9, 12])
    self.assertAllEqual(rt_result.values.values,
                        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])


class StructuredTensorExpandDims(test_util.TensorFlowTestCase,
                                 parameterized.TestCase):

  @parameterized.named_parameters([
      {
          # Structure:{},{}
          "testcase_name": "no_fields",
          "shape": [2],
          "fields": {
          },
          "axis": 0,
      },
      {
          # Structure:[]
          "testcase_name": "field_of_zeroes",
          "shape": [0],
          "fields": {
              "t": np.zeros([0, 5]),
          },
          "axis": -1,
      },
      {
          # Structure:[][unknown ragged]
          "testcase_name": "empty_2d",
          "shape": [2, 2],
          "fields": {
          },
          "axis": 1,
      },
      {
          # Structure:{"r":[1,2]},{"r":[3]}
          "testcase_name": "one_normal_field",
          "shape": [2],
          "fields": {
              "r": tf.ragged.constant([[1, 2], [3]]),
          },
          "axis": 1,
          "target_field": "r",
          "expected": [[[1, 2]], [[3]]],
      },
      {
          # Structure:{"r":[1, 2]},{"r":[3, 4]}
          "testcase_name": "one_dense_field_axis_0",
          "shape": [2],
          "fields": {
              "r": tf.constant([[1, 2], [3, 4]]),
          },
          "axis": 0,
          "target_field": "r",
          "expected": [[[1, 2], [3, 4]]],
      },
      {
          # Structure:{"r":[1, 2]},{"r":[3, 4]}
          "testcase_name": "one_dense_field_axis_1",
          "shape": [2],
          "fields": {
              "r": tf.constant([[1, 2], [3, 4]]),
          },
          "axis": 1,
          "target_field": "r",
          "expected": [[[1, 2]], [[3, 4]]]
      },
      {
          # Structure:{"zzz":[{"data":[1, 2]}, {"data":[3]}]},
          #           {"zzz":[{"data":[4, 5, 6]}, {"data":[7]},
          #                   {"data":[8, 9]}]},
          "testcase_name": "field_with_two_dimensions_axis_0",
          "shape": [2],
          "fields": {
              "zzz": tf.ragged.constant(
                  [[[1, 2], [3]], [[4, 5, 6], [7], [8, 9]]]),

          },
          "axis": 0,
          "target_field": "zzz",
          "expected": [[[[1, 2], [3]], [[4, 5, 6], [7], [8, 9]]]],

      },
      {
          # Structure:{"r":[{"data":[1, 2]}, {"data":[3]}]},
          #           {"r":[{"data":[4, 5, 6]}, {"data":[7]}, {"data":[8, 9]}]},
          "testcase_name": "test_field_with_two_dimensions_axis_1",
          "shape": [2],
          "fields": {
              "r": tf.ragged.constant(
                  [[[1, 2], [3]], [[4, 5, 6], [7], [8, 9]]]),

          },
          "axis": 1,
          "target_field": "r",
          "expected": [
              [[[1, 2], [3]]],
              [[[4, 5, 6], [7], [8, 9]]]
          ]
      },
      {
          # Structure:{"foo":[{"bar":[1, 2]}, {"bar":[3]}]},
          #           {"foo":[{"bar":[4, 5, 6]}, {"bar":[7]}, {"bar":[8, 9]}]},
          "testcase_name": "StructTensor_within_StructTensor",
          "shape": [2],
          "fields": {
              "foo":
                  _make_structured_tensor([2, None],
                                          {"bar":
                                           tf.ragged.constant(
                                               [[[1, 2], [3]],
                                                [[4, 5, 6], [7], [8, 9]]])}),
          },
          "axis": 0,
      },
      {
          "testcase_name": "multiple_normal_fields",
          "shape": [2],
          "fields": {
              "a": tf.ragged.constant([[1, 2], [3]]),
              "b": tf.ragged.constant([[4, 5], [6]]),
          },
          "axis": 1,
          "target_field": "a",
          "expected": [[[1, 2]], [[3]]],
      },
      {
          "testcase_name": "multiple_fields_with_two_dimensions",
          "shape": [2],
          "fields": {
              "a": tf.ragged.constant(
                  [[[1, 2], [3]], [[4, 5, 6], [7], [8, 9]]]),
              "b": tf.ragged.constant(
                  [[[1], []], [[2, 3], [4, 5, 6], [7, 8]]]),
          },
          "axis": 1,
          "target_field": "a",
          "expected": [[[[1, 2], [3]]], [[[4, 5, 6], [7], [8, 9]]]],
      },
      {
          "testcase_name": "one_empty_field",
          "shape": [0],
          "fields": {
              "t": [],
          },
          "axis": 0,
      },
      {
          "testcase_name": "shallow_field",
          "shape": [2],
          "fields": {"t": [5, 8]},
          "axis": 0,
          "target_field": "t",
          "expected": [[5, 8]],
      },
      {
          "testcase_name": "shallow_message",
          "shape": [2],
          "fields": {"t": structured_tensor.StructuredTensor.from_pyval([{"a": [7]}, {"a": [8]}])},
          "axis": 0,
      },
      {
          "testcase_name": "zero_list_tensor",
          "shape": [],
          "fields": {
          },
          "axis": 0,
      },
      {
          # A scalar value for a scalar structured tensor.
          "testcase_name": "one_scalar",
          "shape": [],
          "fields": {
              "t": 3
          },
          "axis": 0,
          "target_field": "t",
          "expected": [3],
      },
      {
          # A repeated value for a scalar structured tensor.
          "testcase_name": "scalar_repeated",
          "shape": [],
          "fields": {
              "t": [3, 4, 5]
          },
          "axis": 0,
          "target_field": "t",
          "expected": [[3, 4, 5]],
      },
  ])  # pyformat: disable
  def testField(self, shape, fields, axis=0, target_field=None, expected=None):
    struct = _make_structured_tensor(shape, fields)
    actual = structured_tensor_to_prensor._expand_dims(struct, axis=axis)
    if expected is not None:
      self.assertAllEqual(actual.field_value(target_field), expected)

  @parameterized.named_parameters([
      {
          # Structure:{},{}
          "testcase_name": "large_axis_small_rank",
          "shape": [2],
          "fields": {
          },
          "axis": 2,
          "reg_error": "Axis larger than rank: 2 > 1",
      },
      {
          # Structure:[]
          "testcase_name": "big_negative",
          "shape": [0],
          "fields": {
              "t": np.zeros([0, 5]),
          },
          "axis": -3,
          "reg_error": "Axis out of range: -3",
      },
      {
          # Currently, axis >= 2 is disallowed.
          "testcase_name": "temporary_test_limiting_performance",
          "shape": [3, 3, 3, 3],
          "fields": {
          },
          "axis": 2,
          "reg_error": "Unimplemented: .* > 1 for _expand_dims",
      },
  ])  # pyformat: disable
  def testRaises(self, shape, fields, axis=0, reg_error=None):
    struct = _make_structured_tensor(shape, fields)
    with self.assertRaisesRegexp(ValueError, reg_error):
      structured_tensor_to_prensor._expand_dims(struct, axis=axis)


if __name__ == "__main__":
  unittest.main()
