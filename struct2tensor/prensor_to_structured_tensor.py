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
"""Converting a prensor to a structured tensor.

Structured tensors are almost completely more general than a prensor.
There are only four issues:

1. There is no disambiguation between an optional and a repeated field in a
   structured test. This is lost.
2. The set of field names is more restricted in a structured tensor. This is
   ignored for now: behavior of disallowed field names is undefined.
3. We could try to keep more shape information, especially if the size of the
   root prensor is known statically.
4. TODO:interpret a given field name to represent "anonymous" dimensions, and
   create multidimensional fields when that field name is used.

"""

from typing import Mapping, Union

from struct2tensor import path
from struct2tensor import prensor
import tensorflow as tf

from tensorflow.python.ops.ragged.row_partition import RowPartition  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.ops.structured import structured_tensor  # pylint: disable=g-direct-tensorflow-import


def prensor_to_structured_tensor(
    p: prensor.Prensor) -> structured_tensor.StructuredTensor:
  """Creates a structured tensor from a prensor.

  All information about optional and repeated fields is dropped.
  If the field names in the proto do not meet the specifications for
  StructuredTensor, the behavior is undefined.

  Args:
    p: the prensor to convert.

  Returns:
    An equivalent StructuredTensor.

  Raises:
    ValueError: if the root of the prensor is not a RootNodeTensor.
  """
  node = p.node
  if isinstance(node, prensor.RootNodeTensor):
    return _root_node_to_structured_tensor(
        _prensor_to_field_map(p.get_children(), node.size))
  raise ValueError("Must be a root prensor")


def _root_node_to_structured_tensor(
    fields: Mapping[path.Step, prensor.Prensor]
) -> structured_tensor.StructuredTensor:
  """Convert a map of prensors to a structured tensor."""
  return structured_tensor.StructuredTensor.from_fields(
      fields=fields, shape=tf.TensorShape([None]))


def _prensor_to_structured_tensor_helper(
    p: prensor.Prensor, nrows: tf.Tensor
) -> Union[tf.RaggedTensor, structured_tensor.StructuredTensor]:
  """Convert a prensor to a structured tensor with a certain number of rows."""
  node = p.node
  if isinstance(node, prensor.LeafNodeTensor):
    return _leaf_node_to_ragged_tensor(node, nrows)
  assert isinstance(node, prensor.ChildNodeTensor)
  return _child_node_to_structured_tensor(
      node, _prensor_to_field_map(p.get_children(), node.size), nrows)


def _prensor_to_field_map(
    p_fields: Mapping[path.Step, prensor.Prensor],
    nrows: tf.Tensor) -> Mapping[path.Step, structured_tensor.StructuredTensor]:
  """Convert a map of prensors to map of structured tensors."""
  result = {}
  for step, child in p_fields.items():
    try:
      result[step] = _prensor_to_structured_tensor_helper(child, nrows)
    except ValueError as err:
      raise ValueError(f"Error in field: {step}") from err
  return result


def _child_node_to_structured_tensor(
    node: prensor.ChildNodeTensor, fields: Mapping[path.Step, prensor.Prensor],
    nrows: tf.Tensor) -> structured_tensor.StructuredTensor:
  """Convert a map of prensors to map of structured tensors."""
  st = structured_tensor.StructuredTensor.from_fields(
      fields=fields, shape=tf.TensorShape([None]), nrows=node.size)
  row_partition = RowPartition.from_value_rowids(
      value_rowids=node.parent_index, nrows=nrows)
  return st.partition_outer_dimension(row_partition)


def _leaf_node_to_ragged_tensor(node: prensor.LeafNodeTensor,
                                nrows: tf.Tensor) -> tf.RaggedTensor:
  """Converts a LeafNodeTensor to a 2D ragged tensor."""
  return tf.RaggedTensor.from_value_rowids(
      values=node.values, value_rowids=node.parent_index, nrows=nrows)
