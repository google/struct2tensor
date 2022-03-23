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
"""Converting a structured tensor to a prensor.

This conversion handles a variety of differences in the implementation.

1. Prensors disambiguate between optional and repeated fields. All fields
   from a structured tensor are treated as repeated.
2. Structured tensors can represent a scalar structured object. This is not
   allowed in a prensor, so it is converted to a vector of length one.
3. Structured tensors can have dimensions that are unnamed. For instance,
   [{"foo":[[3],[4,5]]}] is a valid structured tensor. This is translated
   into [{"foo":[{"data":[3]},{"data":[4,5]}]}], where "data" is the default
   default_field_name, which can be changed by the user.
4. Structured tensors have a lot of flexibility in how fields are
   represented.
   4A. First of all, they can be dense tensors. These are converted
       to ragged tensors with the same shape. If the rank is unknown, then
       this will fail.
   4B. Also, ragged tensors may have a multidimensional tensor as the final
       value. If the rank is unknown, then this will fail.
5. Prensors encode all parent-child relationships with a parent_index tensor
   (equivalent to a value_rowid tensor), and at the head with a size tensor
   (equivalent to an nrows tensor).
6. Structured tensors can have "required" fields. Specifically, a structured
   tensor could have:
   StructuredTensor(shape=[2], fields={t:[3,7]}). This is transformed into a
   repeated field.
"""

from typing import Mapping, Union

from struct2tensor import path
from struct2tensor import prensor
import tensorflow.compat.v2 as tf

from tensorflow.python.ops.ragged.row_partition import RowPartition  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.ops.structured import structured_tensor  # pylint: disable=g-direct-tensorflow-import


def structured_tensor_to_prensor(
    st: structured_tensor.StructuredTensor,
    default_field_name: path.Step = "data") -> prensor.Prensor:
  """Converts a structured tensor to a prensor.

  Certain rank information must be known. For more details about the
  transformation, see the notes above.

  Args:
    st: the structured tensor to convert.
    default_field_name: the name to use when there is an unnamed dimension.

  Returns:
    a logically equivalent Prensor.

  Raises:
    ValueError: if there is an issue with the structured tensor.
  """
  row_partitions = st.row_partitions
  if len(row_partitions) >= 1:
    child_prensor = _structured_tensor_to_child_prensor(st, default_field_name)
    return prensor.create_prensor_from_root_and_children(
        prensor.RootNodeTensor((st).nrows()),
        {default_field_name: child_prensor})
  elif st.rank == 1:
    return prensor.create_prensor_from_root_and_children(
        prensor.RootNodeTensor((st).nrows()),
        _structured_tensor_prensor_map(st, default_field_name))
  else:
    # st is a scalar StructuredTensor.
    return structured_tensor_to_prensor(_expand_dims(st, 0), default_field_name)


def _structured_tensor_prensor_map(
    st: structured_tensor.StructuredTensor,
    default_field_name: path.Step) -> Mapping[path.Step, prensor.Prensor]:
  """Creates a map of fields, to put in a child or root prensor."""
  return {
      k: _structured_tensor_field_to_prensor(
          st.field_value(k), default_field_name) for k in st.field_names()
  }


# _expand_dims requires special treatment for scalar StructuredTensors, because
# it is not adding a partition dimension. Therefore, we have to expand the
# dimension of each field explicitly.
def _expand_dims_scalar(st: structured_tensor.StructuredTensor):
  """_expand_dims for a scalar structured tensor."""
  new_shape = tf.constant([1], dtype=tf.int64)
  new_fields = {k: _expand_dims(st.field_value(k), 0) for k in st.field_names()}
  return structured_tensor.StructuredTensor.from_fields(
      new_fields, shape=new_shape)


def _expand_dims_nonnegative_axis(axis, rank):
  """Get the nonnegative axis according to the rules of tf.expand_dims."""
  # Implementation note: equivalent to get_positive_axis(axis, rank + 1)
  if axis < 0:
    new_axis = (1 + rank) + axis
    if new_axis < 0:
      # Note: this is unreachable in the current code.
      raise ValueError("Axis out of range: " + str(axis))
    return new_axis
  elif axis > rank:
    # Note: this is unreachable in the current code.
    raise ValueError("Axis larger than rank: " + str(axis) + " > " + str(rank))
  return axis


def _expand_dims(st, axis):
  """tf.expand_dims, but works on StructuredTensor too.

  Note: the implementation does not work if axis > 1, and will throw a
  ValueError.

  Args:
    st: a Tensor, RaggedTensor, or StructuredTensor.
    axis: the axis to insert a dimension before.

  Returns:
    a tensor with one more dimension (see tf.expand_dims).
  Raises:
    ValueError:
      if the axis is not valid.
  """
  if not isinstance(st, structured_tensor.StructuredTensor):
    return tf.expand_dims(st, axis)
  nn_axis = _expand_dims_nonnegative_axis(axis, st.rank)
  if st.rank == 0:
    return _expand_dims_scalar(st)
  if nn_axis == 0:
    # Here, we can add a dimension 1 at the front.
    nrows = st.nrows()
    return st.partition_outer_dimension(
        RowPartition.from_uniform_row_length(nrows, nrows))
  elif nn_axis == 1:
    # Again, by partitioning the first dimension into vectors of length 1,
    # we can solve this problem.
    nrows = st.nrows()
    return st.partition_outer_dimension(
        RowPartition.from_uniform_row_length(
            tf.constant(1, dtype=nrows.dtype), nrows))
  else:
    # Note: this is unreachable in the current code.
    raise ValueError("Unimplemented: non-negative axis > 1 for _expand_dims")


def _structured_tensor_field_to_prensor(
    field_value: Union[structured_tensor.StructuredTensor, tf.RaggedTensor,
                       tf.Tensor],
    default_field_name: path.Step) -> prensor.Prensor:
  """Creates a ChildNodeTensor from a field in a structured tensor."""
  if isinstance(field_value, structured_tensor.StructuredTensor):
    return _structured_tensor_to_child_prensor(field_value, default_field_name)
  else:
    return _to_leaf_prensor(field_value, default_field_name)


def _row_partition_to_child_node_tensor(row_partition: RowPartition):
  """Creates a ChildNodeTensor from a RowPartition."""
  return prensor.ChildNodeTensor(
      tf.cast(row_partition.value_rowids(), tf.int64),
      is_repeated=True)


def _one_child_prensor(row_partition: RowPartition,
                       child_prensor: prensor.Prensor,
                       default_field_name: path.Step) -> prensor.Prensor:
  """Creates a prensor with a ChildNodeTensor at the root with one child."""
  child_node_tensor = _row_partition_to_child_node_tensor(row_partition)
  return prensor.create_prensor_from_root_and_children(
      child_node_tensor, {default_field_name: child_prensor})


def _structured_tensor_to_child_prensor(
    st: structured_tensor.StructuredTensor,
    default_field_name: path.Step) -> prensor.Prensor:
  """Creates a prensor with a ChildNodeTensor at the root."""
  row_partitions = st.row_partitions
  if len(row_partitions) == 1:
    child_st = st.merge_dims(0, 1)
    row_partition = row_partitions[0]
    return prensor.create_prensor_from_root_and_children(
        _row_partition_to_child_node_tensor(row_partition),
        _structured_tensor_prensor_map(child_st, default_field_name))
  elif len(row_partitions) > 1:
    row_partition = row_partitions[0]
    child_st = st.merge_dims(0, 1)
    return _one_child_prensor(
        row_partition,
        _structured_tensor_to_child_prensor(child_st, default_field_name),
        default_field_name)
  # This requires us to transform the scalar to a vector.
  # The fields could be scalars or vectors.
  # We need _expand_dims(...) to make this work.
  return _structured_tensor_to_child_prensor(
      _expand_dims(st, 1), default_field_name)


def _to_leaf_prensor_helper(rt: tf.RaggedTensor,
                            default_field_name: path.Step) -> prensor.Prensor:
  """Converts a fully partitioned ragged tensor to a leaf prensor.

  It is assumed that this is a fully partitioned ragged tensor. Specifically,
  the values at the end are a vector, not a 2D tensor.

  Args:
    rt: a fully partitioned ragged tensor (see
      _fully_partitioned_ragged_tensor).
    default_field_name: a path.Step for unnamed dimensions.

  Returns:
    a prensor, with a leaf as the root node.
  """
  row_partition = rt._row_partition  # pylint: disable=protected-access
  if rt.ragged_rank == 1:
    values = rt.values
    leaf = prensor.LeafNodeTensor(row_partition.value_rowids(), values, True)
    return prensor.create_prensor_from_root_and_children(leaf, {})
  else:
    return _one_child_prensor(
        row_partition, _to_leaf_prensor_helper(rt.values, default_field_name),
        default_field_name)


def _partition_if_not_vector(values: tf.Tensor, dtype: tf.dtypes.DType):
  """Creates a fully partitioned ragged tensor from a multidimensional tensor.

  If the tensor is 1D, then it is unchanged.

  Args:
    values: the tensor to be transformed
    dtype: the type of the row splits.

  Returns:
    A 1D tensor or a ragged tensor.

  Raises:
    ValueError: if the shape cannot be statically determined or is a scalar.
  """

  values_shape = values.shape
  assert values_shape is not None
  values_rank = values_shape.rank
  # values cannot have an unknown rank in a RaggedTensor field
  # in a StructuredTensor.
  assert values_rank is not None
  if values_rank == 1:
    return values
  # This cannot happen inside a ragged tensor.
  assert values_rank > 0
  return tf.RaggedTensor.from_tensor(
      values, ragged_rank=values_rank - 1, row_splits_dtype=dtype)


def _fully_partitioned_ragged_tensor(rt: Union[tf.RaggedTensor, tf.Tensor],
                                     dtype=tf.dtypes.int64):
  """Creates a fully partitioned ragged tensor from a tensor or a ragged tensor.

  If given a tensor, it must be at least two-dimensional.

  A fully partitioned ragged tensor is:
  1. A ragged tensor.
  2. The final values are a vector.
  Args:
    rt: input to coerce from RaggedTensor or Tensor. Must be at least 2D.
    dtype: requested dtype for partitions: tf.int64 or tf.int32.

  Returns:
    A ragged tensor where the flat values are a 1D tensor.
  Raises:
    ValueError: if the tensor is 0D or 1D.
  """
  if isinstance(rt, tf.RaggedTensor):
    rt = rt.with_row_splits_dtype(dtype)
    flattened_values = _partition_if_not_vector(rt.flat_values, dtype=dtype)
    return rt.with_flat_values(flattened_values)
  else:
    rt_shape = rt.shape
    assert rt_shape is not None
    rt_rank = rt_shape.rank
    assert rt_rank is not None
    if rt_rank < 2:
      # Increase the rank if it is a scalar.
      return _fully_partitioned_ragged_tensor(tf.expand_dims(rt, -1))
    return tf.RaggedTensor.from_tensor(
        rt, ragged_rank=rt_rank - 1, row_splits_dtype=dtype)


def _to_leaf_prensor(rt: Union[tf.RaggedTensor, tf.Tensor],
                     default_field_name: path.Step) -> prensor.Prensor:
  """Creates a leaf tensor from a ragged tensor or tensor."""
  return _to_leaf_prensor_helper(
      _fully_partitioned_ragged_tensor(rt), default_field_name)
