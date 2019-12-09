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
"""Utilities for manipulating prensors."""

# pylint: disable=g-ambiguous-str-annotation

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import collections
from struct2tensor import path
from struct2tensor.ops import file_descriptor_set
from struct2tensor.ops import gen_decode_proto_map_op
from struct2tensor.ops import gen_decode_proto_sparse
from struct2tensor.ops import gen_equi_join_indices
from struct2tensor.ops import gen_run_length_before

import tensorflow as tf

from typing import Sequence, Text, Tuple

from google.protobuf import descriptor


def _get_dtype_from_cpp_type(cpp_type):
  """Converts a cpp type in FieldDescriptor to the appropriate dtype."""
  library = {
      descriptor.FieldDescriptor.CPPTYPE_INT32: tf.int32,
      descriptor.FieldDescriptor.CPPTYPE_INT64: tf.int64,
      descriptor.FieldDescriptor.CPPTYPE_UINT32: tf.uint32,
      descriptor.FieldDescriptor.CPPTYPE_UINT64: tf.uint64,
      descriptor.FieldDescriptor.CPPTYPE_DOUBLE: tf.float64,
      descriptor.FieldDescriptor.CPPTYPE_FLOAT: tf.float32,
      descriptor.FieldDescriptor.CPPTYPE_BOOL: tf.bool,
      descriptor.FieldDescriptor.CPPTYPE_ENUM: tf.int32,
      descriptor.FieldDescriptor.CPPTYPE_STRING: tf.string,
      descriptor.FieldDescriptor.CPPTYPE_MESSAGE: tf.string
  }
  return library[cpp_type]


# A named tuple for parse_full_message_level and parse_message_level
# value and index are tensors.
# TODO(martinz): make this struct public.
#   field_name: str
#   field_descriptor: descriptor.FieldDescriptor (note: not used in V2).
#   value: tf.Tensor
#   index: tf.Tensor
_ParsedField = collections.namedtuple(
    "_ParsedField", ["field_name", "field_descriptor", "value", "index"])


def parse_full_message_level(
    tensor_of_protos,
    descriptor_type,
    message_format = "binary"):
  """Parses all of the fields at a level of a message.

  If there is a field with a message type, it is parsed as a string. Then, the
  function can be applied recursively.
  Note: this will not extract extensions.
  Args:
    tensor_of_protos: a 1-D tensor of strings of protocol buffers.
    descriptor_type: a descriptor for the protocol buffer to parse. See
      https://github.com/protocolbuffers/protobuf/blob/master/python/google/protobuf/descriptor.py
    message_format: Indicates the format of the protocol buffer: is one of
      'text' or 'binary'.
  Returns:
    list of named tuples, one per field_name in field_names:
    field_name: the string from field_names.
    field_descriptor: descriptor_type.fields_by_name[field_name]
    value: a 1-D tensor of the values from the field field_name.
    index: an index, such that for all i, tensor_of_protos[index[i]] has a
      value value[i]. Note that sometimes index[i]=index[i+1], implying a
      repeated field field_name.
  """

  field_names = [field.name for field in descriptor_type.fields]
  return parse_message_level(tensor_of_protos, descriptor_type, field_names,
                             message_format)


def _get_field_descriptor(descriptor_type,
                          field_name):
  if path.is_extension(field_name):
    return descriptor_type.file.pool.FindExtensionByName(
        path.get_raw_extension_name(field_name))
  else:
    return descriptor_type.fields_by_name[field_name]


def parse_message_level(
    tensor_of_protos,
    descriptor_type,
    field_names,
    message_format = "binary"):
  """Parses a subset of the fields at a level of a message.

  If there is a field with a message type, it is parsed as a string. Then, the
  function can be applied recursively.

  Args:
    tensor_of_protos: a 1-D tensor of strings of protocol buffers.
    descriptor_type: a descriptor for the protocol buffer to parse. See
      https://github.com/protocolbuffers/protobuf/blob/master/python/google/protobuf/descriptor.py
    field_names: the names of the fields to parse.
    message_format: Indicates the format of the protocol buffer: is one of
      'text' or 'binary'.
  Returns:
    list of named _ParsedField, one per field_name in field_names:
    field_name: the string from field_names.
    field_descriptor: descriptor_type.fields_by_name[field_name]
    value: a 1-D tensor of the values from the field field_name.
    index: an index, such that for all i, tensor_of_protos[index[i]] has a
      value value[i]. Note that sometimes index[i]=index[i+1], implying a
      repeated field field_name.

  """
  if not field_names:
    return []
  # We sort the field names so that the input attr to DecodeProtoSparseV2 op
  # is deterministic.
  field_names = sorted(field_names)
  message_type = descriptor_type.full_name
  descriptor_set = file_descriptor_set.get_file_descriptor_set_proto(
      descriptor_type, field_names)
  descriptor_literal = descriptor_set.SerializeToString()
  # TODO(martinz): catch KeyError and give a better error.
  field_descriptors = [
      _get_field_descriptor(descriptor_type, field_name)
      for field_name in field_names
  ]
  output_types = [
      _get_dtype_from_cpp_type(field_descriptor.cpp_type)
      for field_descriptor in field_descriptors
  ]
  values, indices = gen_decode_proto_sparse.decode_proto_sparse_v2(
      tensor_of_protos,
      descriptor_literal=descriptor_literal,
      message_type=message_type,
      num_fields=len(field_names),
      field_names=list(field_names),
      output_types=output_types,
      message_format=message_format)

  result = []
  for field_name, field_descriptor, value, index in zip(field_names,
                                                        field_descriptors,
                                                        values, indices):
    result.append(
        _ParsedField(
            field_name=field_name,
            field_descriptor=field_descriptor,
            value=value,
            index=index))

  return result


def run_length_before(a):
  r"""Returns the run length of each set of elements in a vector.


  Args:
    a: a 1D int64 tensor. This assumes that for all a_i, a_j, if i <= j, then
      a_i <= a_j.

  Returns:
    1D int64 tensor [b_0,...,b_n] where b_n := \sum_{i=0}^{n-1} I(a_i=a_n)
  """
  return gen_run_length_before.run_length_before(a)


def create_sparse_tensor_for_repeated(parent_index,
                                      values, dense_shape
                                     ):
  """Helps to get the sparse tensor for a repeated PrensorField.

  Args:
    parent_index: a 1D int64 tensor, which has a pointer to the parent message.
    values: a 1D tensor, which has the primitive values associated with a field.
    dense_shape: a 1D int64 tensor, representing the dense shape.

  Returns:
    A Sparse Tensor representing a ragged array.
  """
  run_length_before_tensor = run_length_before(parent_index)
  indices = tf.stack([parent_index, run_length_before_tensor], axis=1)
  return tf.SparseTensor(
      indices=indices, values=values, dense_shape=dense_shape)


def equi_join_indices(a, b):
  """A custom op such that a broadcast operation can be done.

  Args:
    a: a tensor that is an int64 vector, where for all i, a[i] <= a[i+1]
    b: a tensor that is an int64 vector, where for all i, b[i] <= b[i+1]

  Returns:
    [index_a, index_b] where:
    1. For every k, a[index_a[k]] = b[index_b[k]]
    2. for every i,j, iff a[i]==b[j], then there exists a k where
       index_a[k]=i and index_b[k]=j.
    3. Moreover, for any k, k' where k < k',
       index_a[k] <= index_a[k'], and if index_a[k] == index_a[k'], then
       index_b[k] <= index_b[k'].
  """
  return gen_equi_join_indices.equi_join_indices(a, b)


def parse_proto_map(map_entries, map_entry_parent_indices,
                    map_entry_descriptor,
                    keys_needed
                   ):
  """A custom op to parse serialized Protobuf map entries.

  Args:
    map_entries: a 1D string tensor that contains serialized map entry
      sub-messages.
    map_entry_parent_indices: a 1D int64 tensor of the same length as
      map_entries. map_entry_parent_indices[i] == j means map_entries[i] belongs
      to the j-th map.
    map_entry_descriptor: the proto descriptor of the map entry sub-message.
    keys_needed: keys that are needed to be looked up in the map. If the map's
      keys are integers, then these strings will be parsed as integers in
      decimal. If the map's keys are booleans, then only "0" and "1" are
      expected.

  Returns:
    A list of tuples one for each key in `keys_needed`. In each tuple, the first
    term contains decoded values; the second term contains the parent indices
    for the values.
  """
  keys_needed_as_list = list(keys_needed)
  value_fd = map_entry_descriptor.fields_by_name["value"]
  values, parent_indices = gen_decode_proto_map_op.decode_proto_map(
      map_entries, map_entry_parent_indices, map_entry_descriptor.full_name,
      keys_needed_as_list, len(keys_needed_as_list),
      _get_dtype_from_cpp_type(value_fd.cpp_type),
      file_descriptor_set.get_file_descriptor_set_proto(
          map_entry_descriptor, ["key", "value"]).SerializeToString())
  return list(zip(values, parent_indices))
