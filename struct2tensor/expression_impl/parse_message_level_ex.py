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
r"""Parses regular fields, extensions, any casts, and map protos.

This is intended for use within proto.py, not independently.

parse_message_level(...) in struct2tensor_ops provides a direct interface to
parsing a protocol buffer message. In particular, extensions and regular fields
can be directly extracted from the protobuf. However, prensors provide other
syntactic sugar to parse protobufs, and parse_message_level_ex(...) handles
these in addition to regular fields and extensions.



Specifically, consider google.protobuf.Any and proto maps:

```
package foo.bar;

message MyMessage {
  Any my_any = 1;
  map<string, Baz> my_map = 2;
}
message Baz {
  int32 my_int = 1;
  ...
}
```

Then for MyMessage, the path my_any.(foo.bar.Baz).my_int is an optional path.
Also, my_map[x].my_int is an optional path.

```
  MyMessage--------------
     \  my_any?          \ my_map[x]
      *                   *
       \  (foo.bar.Baz)?   \  my_int?
        *                   *
         \  my_int?
          *
```

Thus, we can run:

```
my_message_serialized_tensor = ...

my_message_parsed = parse_message_level_ex(
    my_message_serialized_tensor,
    MyMessage.DESCRIPTOR,
    {"my_any", "my_map[x]"})

my_any_serialized = my_message_parsed["my_any"].value

my_any_parsed = parse_message_level_ex(
    my_any_serialized,
    Any.DESCRIPTOR,
    {"(foo.bar.Baz)"})
```

At this point, my_message_parsed["my_map[x]"].value AND
my_any_parsed["(foo.bar.Baz)"].value are serialized Baz tensors.
"""
# pylint: disable=protected-access

import collections
from typing import List, Mapping, Optional, Sequence, Set

from struct2tensor import path
from struct2tensor.ops import struct2tensor_ops
import tensorflow as tf

from google.protobuf import descriptor

# To the best of my knowledge, ProtoFieldNames ARE strings.
ProtoFieldName = str
ProtoFullName = str

# A string representing a step in a path.
StrStep = str


def parse_message_level_ex(
    tensor_of_protos: tf.Tensor,
    desc: descriptor.Descriptor,
    field_names: Set[ProtoFieldName],
    message_format: str = "binary",
    backing_str_tensor: Optional[tf.Tensor] = None,
    honor_proto3_optional_semantics: bool = False
) -> Mapping[StrStep, struct2tensor_ops._ParsedField]:
  """Parses regular fields, extensions, any casts, and map protos."""
  raw_field_names = _get_field_names_to_parse(desc, field_names)
  regular_fields = list(
      struct2tensor_ops.parse_message_level(
          tensor_of_protos,
          desc,
          raw_field_names,
          message_format=message_format,
          backing_str_tensor=backing_str_tensor,
          honor_proto3_optional_semantics=honor_proto3_optional_semantics))
  regular_field_map = {x.field_name: x for x in regular_fields}

  any_fields = _get_any_parsed_fields(desc, regular_field_map, field_names)
  map_fields = _get_map_parsed_fields(desc, regular_field_map, field_names,
                                      backing_str_tensor)
  result = regular_field_map
  result.update(any_fields)
  result.update(map_fields)
  return result


def is_any_descriptor(desc: descriptor.Descriptor) -> bool:
  """Returns true if it is an Any descriptor."""
  return desc.full_name == "google.protobuf.Any"


def get_full_name_from_any_step(
    step: ProtoFieldName) -> Optional[ProtoFieldName]:
  """Gets the full name of a protobuf from a google.protobuf.Any step.

  An any step is of the form (foo.com/bar.Baz). In this case the result would
  be bar.Baz.

  Args:
    step: the string of a step in a path.

  Returns:
    the full name of a protobuf if the step is an any step, or None otherwise.
  """
  if not step:
    return None
  if step[0] != "(":
    return None
  if step[-1] != ")":
    return None
  step_without_parens = step[1:-1]
  return step_without_parens.split("/")[-1]


def _any_indices_with_type(type_url: struct2tensor_ops._ParsedField,
                           full_name: ProtoFullName) -> tf.Tensor:
  """Returns the parent indices that have a type_url of full_name."""
  tensors_parsed = tf.compat.v1.string_split(type_url.value, delimiter="/")
  second_column_shape = tf.stack([
      tf.shape(type_url.value, out_type=tf.int64)[0],
      tf.constant(1, dtype=tf.int64)
  ],
                                 axis=0)
  second_column = tf.reshape(
      tf.sparse.to_dense(
          tf.sparse.slice(tensors_parsed, tf.constant([0, 1], dtype=tf.int64),
                          second_column_shape),
          default_value=""), [-1])
  equal_to_full_name = tf.equal(second_column, full_name)
  return tf.boolean_mask(type_url.index, equal_to_full_name)


def _get_any_parsed_field(value_field: struct2tensor_ops._ParsedField,
                          type_url_field: struct2tensor_ops._ParsedField,
                          field_name: StrStep
                         ) -> struct2tensor_ops._ParsedField:
  """Helper function for _get_any_parsed_fields."""
  full_name = get_full_name_from_any_step(field_name)
  indices_with_type = _any_indices_with_type(type_url_field, full_name)
  [index_to_solution_index, index_to_values
  ] = struct2tensor_ops.equi_join_indices(indices_with_type, value_field.index)
  solution_index = tf.gather(indices_with_type, index_to_solution_index)
  solution_value = tf.gather(value_field.value, index_to_values)
  # TODO(martinz): make _ParsedField public.
  return struct2tensor_ops._ParsedField(  # pylint: disable=protected-access
      field_name=field_name,
      field_descriptor=None,
      index=solution_index,
      value=solution_value)


def _get_any_parsed_fields(
    desc: descriptor.Descriptor,
    raw_parsed_fields: Mapping[StrStep, struct2tensor_ops._ParsedField],
    field_names: Set[StrStep]
) -> Mapping[StrStep, struct2tensor_ops._ParsedField]:
  """Gets the _ParsedField sequence for an Any protobuf."""
  if not is_any_descriptor(desc):
    return {}

  result = []  # type: List[struct2tensor_ops._ParsedField]

  for x in field_names:
    if path.is_extension(x):
      result.append(
          _get_any_parsed_field(raw_parsed_fields["value"],
                                raw_parsed_fields["type_url"], x))
  return {x.field_name: x for x in result}


def _get_field_names_to_parse(
    desc: descriptor.Descriptor,
    needed_field_names: Set[StrStep]) -> Sequence[ProtoFieldName]:
  """Gets the field names to parse from the original protobuf."""
  result = set()  # Set[ProtoFieldName]
  for x in needed_field_names:
    if path.is_map_indexing_step(x):
      map_field_name, _ = path.parse_map_indexing_step(x)
      result.add(map_field_name)
    elif path.is_extension(x) and is_any_descriptor(desc):
      result.add("type_url")
      result.add("value")
    else:
      result.add(x)
  return list(result)


def _get_map_parsed_fields(
    desc: descriptor.Descriptor,
    regular_fields: Mapping[StrStep, struct2tensor_ops._ParsedField],
    field_names: Set[StrStep],
    backing_str_tensor: Optional[tf.Tensor] = None
) -> Mapping[StrStep, struct2tensor_ops._ParsedField]:
  """Gets the map proto ParsedFields.

  field_names includes all the fields: map fields, any fields, and
  regular fields.

  Args:
    desc: the descriptor of the parent proto.
    regular_fields: the fields that are parsed directly from the proto.
    field_names: all fields needed: map fields, any fields, and regular fields.
    backing_str_tensor: a string tensor representing the root serialized proto.
      This is passed to keep string_views of the tensor valid for all children
      of the root expression

  Returns:
    A map from field names to ParsedFields, only for the field names of the form
    foo[bar].
  """
  maps_to_parse = collections.defaultdict(dict)
  for x in field_names:
    if path.is_map_indexing_step(x):
      map_field_name, key = path.parse_map_indexing_step(x)
      maps_to_parse[map_field_name][key] = x
  result_as_list = []
  for map_field_name, v in maps_to_parse.items():
    parsed_map_field = regular_fields[map_field_name]
    keys_needed = list(v.keys())
    map_field_value = parsed_map_field.value
    map_field_index = parsed_map_field.index
    map_field_desc = desc.fields_by_name[map_field_name].message_type
    values_and_parent_indices = struct2tensor_ops.parse_proto_map(
        map_field_value, map_field_index, map_field_desc, keys_needed,
        backing_str_tensor)
    for map_key, [value, parent_index] in zip(keys_needed,
                                              values_and_parent_indices):
      result_as_list.append(
          struct2tensor_ops._ParsedField(
              field_name=v[map_key],
              field_descriptor=None,
              index=parent_index,
              value=value))
  return {x.field_name: x for x in result_as_list}


def _get_parsed_field(data: Sequence[struct2tensor_ops._ParsedField],
                      field_name: ProtoFieldName
                     ) -> Optional[struct2tensor_ops._ParsedField]:
  for x in data:
    if x.field_name == field_name:
      return x
  return None
