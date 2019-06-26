/* Copyright 2019 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using ::tensorflow::Status;
using ::tensorflow::shape_inference::InferenceContext;

REGISTER_OP("DecodeProtoMap")
    .Input("serialized_map_entries: string")
    .Input("map_entries_parent_indices: int64")
    .Attr("message_type: string")
    .Attr("keys: list(string) >= 0")
    .Attr("num_keys: int")
    .Attr("output_type: type")
    .Attr("descriptor_literal: string")
    .Output("values: num_keys * output_type")
    .Output("indices: num_keys * int64")
    .SetShapeFn([](InferenceContext* c) {
      int num_keys;
      TF_RETURN_IF_ERROR(c->GetAttr("num_keys", &num_keys));
      for (int i = 0; i < 2 * num_keys; ++i) {
        c->set_output(i, c->Vector(c->UnknownDim()));
      }
      return Status::OK();
    })
    .Doc(R"doc(
An op to decode serialized protobuf map entries with given keys into Tensors.

`serialized_map_entries`: on wire, a protobuf map is encoded into repeated
map entries where each entry is a submessage that cotnains a "key" and a "value"
field. This input Tensor should be a vector containing all such submessages from
the maps to be decoded in serialized form.

`map_entries_parent_indices`: this op supports decoding multiple logical maps.
this Tensor should have the same shape as `serialized_map_entries`.
map_entries_parent_indices[i] == j means serialized_map_entries[i] came from
the j-th logical map.

`message_type`: fully qualified name of the map entry submessage. (e.g.
some.package.SomeMapMapEntry).

`keys`: keys to look up from the map. If the map's keys are integers, then
these string attributes are parsed as integers in decimal. If the map's
keys are booleans, then only "0" and "1" are expected.

`num_keys`: Number of `keys`.

`output_type`: the DataType of the output value tensor. Note that for each
map value type, there is only one corresponding DataType. The op will enforce
it in the runtime.

`descriptor_literal`: a Serialized proto2.FileDescriptorSet proto that contains
the FileDescriptor of the map entry proto.

`values`: there are `num_keys` Tensors corresponds to this output port. Each
contains the decoded values for a key specified in `keys`.

`indices`: there are `num_keys` Tensors corresponds to this output port.
indices[i][j] == k means values[i][j] was decoded from the k-th logical map (
see `map_entries_parent_indices`)

The OP might raise DataLoss if any of the serialized map entries is corrupted.
It might also raise InvalidArgumentError if the attributes are not expected.
)doc");
