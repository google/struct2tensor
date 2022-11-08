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

# CPU kernels for struct2tensors.

load("//struct2tensor:struct2tensor.bzl", "s2t_dynamic_library")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

s2t_dynamic_library(
    name = "run_length_before_op_dynamic",
    srcs = [
        "run_length_before_op.cc",
    ],
)

cc_library(
    name = "run_length_before_kernel",
    srcs = [
        "run_length_before_op.cc",
    ],
    deps = [
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:framework_lite",
    ],
    alwayslink = 1,
)

cc_library(
    name = "decode_proto_sparse_kernel",
    srcs = [
        "decode_proto_sparse_op.cc",
    ],
    deps = [
        ":vector_to_tensor",
        "@com_google_absl//absl/memory",
        "@com_google_protobuf//:protobuf",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:framework_lite",
        "@org_tensorflow//tensorflow/core:lib",
    ],
    alwayslink = 1,
)

s2t_dynamic_library(
    name = "decode_proto_sparse_op_dynamic",
    srcs = [
        "//struct2tensor/kernels:decode_proto_sparse_op.cc",
        "//struct2tensor/kernels:vector_to_tensor.h",
    ],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:endian",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/container:inlined_vector_internal",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "decode_proto_map_kernel",
    srcs = ["decode_proto_map_op.cc"],
    deps = [
        ":streaming_proto_reader",
        ":vector_to_tensor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_protobuf//:protobuf",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
    ],
    alwayslink = 1,
)

s2t_dynamic_library(
    name = "decode_proto_map_op_dynamic",
    srcs = [
        "//struct2tensor/kernels:decode_proto_map_op.cc",
        "//struct2tensor/kernels:streaming_proto_reader.cc",
        "//struct2tensor/kernels:streaming_proto_reader.h",
        "//struct2tensor/kernels:vector_to_tensor.h",
    ],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:endian",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/container:inlined_vector_internal",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "vector_to_tensor",
    hdrs = ["vector_to_tensor.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
    ],
    alwayslink = 1,
)

cc_library(
    name = "equi_join_indices_kernel",
    srcs = ["equi_join_indices_op.cc"],
    deps = [
        "@org_tensorflow//tensorflow/core:framework",
    ],
    alwayslink = 1,
)

s2t_dynamic_library(
    name = "equi_join_indices_op_dynamic",
    srcs = ["equi_join_indices_op.cc"],
)

cc_library(
    name = "equi_join_any_indices_kernel",
    srcs = ["equi_join_any_indices_op.cc"],
    deps = [
        "@org_tensorflow//tensorflow/core:framework",
    ],
    alwayslink = 1,
)

s2t_dynamic_library(
    name = "equi_join_any_indices_op_dynamic",
    srcs = ["equi_join_any_indices_op.cc"],
)

# Prensor ops are TF canonical ops.
cc_library(
    name = "struct2tensor_kernels",
    deps = [
        ":decode_proto_map_kernel",
        ":decode_proto_sparse_kernel",
        ":equi_join_any_indices_kernel",
        ":equi_join_indices_kernel",
        ":run_length_before_kernel",
    ],
    alwayslink = 1,
)

cc_library(
    name = "streaming_proto_reader",
    srcs = ["streaming_proto_reader.cc"],
    hdrs = ["streaming_proto_reader.h"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:endian",
        "@com_google_absl//absl/strings",
        "@com_google_protobuf//:protobuf",
        "@org_tensorflow//tensorflow/core:framework_lite",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)
