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
#
# Parquet Dataset Kernel Implementation.

load("//struct2tensor:struct2tensor.bzl", "s2t_dynamic_library")

package(default_visibility = ["//struct2tensor:__subpackages__"])

licenses(["notice"])

s2t_dynamic_library(
    name = "parquet_dataset_kernel_dynamic",
    srcs = [
        ":parent_indices_builder.cc",
        ":parent_indices_builder.h",
        ":parquet_dataset_kernel.cc",
        ":parquet_reader.cc",
        ":parquet_reader.h",
        ":parquet_reader_util.cc",
        ":parquet_reader_util.h",
        "//struct2tensor/kernels:vector_to_tensor.h",
    ],
    deps = [
        "@arrow//:arrow",
    ],
)

cc_library(
    name = "parquet_dataset_kernel",
    srcs = ["parquet_dataset_kernel.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":parquet_reader_lib",
        ":parquet_reader_util_lib",
        "@com_google_absl//absl/container:flat_hash_map",
        "//struct2tensor/kernels:vector_to_tensor",
        "@org_tensorflow//tensorflow/core:framework",
    ],
    alwayslink = 1,
)

cc_library(
    name = "parquet_reader_lib",
    srcs = ["parquet_reader.cc"],
    hdrs = ["parquet_reader.h"],
    deps = [
        ":parent_indices_builder_lib",
        ":parquet_reader_util_lib",
        "@com_google_absl//absl/strings",
        "@arrow//:arrow",
        "//struct2tensor/kernels:vector_to_tensor",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
    ],
    alwayslink = 1,
)

cc_library(
    name = "parquet_reader_util_lib",
    srcs = ["parquet_reader_util.cc"],
    hdrs = ["parquet_reader_util.h"],
    copts = [
        "-fexceptions",
    ],
    features = ["-use_header_modules"],  # The reason this is needed is because the parquet library
    # throws an exception in one of our tests. In order to handle the exception, we need the flag
    # -fexceptions.
    deps = [
        "@arrow//:arrow",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_library(
    name = "parent_indices_builder_lib",
    srcs = ["parent_indices_builder.cc"],
    hdrs = ["parent_indices_builder.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_test(
    name = "parent_indices_builder_test",
    srcs = ["parent_indices_builder_test.cc"],
    deps = [
        ":parent_indices_builder_lib",
        "//testing/base/public:gunit_main",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_test(
    name = "parquet_reader_test",
    srcs = ["parquet_reader_test.cc"],
    data = [
        "//struct2tensor/testdata/parquet_testdata",
    ],
    deps = [
        ":parquet_reader_lib",
        "//testing/base/public:gunit_main",
        "@org_tensorflow//tensorflow/core:framework",
    ],
)

cc_test(
    name = "parquet_reader_util_test",
    srcs = ["parquet_reader_util_test.cc"],
    data = [
        "//struct2tensor/testdata/parquet_testdata",
    ],
    deps = [
        ":parquet_reader_util_lib",
        "//testing/base/public:gunit_main",
    ],
)
