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

# Placeholder: load py_proto_library
# Placeholder: load py_library
# Placeholder: load py_test
# Placeholder: load py_binary
load("//tools/build_defs/proto/cpp:cc_proto_library.bzl", "cc_proto_library")

licenses(["notice"])

proto_library(
    name = "benchmark_proto",
    srcs = [
        "benchmark.proto",
    ],
    cc_api_version = 2,
)

py_proto_library(
    name = "benchmark_proto_py_pb2",
    api_version = 2,
    deps = [":benchmark_proto"],
)

cc_proto_library(
    name = "benchmark_cc_proto",
    deps = [":benchmark_proto"],
)

cc_test(
    name = "serialization_benchmark",
    srcs = ["serialization_benchmark.cc"],
    deps = [
        ":benchmark_cc_proto",
        "//testing/base/public:gunit",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/random:distributions",
        "@com_google_protobuf//:protobuf",
        "@org_tensorflow//tensorflow/core:feature_util",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

py_library(
    name = "struct2tensor_benchmark_lib",
    data = ["//struct2tensor/benchmarks/testdata:data_files"],
    deps = [
        ":benchmark_proto_py_pb2",
        ":struct2tensor_benchmark_util",
        "//struct2tensor",
        "//testing/pybase",
        "//testing/pybase:parameterized",
        "//third_party/py/cpuinfo",
        "//third_party/py/psutil",
    ],
)

py_test(
    name = "struct2tensor_benchmark_test",
    srcs = ["struct2tensor_benchmark.py"],
    # Reduce the run time to fit on TAP.
    # Follow the instructions in the file to properly run the benchmark.
    args = ["--test_mode"],
    main = "struct2tensor_benchmark.py",
    python_version = "PY3",
    shard_count = 4,
    deps = [
        ":struct2tensor_benchmark_lib",
    ],
)

py_binary(
    name = "struct2tensor_benchmark",
    srcs = ["struct2tensor_benchmark.py"],
    python_version = "PY3",
    deps = [
        ":struct2tensor_benchmark_lib",
    ],
)

py_test(
    name = "ops_benchmark_test",
    srcs = ["ops_benchmark.py"],
    # Reduce the run time to fit on TAP.
    # Follow the instructions in the file to properly run the benchmark.
    args = ["--test_mode"],
    main = "ops_benchmark.py",
    python_version = "PY3",
    # shard_count = 4,
    deps = [
        ":struct2tensor_benchmark_lib",
    ],
)

py_binary(
    name = "ops_benchmark",
    srcs = ["ops_benchmark.py"],
    python_version = "PY3",
    deps = [
        ":struct2tensor_benchmark_lib",
    ],
)

py_library(
    name = "struct2tensor_benchmark_util",
    srcs = ["struct2tensor_benchmark_util.py"],
    srcs_version = "PY3ONLY",
    deps = [
        "//file/colossus/public:cns",
        "//third_party/py/cpuinfo",
        "//third_party/py/psutil",
        "@absl_py//absl/flags",
        "@absl_py//absl/testing:parameterized",
    ],
)
