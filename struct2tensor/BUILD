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

# Prensors provide a way to parse protocol buffers and other structured data in tensorflow
# simply and efficiently.

package(default_visibility = [
    "//learning/tfx/autotfx:__subpackages__",
    "//learning/tfx/users/tfx/util/tf_ranking:__subpackages__",
    "//nlp/nlx/infrastructure/multiscale:__subpackages__",
    "//third_party/py/tfx_bsl:__subpackages__",
    "//video/youtube/discovery/tensorflow/python/input/feature_parser:__subpackages__",
    "@//:__subpackages__",
])

load("//struct2tensor:struct2tensor.bzl", "s2t_pytype_library")

licenses(["notice"])

exports_files([
    "LICENSE",
    "__init__.py",
    "expression_impl/__init__.py",
    "version.py",
])

# Contains both op registrations and kernels. Programs that need to run a Prensor
# graph (e.g. servo) should depend on this target.
# Prensor ops are TF canonical ops.
cc_library(
    name = "struct2tensor_kernels_and_ops",
    deps = [
        "//struct2tensor/kernels:struct2tensor_kernels",
        "//struct2tensor/ops:struct2tensor_op_registrations",
    ],
    alwayslink = 1,
)

s2t_pytype_library(
    name = "path",
    srcs = [
        "path.py",
    ],
    srcs_version = "PY2AND3",
    deps = ["@com_github_tensorflow_metadata//tensorflow_metadata/proto/v0:metadata_v0_proto_py_pb2"],
)

s2t_pytype_library(
    name = "tf_types",
    srcs = ["tf_types.py"],
)

# Note: do not separate these, as eventually we want to make sure that they
# are all included in a single package.
s2t_pytype_library(
    name = "prensor",
    srcs = [
        "prensor.py",
        "prensor_util.py",
        "prensor_value.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":calculate_options",
        ":path",
        ":tf_types",
        "//struct2tensor/ops:struct2tensor_ops",
        "//struct2tensor/proto:query_metadata_py_pb2",
        "@protobuf_archive//:protobuf_python",
    ],
)

# Note: do not separate these, as eventually we want to make sure that they
# are all included in a single package.
s2t_pytype_library(
    name = "calculate_options",
    srcs = [
        "calculate_options.py",
    ],
    srcs_version = "PY2AND3",
)

# Note: do not separate these, as eventually we want to make sure that they
# are all included in a single package.
s2t_pytype_library(
    name = "expression",
    srcs = [
        "calculate.py",
        "calculate_with_source_paths.py",
        "create_expression.py",
        "expression.py",
        "expression_add.py",
        "expression_impl/apply_schema.py",
        "expression_impl/broadcast.py",
        "expression_impl/depth_limit.py",
        "expression_impl/filter_expression.py",
        "expression_impl/index.py",
        "expression_impl/map_prensor.py",
        "expression_impl/map_values.py",
        "expression_impl/parse_message_level_ex.py",
        "expression_impl/project.py",
        "expression_impl/promote.py",
        "expression_impl/promote_and_broadcast.py",
        "expression_impl/proto.py",
        "expression_impl/reroot.py",
        "expression_impl/size.py",
        "expression_impl/slice_expression.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":calculate_options",
        ":path",
        ":prensor",
        "//struct2tensor/ops:struct2tensor_ops",
        "@protobuf_archive//:protobuf_python",
    ],
)

# Note: do not separate these, as eventually we want to make sure that they
# are all included in a single package.
s2t_pytype_library(
    name = "map_prensor_to_prensor",
    srcs = [
        "expression_impl/map_prensor_to_prensor.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":expression",
        ":path",
        ":prensor",
    ],
)

s2t_pytype_library(
    name = "placeholder",
    srcs = [
        "expression_impl/placeholder.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":expression",
        ":map_prensor_to_prensor",
        ":path",
        ":prensor",
    ],
)

s2t_pytype_library(
    name = "proto_test_util",
    testonly = 1,
    srcs = [
        "expression_impl/proto_test_util.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":expression",
        "//struct2tensor/test:test_py_pb2",
        "@protobuf_archive//:protobuf_python",
    ],
)
