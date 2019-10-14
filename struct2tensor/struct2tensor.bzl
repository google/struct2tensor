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

"""Bazel macros used in OSS."""

load("@com_google_protobuf//:protobuf.bzl", "cc_proto_library", "py_proto_library")

def s2t_pytype_library(
        name,
        srcs = [],
        deps = [],
        srcs_version = "PY2AND3",
        testonly = False):
    native.py_library(name = name, srcs = srcs, deps = deps, testonly = testonly)

def s2t_proto_library(
        name,
        srcs = [],
        has_services = False,
        deps = [],
        visibility = None,
        testonly = 0,
        cc_grpc_version = None,
        cc_api_version = 2):
    """Opensource proto_library.

    Args:
      name: the name of the build target.
      srcs: .proto file sources.
      has_services: no effect
      deps: dependencies
      visibility: visibility constraints
      testonly: if true, only use in tests.
      cc_grpc_version: If set, use grpc plugin.
      cc_api_version: The version of the API in C++.
    """
    _ignore = [has_services]
    native.filegroup(
        name = name + "_proto_srcs",
        srcs = srcs,
        testonly = testonly,
    )

    use_grpc_plugin = None
    if cc_grpc_version:
        use_grpc_plugin = True

    # TODO(martinz): replace with proto_library, when that works.
    cc_proto_library(
        name = name,
        srcs = srcs,
        deps = deps,
        #cc_api_version = cc_api_version,
        cc_libs = ["@com_google_protobuf//:protobuf"],
        protoc = "@com_google_protobuf//:protoc",
        default_runtime = "@com_google_protobuf//:protobuf",
        testonly = testonly,
        visibility = visibility,
    )

DYNAMIC_COPTS = [
    "-pthread",
    "-std=c++11",
    "-D_GLIBCXX_USE_CXX11_ABI=0",
]

DYNAMIC_DEPS = ["@local_config_tf//:libtensorflow_framework", "@local_config_tf//:tf_header_lib"]

def s2t_dynamic_binary(name, deps):
    """Creates a .so file intended for linking with tensorflow_framework.so."""
    native.cc_binary(
        name = name,
        copts = DYNAMIC_COPTS,
        linkshared = 1,
        deps = deps + DYNAMIC_DEPS,
    )

def s2t_dynamic_library(
        name,
        srcs,
        deps = None):
    """Creates a static library intended for linking with tensorflow_framework.so."""
    true_deps = [] if deps == None else deps
    native.cc_library(
        name = name,
        srcs = srcs,
        copts = DYNAMIC_COPTS,
        deps = true_deps + DYNAMIC_DEPS,
    )

def s2t_gen_op_wrapper_py(
        name,
        out,
        static_library,
        dynamic_library):
    """Applies gen_op_wrapper_py externally.

    Instead of a static library, links to a dynamic library.
    Instead of generating a file, one is provided.

    Args:
      name: name of the target
      out: a file that must be provided. Included as source.
      static_library: a static library (ignored).
      dynamic_library: a dynamic library included as data.
    """
    native.py_library(
        name = name,
        srcs = ([
            out,
        ]),
        data = [
            dynamic_library,
        ],
        srcs_version = "PY2AND3",
    )

def s2t_proto_library_cc(
        name,
        srcs = [],
        has_services = False,
        deps = [],
        visibility = None,
        testonly = 0,
        cc_grpc_version = None):
    """Opensource cc_proto_library.

    Args:
        name: name of library
        srcs: .proto sources
        has_services: no effect
        deps: dependencies
        visibility: visibility constraints
        testonly: if true, can only be used in tests.
        cc_grpc_version: if set, use_grpc_version is True.
    """
    _ignore = [has_services]
    native.filegroup(
        name = name + "_proto_srcs",
        srcs = srcs,
        testonly = testonly,
    )

    use_grpc_plugin = None
    if cc_grpc_version:
        use_grpc_plugin = True
    cc_proto_library(
        name = name,
        srcs = srcs,
        deps = deps,
        cc_libs = ["@com_google_protobuf//:protobuf"],
        protoc = "@com_google_protobuf//:protoc",
        default_runtime = "@com_google_protobuf//:protobuf",
        use_grpc_plugin = use_grpc_plugin,
        testonly = testonly,
        visibility = visibility,
    )

def s2t_proto_library_py(name, proto_library, srcs = [], deps = [], oss_deps = [], visibility = None, testonly = 0, api_version = None):
    """Opensource py_proto_library."""
    _ignore = [proto_library, api_version]
    py_proto_library(
        name = name,
        srcs = srcs,
        srcs_version = "PY2AND3",
        deps = ["@com_google_protobuf//:protobuf_python"] + oss_deps,
        default_runtime = "@com_google_protobuf//:protobuf_python",
        protoc = "@com_google_protobuf//:protoc",
        visibility = visibility,
        testonly = testonly,
    )
