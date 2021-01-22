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

# Description:
#   Apache arrow library

load("@com_github_google_flatbuffers//:build_defs.bzl", "flatbuffer_cc_library")

flatbuffer_cc_library(
    name = "arrow_format",
    srcs = [
        "cpp/src/arrow/ipc/feather.fbs",
        "format/File.fbs",
        "format/Message.fbs",
        "format/Schema.fbs",
        "format/SparseTensor.fbs",
        "format/Tensor.fbs",
    ],
    flatc_args = [
        "--scoped-enums",
        "--gen-object-api",
    ],
    out_prefix = "cpp/src/generated/",
)

package(default_visibility = ["//visibility:public"])

genrule(
    name = "arrow_util_config",
    srcs = ["cpp/src/arrow/util/config.h.cmake"],
    outs = ["cpp/src/arrow/util/config.h"],
    cmd = ("sed " +
           "-e 's/@ARROW_VERSION_MAJOR@/0/g' " +
           "-e 's/@ARROW_VERSION_MINOR@/16/g' " +
           "-e 's/@ARROW_VERSION_PATCH@/0/g' " +
           "-e 's/cmakedefine/define/g' " +
           "$< >$@"),
)

# LINT.IfChange(parquet_gen_version)
genrule(
    name = "parquet_version_h",
    srcs = ["cpp/src/parquet/parquet_version.h.in"],
    outs = ["cpp/src/parquet/parquet_version.h"],
    cmd = ("sed " +
           "-e 's/@PARQUET_VERSION_MAJOR@/1/g' " +
           "-e 's/@PARQUET_VERSION_MINOR@/5/g' " +
           "-e 's/@PARQUET_VERSION_PATCH@/1/g' " +
           "$< >$@"),
)
# LINT.ThenChange(../workspace.bzl:arrow_archive_version)

cc_library(
    name = "xxhash",
    srcs = [],
    hdrs = [
        "cpp/src/arrow/vendored/xxhash/xxh3.h",
        "cpp/src/arrow/vendored/xxhash/xxhash.c",
        "cpp/src/arrow/vendored/xxhash/xxhash.h",
    ],
    copts = ["-Wno-implicit-fallthrough"],
    visibility = ["//visibility:private"],
)

cc_library(
    name = "arrow",
    srcs = glob(
        [
            "cpp/src/arrow/*.cc",
            "cpp/src/arrow/array/*.cc",
            "cpp/src/arrow/csv/*.cc",
            "cpp/src/arrow/io/*.cc",
            "cpp/src/arrow/util/*.cc",
            "cpp/src/arrow/compute/**/*.cc",
            "cpp/src/arrow/compute/**/*.h",
            "cpp/src/arrow/vendored/optional.hpp",
            "cpp/src/arrow/vendored/string_view.hpp",
            "cpp/src/arrow/vendored/variant.hpp",
            "cpp/src/arrow/vendored/**/*.cpp",
            "cpp/src/arrow/vendored/**/*.hpp",
            "cpp/src/arrow/vendored/**/*.cc",
            "cpp/src/arrow/vendored/**/*.c",
            "cpp/src/arrow/vendored/**/*.h",
            "cpp/src/arrow/ipc/*.cc",
            "cpp/src/arrow/ipc/*.h",
            "cpp/src/arrow/**/*.h",
            "cpp/src/parquet/**/*.h",
            "cpp/src/parquet/**/*.cc",
        ],
        exclude = [
            "cpp/src/arrow/util/compression_brotli*",
            "cpp/src/arrow/util/compression_bz2*",
            "cpp/src/arrow/util/compression_lz4*",
            "cpp/src/arrow/util/compression_z*",
            "cpp/src/arrow/util/compression_snappy*",
            "cpp/src/**/*_benchmark.cc",
            "cpp/src/**/*_main.cc",
            "cpp/src/**/*_test.cc",
            "cpp/src/**/*-test.cc",
            "cpp/src/**/test_*.cc",
            "cpp/src/**/*hdfs*.cc",
            "cpp/src/**/*fuzz*.cc",
            "cpp/src/**/file_to_stream.cc",
            "cpp/src/**/stream_to_file.cc",
            "cpp/src/arrow/ipc/json*.cc",
            "cpp/src/arrow/vendored/xxhash/**",
            "cpp/src/parquet/encryption_internal.cc",
        ],
    ) + [
        "@struct2tensor//third_party:parquet/parquet_types.cpp",
    ],
    hdrs = [
        # declare header from above genrule
        "cpp/src/arrow/util/config.h",
        "cpp/src/parquet/parquet_version.h",
    ],
    includes = [
        "cpp/src",
        "cpp/src/arrow/vendored/xxhash",
    ],
    textual_hdrs = [
        "cpp/src/arrow/vendored/xxhash/xxhash.c",
    ],
    deps = [
        ":arrow_format",
        ":xxhash",
        "@struct2tensor//third_party:parquet_types_h",
        "@thrift",
    ],
)
