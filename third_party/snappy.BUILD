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

# Description:
#   Snappy library

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # BSD 3-Clause

exports_files(["COPYING"])

cc_library(
    name = "snappy",
    srcs = glob(
        [
            "*.cc",
            "*.h",
        ],
        exclude = [
            "*test.*",
            "*fuzzer.*",
        ],
    ),
    hdrs = [
        "snappy-stubs-public.h",
    ],
    copts = [],
    includes = ["."],
)

# LINT.IfChange(snappy_gen_version)
genrule(
    name = "snappy_stubs_public_h",
    srcs = ["snappy-stubs-public.h.in"],
    outs = ["snappy-stubs-public.h"],
    cmd = ("sed " +
           "-e 's/$${HAVE_SYS_UIO_H_01}/HAVE_SYS_UIO_H/g' " +
           "-e 's/$${PROJECT_VERSION_MAJOR}/1/g' " +
           "-e 's/$${PROJECT_VERSION_MINOR}/1/g' " +
           "-e 's/$${PROJECT_VERSION_PATCH}/8/g' " +
           "$< >$@"),
)
# LINT.ThenChange(../WORKSPACE:snappy_archive_version)
