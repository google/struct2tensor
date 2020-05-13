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

genrule(
    name = "config_h",
    outs = ["config.h"],
    cmd = "\n".join([
        "cat <<'EOF' >$@",
        "#define HAVE_STDDEF_H 1",
        "#define HAVE_STDINT_H 1",
        "",
        "#ifdef __has_builtin",
        "#  if !defined(HAVE_BUILTIN_EXPECT) && __has_builtin(__builtin_expect)",
        "#    define HAVE_BUILTIN_EXPECT 1",
        "#  endif",
        "#  if !defined(HAVE_BUILTIN_CTZ) && __has_builtin(__builtin_ctzll)",
        "#    define HAVE_BUILTIN_CTZ 1",
        "#  endif",
        "#elif defined(__GNUC__) && (__GNUC__ > 3 || __GNUC__ == 3 && __GNUC_MINOR__ >= 4)",
        "#  ifndef HAVE_BUILTIN_EXPECT",
        "#    define HAVE_BUILTIN_EXPECT 1",
        "#  endif",
        "#  ifndef HAVE_BUILTIN_CTZ",
        "#    define HAVE_BUILTIN_CTZ 1",
        "#  endif",
        "#endif",
        "",
        "#ifdef __has_include",
        "#  if !defined(HAVE_BYTESWAP_H) && __has_include(<byteswap.h>)",
        "#    define HAVE_BYTESWAP_H 1",
        "#  endif",
        "#  if !defined(HAVE_UNISTD_H) && __has_include(<unistd.h>)",
        "#    define HAVE_UNISTD_H 1",
        "#  endif",
        "#  if !defined(HAVE_SYS_ENDIAN_H) && __has_include(<sys/endian.h>)",
        "#    define HAVE_SYS_ENDIAN_H 1",
        "#  endif",
        "#  if !defined(HAVE_SYS_MMAN_H) && __has_include(<sys/mman.h>)",
        "#    define HAVE_SYS_MMAN_H 1",
        "#  endif",
        "#  if !defined(HAVE_SYS_UIO_H) && __has_include(<sys/uio.h>)",
        "#    define HAVE_SYS_UIO_H 1",
        "#  endif",
        "#endif",
        "",
        "#ifndef SNAPPY_IS_BIG_ENDIAN",
        "#  ifdef __s390x__",
        "#    define SNAPPY_IS_BIG_ENDIAN 1",
        "#  elif defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__",
        "#    define SNAPPY_IS_BIG_ENDIAN 1",
        "#  endif",
        "#endif",
        "EOF",
    ]),
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
