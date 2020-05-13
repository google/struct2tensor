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

workspace(name = "struct2tensor")

load("//tf:tf_configure.bzl", "tf_configure")

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")


tf_configure(name = "local_config_tf")

git_repository(
    name = "com_github_nelhage_rules_boost",
    commit = "9f9fb8b2f0213989247c9d5c0e814a8451d18d7f",
    remote = "https://github.com/nelhage/rules_boost",
    shallow_since = "1570056263 -0700",
)

load("@com_github_nelhage_rules_boost//:boost/boost.bzl", "boost_deps")
boost_deps()

http_archive(
    name = "com_github_google_flatbuffers",
    sha256 = "12a13686cab7ffaf8ea01711b8f55e1dbd3bf059b7c46a25fefa1250bdd9dd23",
    strip_prefix = "flatbuffers-b99332efd732e6faf60bb7ce1ce5902ed65d5ba3",
    urls = [
        "https://mirror.bazel.build/github.com/google/flatbuffers/archive/b99332efd732e6faf60bb7ce1ce5902ed65d5ba3.tar.gz",
        "https://github.com/google/flatbuffers/archive/b99332efd732e6faf60bb7ce1ce5902ed65d5ba3.tar.gz",
    ],
)

# LINT.IfChange(thrift_archive_version)
http_archive(
    name = "thrift",
    build_file = "//third_party:thrift.BUILD",
    sha256 = "b7452d1873c6c43a580d2b4ae38cfaf8fa098ee6dc2925bae98dce0c010b1366",
    strip_prefix = "thrift-0.12.0",
    urls = [
        "https://github.com/apache/thrift/archive/0.12.0.tar.gz",
    ],
)
# LINT.ThenChange(third_party/thrift.BUILD:thrift_gen_version)

# LINT.IfChange(snappy_archive_version)
http_archive(
    name = "s2t_snappy",
    build_file = "//third_party:snappy.BUILD",
    sha256 = "16b677f07832a612b0836178db7f374e414f94657c138e6993cbfc5dcc58651f",
    strip_prefix = "snappy-1.1.8",
    urls = [
        "https://github.com/google/snappy/archive/1.1.8.tar.gz",
    ],
)
# LINT.ThenChange(third_party/snappy.BUILD:snappy_gen_version)

# LINT.IfChange(arrow_archive_version)
http_archive(
    name = "arrow",
    build_file = "//third_party:arrow.BUILD",
    sha256 = "d7b3838758a365c8c47d55ab0df1006a70db951c6964440ba354f81f518b8d8d",
    strip_prefix = "arrow-apache-arrow-0.16.0",
    urls = [
        "https://github.com/apache/arrow/archive/apache-arrow-0.16.0.tar.gz",
    ],
)
# LINT.ThenChange(third_party/arrow.BUILD:parquet_gen_version)

# https://github.com/protocolbuffers/protobuf/tree/v3.8.0
PROTOBUF_COMMIT="09745575a923640154bcf307fba8aedff47f240a"

# This is needed by the version of tensorflow metadata we depend on.
# TODO(andylou): remove this once com_github_tensorflow_metadata is updated.
git_repository(
    name = "protobuf_archive",
    commit = PROTOBUF_COMMIT,
    remote = "https://github.com/google/protobuf.git",
    shallow_since = "1558721209 -0700",
)

#####################################################################################

_TENSORFLOW_GIT_COMMIT = "2b96f3662bd776e277f86997659e61046b56c315" # 2.2.0

http_archive(
    name = "org_tensorflow",
    sha256 = "b3d7829fac84e3a26264d84057367730b6b85b495a0fce15929568f4b55dc144",
    urls = [
      "https://mirror.bazel.build/github.com/tensorflow/tensorflow/archive/%s.tar.gz" % _TENSORFLOW_GIT_COMMIT,
      "https://github.com/tensorflow/tensorflow/archive/%s.tar.gz" % _TENSORFLOW_GIT_COMMIT,
    ],
    strip_prefix = "tensorflow-%s" % _TENSORFLOW_GIT_COMMIT,
)

# START: Upstream TensorFlow dependencies
# TensorFlow build depends on these dependencies.
# Needs to be in-sync with TensorFlow sources.
http_archive(
    name = "io_bazel_rules_closure",
    sha256 = "5b00383d08dd71f28503736db0500b6fb4dda47489ff5fc6bed42557c07c6ba9",
    strip_prefix = "rules_closure-308b05b2419edb5c8ee0471b67a40403df940149",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/bazelbuild/rules_closure/archive/308b05b2419edb5c8ee0471b67a40403df940149.tar.gz",
        "https://github.com/bazelbuild/rules_closure/archive/308b05b2419edb5c8ee0471b67a40403df940149.tar.gz",  # 2019-06-13
    ],
)
# END: Upstream TensorFlow dependencies

# Please add all new TensorFlow struct2tensor dependencies in workspace.bzl.
load("//struct2tensor:workspace.bzl", "struct2tensor_workspace")
struct2tensor_workspace()

load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")
protobuf_deps()

# Specify the minimum required bazel version.
load("@org_tensorflow//tensorflow:version_check.bzl", "check_bazel_version_at_least")

check_bazel_version_at_least("2.0.0")

