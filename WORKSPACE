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

#####################################################################################

# ===== TensorFlow dependency =====
#
# TensorFlow is imported here instead of in tf_serving_workspace() because
# existing automation scripts that bump the TF commit hash expect it here.
#
# To update TensorFlow to a new revision.
# 1. Update the _TENSORFLOW_GIT_COMMIT value below to include the new git hash.
#    To find it look for the commit which updated the version number:
#    https://github.com/tensorflow/tensorflow/blob/3e6e3ceeedb0dbf2961051fe22002c98a255a6b8/tensorflow/core/public/version.h#L24
# 2. Get the sha256 hash of the archive with a command such as...
#    curl -L https://github.com/tensorflow/tensorflow/archive/<_TENSORFLOW_GIT_COMMIT>.tar.gz | sha256sum
#    and update the 'sha256' arg with the result.
# 3. Request the new archive to be mirrored on mirror.bazel.build for more
#    reliable downloads.

_TENSORFLOW_GIT_COMMIT = "6887368d6d46223f460358323c4b76d61d1558a8"  # tf 2.15.0
_TENSORFLOW_ARCHIVE_SHA256 = "bb25fa4574e42ea4d452979e1d2ba3b86b39569d6b8106a846a238b880d73652"

http_archive(
    name = "org_tensorflow",
    sha256 = _TENSORFLOW_ARCHIVE_SHA256,
    urls = [
        "https://github.com/tensorflow/tensorflow/archive/%s.tar.gz" % _TENSORFLOW_GIT_COMMIT,
    ],
    strip_prefix = "tensorflow-%s" % _TENSORFLOW_GIT_COMMIT,
)

load("//third_party:python_configure.bzl", "local_python_configure")
local_python_configure(name = "local_config_python")
local_python_configure(name = "local_execution_config_python")


# Please add all new struct2tensor dependencies in workspace.bzl.
load("//struct2tensor:workspace.bzl", "struct2tensor_workspace")
struct2tensor_workspace()

# Initialize TensorFlow's external dependencies.
load("@org_tensorflow//tensorflow:workspace3.bzl", "tf_workspace3")
tf_workspace3()
load("@org_tensorflow//tensorflow:workspace2.bzl", "tf_workspace2")
tf_workspace2()
load("@org_tensorflow//tensorflow:workspace1.bzl", "tf_workspace1")
tf_workspace1()
load("@org_tensorflow//tensorflow:workspace0.bzl", "tf_workspace0")
tf_workspace0()

# boost is required for @thrift
git_repository(
    name = "com_github_nelhage_rules_boost",
    commit = "ce0caa8aa9593cb8664d4b5448978fbb94acc8b9",
    remote = "https://github.com/nelhage/rules_boost",
)

load("@com_github_nelhage_rules_boost//:boost/boost.bzl", "boost_deps")
boost_deps()

# Initialize bazel package rules' external dependencies.
load("@rules_pkg//:deps.bzl", "rules_pkg_dependencies")
rules_pkg_dependencies()

# Specify the minimum required bazel version.
load("@bazel_skylib//lib:versions.bzl", "versions")
versions.check("6.1.0")
