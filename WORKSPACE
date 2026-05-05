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

local_repository(
    name = "python_version_repo",
    path = "third_party/python_version_repo",
)

local_repository(
    name = "python_3_11_host",
    path = "third_party/python_3_11_host",
)

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

local_repository(
    name = "rules_java",
    path = "third_party/rules_java",
)

local_repository(
    name = "local_config_cuda",
    path = "third_party/local_config_cuda",
)

local_repository(
    name = "local_config_tensorrt",
    path = "third_party/local_config_tensorrt",
)

local_repository(
    name = "local_config_rocm",
    path = "third_party/local_config_rocm",
)

local_repository(
    name = "local_config_sycl",
    path = "third_party/local_config_sycl",
)

local_repository(
    name = "tf_wheel_version_suffix",
    path = "third_party/tf_wheel_version_suffix",
)

maybe(
    http_archive,
    name = "platforms",
    urls = [
        "https://github.com/bazelbuild/platforms/releases/download/0.0.11/platforms-0.0.11.tar.gz",
    ],
    sha256 = "29742e87275809b5e598dc2f04d86960cc7a55b3067d97221c9abbc9926bff0f",
)

load("//tf:tf_configure.bzl", "tf_configure")

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")


tf_configure(name = "local_config_tf")

#####################################################################################



# ===== Abseil dependency =====
http_archive(
    name = "com_google_absl",
    sha256 = "d1abe9da2003e6cbbd7619b0ced3e52047422f4f4ac6c66a9bef5d2e99fea837",
    strip_prefix = "abseil-cpp-d38452e1ee03523a208362186fd42248ff2609f6",
    urls = [
        "https://github.com/abseil/abseil-cpp/archive/d38452e1ee03523a208362186fd42248ff2609f6.tar.gz",
    ],
    patches = [
        "//third_party:abseil_visibility.patch",
    ],
    patch_args = ["-p0"],
)

http_archive(
    name = "abseil-cpp",
    sha256 = "d1abe9da2003e6cbbd7619b0ced3e52047422f4f4ac6c66a9bef5d2e99fea837",
    strip_prefix = "abseil-cpp-d38452e1ee03523a208362186fd42248ff2609f6",
    urls = [
        "https://github.com/abseil/abseil-cpp/archive/d38452e1ee03523a208362186fd42248ff2609f6.tar.gz",
    ],
    patches = [
        "//third_party:abseil_visibility.patch",
    ],
    patch_args = ["-p0"],
)




# ===== Protobuf 6.31.1 dependency =====
# Must be declared BEFORE TensorFlow's workspaces to override the version they pull
http_archive(
    name = "com_google_protobuf",
    sha256 = "6e09bbc950ba60c3a7b30280210cd285af8d7d8ed5e0a6ed101c72aff22e8d88",
    strip_prefix = "protobuf-6.31.1",
    urls = [
        "https://github.com/protocolbuffers/protobuf/archive/refs/tags/v6.31.1.zip",
    ],
    patches = ["//third_party:protobuf_tensorflow.patch"],
    patch_args = ["-p1"],
)

register_toolchains(
    "@com_google_protobuf//bazel/private/toolchains:cc_source_toolchain_bazel7",
)

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

_TENSORFLOW_GIT_COMMIT = "2.21.0"  # tf 2.21.0
_TENSORFLOW_ARCHIVE_SHA256 = "ef3568bb4865d6c1b2564fb5689c19b6b9a5311572cd1f2ff9198636a8520921"

http_archive(
    name = "org_tensorflow",
    sha256 = _TENSORFLOW_ARCHIVE_SHA256,
    urls = [
        "https://github.com/tensorflow/tensorflow/archive/v%s.tar.gz" % _TENSORFLOW_GIT_COMMIT,
    ],
    strip_prefix = "tensorflow-%s" % _TENSORFLOW_GIT_COMMIT,
    patches = ["//third_party:tensorflow.patch"],
    patch_args = ["-p1"],
    repo_mapping = {
        "@abseil-cpp": "@com_google_absl",
    },
)

http_archive(
    name = "llvm-raw",
    sha256 = "3f986184ee126677dbd77edb16d6b82c057ec869fefd7a9871979941e52e837a",
    strip_prefix = "llvm-project-909041e4802c4b9a2223ca04099f35bf1dbbd460",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/llvm/llvm-project/archive/909041e4802c4b9a2223ca04099f35bf1dbbd460.tar.gz",
        "https://github.com/llvm/llvm-project/archive/909041e4802c4b9a2223ca04099f35bf1dbbd460.tar.gz",
    ],
    build_file = "@xla//third_party/llvm:llvm.BUILD",
    patches = [
        "@xla//third_party/llvm:generated.patch",
        "@xla//third_party/llvm:build.patch",
        "@xla//third_party/llvm:mathextras.patch",
        "@xla//third_party/llvm:toolchains.patch",
        "@xla//third_party/llvm:zstd.patch",
        "@xla//third_party/llvm:lit_test.patch",
        "//third_party:llvm_configure.patch",
    ],
    patch_args = ["-p1"],
)

http_archive(
    name = "zlib",
    build_file = "@com_google_protobuf//:third_party/zlib.BUILD",
    sha256 = "17e88863f3600672ab49182f217281b6fc4d3c762bde361935e436a95214d05c",
    strip_prefix = "zlib-1.3.1",
    urls = ["https://github.com/madler/zlib/archive/v1.3.1.tar.gz"],
)

# load("//third_party:python_configure.bzl", "local_python_configure")
# local_python_configure(name = "local_config_python")
# local_python_configure(name = "local_execution_config_python")


# Please add all new struct2tensor dependencies in workspace.bzl.
load("//struct2tensor:workspace.bzl", "struct2tensor_workspace")
struct2tensor_workspace()

# Load Protobuf dependencies
load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")
protobuf_deps()

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
versions.check("7.7.0")
