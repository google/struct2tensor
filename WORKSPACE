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

# https://github.com/protocolbuffers/protobuf/tree/v3.8.0
PROTOBUF_COMMIT="09745575a923640154bcf307fba8aedff47f240a"

# Needed by tf_py_wrap_cc rule from Tensorflow.
# When upgrading tensorflow version, also check tensorflow/WORKSPACE for the
# version of this -- keep in sync.
# NOTE: tensorflow-serving uses:
# = "2c62d8cd4ab1e65c08647eb4afe38f51591f43f7f0885e7769832fa137633dcb",

git_repository(
    name = "com_google_protobuf",
    commit = PROTOBUF_COMMIT,
    remote = "https://github.com/google/protobuf.git",
)


git_repository(
    name = "protobuf_archive",
    commit = PROTOBUF_COMMIT,
    remote = "https://github.com/google/protobuf.git",
)

load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")

protobuf_deps()



# Fetch tf.Metadata repo from GitHub.
git_repository(
    name = "com_github_tensorflow_metadata",
    commit = "8452a799153412972a4fbf00b9a019db23ef60f9",
    remote = "https://github.com/tensorflow/metadata.git",
)


#####################################################################################

_TENSORFLOW_GIT_COMMIT = "dada3c36881425c5824ac6cf7389548b76a8c91c"

http_archive(
    name = "org_tensorflow",
    sha256 = "5044554782257c269bd6b28e30560aa1d18ed36fe38c6dc52d4bb90c46009592",
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
    sha256 = "e0a111000aeed2051f29fcc7a3f83be3ad8c6c93c186e64beb1ad313f0c7f9f9",
    strip_prefix = "rules_closure-cf1e44edb908e9616030cc83d085989b8e6cd6df",
    urls = [
        "http://mirror.tensorflow.org/github.com/bazelbuild/rules_closure/archive/cf1e44edb908e9616030cc83d085989b8e6cd6df.tar.gz",
        "https://github.com/bazelbuild/rules_closure/archive/cf1e44edb908e9616030cc83d085989b8e6cd6df.tar.gz",  # 2019-04-04
    ],
)

http_archive(
    name = "bazel_skylib",
    sha256 = "2c62d8cd4ab1e65c08647eb4afe38f51591f43f7f0885e7769832fa137633dcb",
    strip_prefix = "bazel-skylib-0.7.0",
    urls = ["https://github.com/bazelbuild/bazel-skylib/archive/0.7.0.tar.gz"],
)
# END: Upstream TensorFlow dependencies


# Please add all new TensorFlow struct2tensor dependencies in workspace.bzl.
load("//struct2tensor:workspace.bzl", "struct2tensor_workspace")

struct2tensor_workspace()

# Specify the minimum required bazel version.
load("@org_tensorflow//tensorflow:version_check.bzl", "check_bazel_version_at_least")

check_bazel_version_at_least("0.24.1")

