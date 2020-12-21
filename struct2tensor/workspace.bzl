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

"""TensorFlow struct2tensor external dependencies that can be loaded in WORKSPACE files.
"""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@org_tensorflow//tensorflow:workspace.bzl", "tf_workspace")

def struct2tensor_workspace():
    """All TensorFlow struct2tensor external dependencies."""
    tf_workspace(
        path_prefix = "",
        tf_repo_name = "org_tensorflow",
    )

    _TFMD_COMMIT_HASH = "1194a9de032c1eaf9d4e13efb31934f88e4fa4f5"  # 0.26.0
    http_archive(
        name = "com_github_tensorflow_metadata",
        sha256 = "61c854f4f0605106517bb07c6574ff27b8fcdf71485ebf009eaf9759391e3d8c",
        strip_prefix = "metadata-%s" % _TFMD_COMMIT_HASH,
        urls = [
            "https://github.com/tensorflow/metadata/archive/%s.tar.gz" % _TFMD_COMMIT_HASH,
        ],
    )
