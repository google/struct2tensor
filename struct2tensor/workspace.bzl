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

"""struct2tensor external dependencies that can be loaded in WORKSPACE files."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def struct2tensor_workspace():
    """All struct2tensor external dependencies."""

    # ===== Bazel package rules dependency =====
    http_archive(
        name = "rules_pkg",
        sha256 = "352c090cc3d3f9a6b4e676cf42a6047c16824959b438895a76c2989c6d7c246a",
        url = "https://github.com/bazelbuild/rules_pkg/releases/download/0.2.5/rules_pkg-0.2.5.tar.gz",
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

    # Use the last commit on the relevant release branch to update.
    # LINT.IfChange(arrow_archive_version)
    ARROW_COMMIT = "347a88ff9d20e2a4061eec0b455b8ea1aa8335dc"  # 6.0.1
    # LINT.ThenChange(third_party/arrow.BUILD:arrow_gen_version)

    http_archive(
        name = "arrow",
        build_file = "//third_party:arrow.BUILD",
        sha256 = "55fc466d0043c4cce0756bc18e1e62b3233be74c9afe8dc0d18420b9a5fd9714",
        strip_prefix = "arrow-%s" % ARROW_COMMIT,
        urls = ["https://github.com/apache/arrow/archive/%s.zip" % ARROW_COMMIT],
    )

    _TFMD_COMMIT_HASH = "cdf4d3735a8a94ef5f63b26b87a638dd4d341c5d"  # 1.11.0
    http_archive(
        name = "com_github_tensorflow_metadata",
        sha256 = "18dbfbe8f48ea8de5c5eff18f74a2652e104f7464636d044d790d6bc56bec404",
        strip_prefix = "metadata-%s" % _TFMD_COMMIT_HASH,
        urls = [
            "https://github.com/tensorflow/metadata/archive/%s.tar.gz" % _TFMD_COMMIT_HASH,
        ],
    )
