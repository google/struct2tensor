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

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

sh_binary(
    name = "build_pip_pkg",
    testonly = True,  # Some files are testonly
    srcs = ["build_pip_pkg.sh"],
    data = [
        "LICENSE",
        "MANIFEST.in",
        "setup.py",
        "//struct2tensor:__init__.py",
        "//struct2tensor:calculate_options",
        "//struct2tensor:expression",
        "//struct2tensor:expression_impl/__init__.py",
        "//struct2tensor:map_prensor_to_prensor",
        "//struct2tensor:path",
        "//struct2tensor:placeholder",
        "//struct2tensor:prensor",
        "//struct2tensor:proto_test_util",
        "//struct2tensor:version.py",
        "//struct2tensor/ops:__init__.py",
        "//struct2tensor/test:__init__.py",
        "//struct2tensor/test:dependent_test_py_pb2",
        "//struct2tensor/test:expression_test_util",
        "//struct2tensor/test:prensor_test_util",
        "//struct2tensor/test:test_any_py_pb2",
        "//struct2tensor/test:test_extension_py_pb2",
        "//struct2tensor/test:test_map_py_pb2",
        "//struct2tensor/test:test_py_pb2",
    ],
)
