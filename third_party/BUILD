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
#   librarys for exposing vendored files in third_party.

licenses(["notice"])  # Apache 2.0

exports_files([
    "parquet/parquet_types.cpp",
])

cc_library(
    name = "parquet_types_h",
    hdrs = ["parquet/parquet_types.h"],
    includes = ["."],
    visibility = ["//visibility:public"],
)
