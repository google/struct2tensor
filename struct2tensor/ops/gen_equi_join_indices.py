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
"""Wrapper for _equi_join_indices_op.so."""

from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader

equi_join_indices_module = load_library.load_op_library(
    resource_loader.get_path_to_datafile("_equi_join_indices_op.so")
)
equi_join_indices = equi_join_indices_module.equi_join_indices
