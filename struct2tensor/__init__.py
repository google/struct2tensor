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
"""Import core names for struct2tensor."""

# Import calculate API.
from struct2tensor.calculate import calculate_prensors
from struct2tensor.calculate import calculate_prensors_with_graph
from struct2tensor.calculate_options import get_default_options
from struct2tensor.calculate_options import get_options_with_minimal_checks
from struct2tensor.calculate_with_source_paths import calculate_prensors_with_source_paths

# Import expressions API.
from struct2tensor.create_expression import create_expression_from_prensor
from struct2tensor.expression import Expression

# Import expression queries API
from struct2tensor.expression_impl.proto import create_expression_from_file_descriptor_set
from struct2tensor.expression_impl.proto import create_expression_from_proto

# Import path API
from struct2tensor.path import create_path
from struct2tensor.path import Path
from struct2tensor.path import Step

# Import prensor API
from struct2tensor.prensor import ChildNodeTensor
from struct2tensor.prensor import create_prensor_from_descendant_nodes
from struct2tensor.prensor import create_prensor_from_root_and_children
from struct2tensor.prensor import LeafNodeTensor
from struct2tensor.prensor import NodeTensor
from struct2tensor.prensor import Prensor
from struct2tensor.prensor import RootNodeTensor

# TODO(b/163167832): Remove these after 0.32.0 is released.
from struct2tensor.prensor_util import get_ragged_tensor
from struct2tensor.prensor_util import get_ragged_tensors
from struct2tensor.prensor_util import get_sparse_tensor
from struct2tensor.prensor_util import get_sparse_tensors

# Importing this will register the session handler for PrensorValue, and
# tf.compat.v1.Session.run() will be able to take a Prensor and return a
# PrensorValue.
import struct2tensor.prensor_value
