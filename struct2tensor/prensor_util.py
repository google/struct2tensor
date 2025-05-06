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
"""Utility methods for using Prensors.

1. get_sparse_tensors(...) gets sparse tensors from a Prensor.
2. get_ragged_tensors(...) gets ragged tensors from a Prensor.

"""

from typing import Mapping

from struct2tensor import calculate_options
from struct2tensor import path
from struct2tensor import prensor
import tensorflow as tf

from tensorflow.python.util import deprecation  # pylint: disable=g-direct-tensorflow-import


@deprecation.deprecated(None, "Use the Prensor class method instead.")
def get_sparse_tensor(
    t: prensor.Prensor,
    p: path.Path,
    options: calculate_options.Options = calculate_options.get_default_options(
    )
) -> tf.SparseTensor:
  """Gets a sparse tensor for path p.

  Note that any optional fields are not registered as dimensions, as they can't
  be represented in a sparse tensor.

  Args:
    t: The Prensor to extract tensors from.
    p: The path to a leaf node in `t`.
    options: Currently unused.

  Returns:
    A sparse tensor containing values of the leaf node, preserving the
    structure along the path. Raises an error if the path is not found.
  """
  return t.get_sparse_tensor(p, options)


@deprecation.deprecated(None, "Use the Prensor class method instead.")
def get_sparse_tensors(
    t: prensor.Prensor,
    options: calculate_options.Options = calculate_options.get_default_options(
    )
) -> Mapping[path.Path, tf.SparseTensor]:
  """Gets sparse tensors for all the leaves of the prensor expression.

  Args:
    t: The Prensor to extract tensors from.
    options: Currently unused.

  Returns:
    A map from paths to sparse tensors.
  """
  return t.get_sparse_tensors(options)


@deprecation.deprecated(None, "Use the Prensor class method instead.")
def get_ragged_tensor(
    t: prensor.Prensor,
    p: path.Path,
    options: calculate_options.Options = calculate_options.get_default_options(
    )
) -> tf.RaggedTensor:
  """Get a ragged tensor for a path.

  All steps are represented in the ragged tensor.

  Args:
    t: The Prensor to extract tensors from.
    p: the path to a leaf node in `t`.
    options: used to pass options for calculating ragged tensors.

  Returns:
    A ragged tensor containing values of the leaf node, preserving the
    structure along the path. Raises an error if the path is not found.
  """
  return t.get_ragged_tensor(p, options)


@deprecation.deprecated(None, "Use the Prensor class method instead.")
def get_ragged_tensors(
    t: prensor.Prensor,
    options: calculate_options.Options = calculate_options.get_default_options(
    )
) -> Mapping[path.Path, tf.RaggedTensor]:
  """Gets ragged tensors for all the leaves of the prensor expression.

  Args:
    t: The Prensor to extract tensors from.
    options: used to pass options for calculating ragged tensors.

  Returns:
    A map from paths to ragged tensors.
  """
  return t.get_ragged_tensors(options)
