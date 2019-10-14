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
"""Contains aliases of Tensorflow types for compatibility.

This can be removed once a TF minor release after 1.14 is available.
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import tensorflow as tf
# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.framework import composite_tensor

TypeSpec = object


class CompositeTensorMixin(composite_tensor.CompositeTensor):
  """Backported TF 2.0's CompositeTensor which uses _type_spec."""

  # pylint: disable=protected-access
  def _to_components(self):
    return self._type_spec._to_components(self)

  def _component_metadata(self):
    return self._type_spec

  @staticmethod
  def _from_components(components, metadata):
    return metadata._from_components(components)

  def _shape_invariant_to_components(self, shape=None):
    raise NotImplementedError("%s._shape_invariant_to_type_spec"
                              % type(self).__name__)

  def _is_graph_tensor(self):
    components = self._type_spec._to_components(self)
    tensors = tf.nest.flatten(components, expand_composites=True)
    return any(hasattr(t, "graph") for t in tensors)
