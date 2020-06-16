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
"""Tests for struct2tensor.__init__.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest

import struct2tensor as s2t


class Struct2tensorModuleTest(absltest.TestCase):

  def test_importing_modules(self):
    """This tests that the exposed packages in __init__.py are found."""
    # pylint: disable=pointless-statement
    s2t.calculate
    s2t.calculate_options
    s2t.calculate_with_source_paths
    s2t.create_expression
    s2t.expression
    s2t.expression_add
    s2t.expression_impl.apply_schema
    s2t.expression_impl.broadcast
    s2t.expression_impl.depth_limit
    s2t.expression_impl.filter_expression
    s2t.expression_impl.index
    s2t.expression_impl.map_prensor
    s2t.expression_impl.map_prensor_to_prensor
    s2t.expression_impl.map_values
    s2t.expression_impl.parquet
    s2t.expression_impl.placeholder
    s2t.expression_impl.project
    s2t.expression_impl.promote
    s2t.expression_impl.promote_and_broadcast
    s2t.expression_impl.proto
    s2t.expression_impl.reroot
    s2t.expression_impl.size
    s2t.expression_impl.slice_expression
    s2t.path
    s2t.prensor
    s2t.prensor_util
    s2t.prensor_value
    # pylint: enable=pointless-statement


if __name__ == '__main__':
  absltest.main()
