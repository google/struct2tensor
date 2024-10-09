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

from absl.testing import absltest

import struct2tensor as s2t


class Struct2tensorModuleTest(absltest.TestCase):

  def test_importing_struct2tensor_modules(self):
    """This tests that the exposed packages in root __init__.py are found."""
    # pylint: disable=pointless-statement

    # calculate APIs
    s2t.calculate_prensors
    s2t.calculate_prensors_with_graph
    s2t.get_default_options
    s2t.get_options_with_minimal_checks
    s2t.calculate_prensors_with_source_paths

    # expression APIs
    s2t.create_expression_from_prensor
    s2t.create_expression_from_file_descriptor_set
    s2t.create_expression_from_proto
    s2t.Expression

    # path API
    s2t.create_path
    s2t.Path
    s2t.Step

    # prensor APIs
    s2t.ChildNodeTensor
    s2t.LeafNodeTensor
    s2t.NodeTensor
    s2t.Prensor
    s2t.RootNodeTensor
    s2t.create_prensor_from_descendant_nodes
    s2t.create_prensor_from_root_and_children
    s2t.prensor_value
    # pylint: enable=pointless-statement

  def test_importing_expression_impl_modules(self):
    """This tests that the expression_impl/__init__.py imports are found."""
    from struct2tensor import expression_impl  # pylint: disable=g-import-not-at-top

    modules = [
        'apply_schema', 'broadcast', 'depth_limit', 'filter_expression',
        'index', 'map_prensor', 'map_prensor_to_prensor', 'map_values',
        'parquet', 'placeholder', 'project', 'promote', 'promote_and_broadcast',
        'proto', 'reroot', 'size', 'slice_expression'
    ]

    for module in modules:
      getattr(expression_impl, module)


if __name__ == '__main__':
  absltest.main()
