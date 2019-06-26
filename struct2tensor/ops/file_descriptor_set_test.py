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

"""Tests for struct2tensor.ops.file_descriptor_set."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from struct2tensor.ops import file_descriptor_set
from struct2tensor.test import dependent_test_pb2

# Notice that while text_extension_pb2 is not used directly,
# it must be linked in and imported so that the extension can be found in the
# pool.
from struct2tensor.test import test_extension_pb2  # pylint: disable=unused-import
from struct2tensor.test import test_map_pb2
from struct2tensor.test import test_pb2


def _get_base_directory():
  return ""  # pylint: disable=unreachable


class FileDescriptorSetTest(absltest.TestCase):

  def test_get_file_descriptor_set_proto_simple_test_map(self):
    file_set_proto = file_descriptor_set.get_file_descriptor_set_proto(
        test_map_pb2.SubMessage.DESCRIPTOR, [])
    self.assertLen(file_set_proto.file, 1)
    self.assertEqual(
        file_set_proto.file[0].name,
        _get_base_directory() + "struct2tensor/test/test_map.proto")


if __name__ == "__main__":
  absltest.main()
