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
"""Tests for struct2tensor.create_expression."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from struct2tensor import path
from struct2tensor.test import prensor_test_util
from struct2tensor import prensor_value


class PrensorValueTest(tf.test.TestCase):

  def test_materialize(self):
    """Tests get_sparse_tensors on a deep expression."""
    with self.session(use_gpu=False) as sess:
      pren = prensor_test_util.create_nested_prensor()
      mat = prensor_value.materialize(pren, sess)
      print("materialized: {}".format(str(mat)))
      print("size:{}".format(str(type(mat.node.size))))
      self.assertAllEqual(
          mat.get_descendant_or_error(path.Path(["doc", "bar"])).node.values,
          [b"a", b"b", b"c", b"d"])
      self.assertAllEqual(
          mat.get_descendant_or_error(path.Path(
              ["doc", "bar"])).node.parent_index, [0, 1, 1, 2])
      self.assertAllEqual(
          mat.get_descendant_or_error(path.Path(["doc"])).node.parent_index,
          [0, 1, 1])
      self.assertAllEqual(
          mat.get_descendant_or_error(path.Path([])).node.size, 3)
      self.assertAllEqual(
          mat.get_descendant_or_error(path.Path(
              ["doc", "keep_me"])).node.parent_index, [0, 1])
      self.assertAllEqual(
          mat.get_descendant_or_error(path.Path(
              ["doc", "keep_me"])).node.values, [False, True])


if __name__ == "__main__":
  tf.test.main()
