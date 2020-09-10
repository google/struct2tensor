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
"""Tests for struct2tensor.broadcast."""

from absl.testing import absltest
from struct2tensor import create_expression
from struct2tensor import path
from struct2tensor.expression_impl import promote_and_broadcast
from struct2tensor.test import expression_test_util
from struct2tensor.test import prensor_test_util
import tensorflow as tf


class PromoteAndBroadcastTest(absltest.TestCase):

  def test_promote_and_broadcast_anonymous(self):
    """A basic promote and broadcast."""
    expr = create_expression.create_expression_from_prensor(
        prensor_test_util.create_big_prensor())
    new_root, p = promote_and_broadcast.promote_and_broadcast_anonymous(
        expr, path.Path(["user", "friends"]), path.Path(["doc"]))
    new_field = new_root.get_descendant_or_error(p)
    self.assertTrue(new_field.is_repeated)
    self.assertEqual(new_field.type, tf.string)
    self.assertTrue(new_field.is_leaf)
    self.assertTrue(new_field.calculation_equal(new_field))
    self.assertFalse(new_field.calculation_equal(expr))
    leaf_node = expression_test_util.calculate_value_slowly(new_field)
    self.assertEqual(leaf_node.values.dtype, tf.string)
    self.assertEqual(new_field.known_field_names(), frozenset())


if __name__ == "__main__":
  absltest.main()
