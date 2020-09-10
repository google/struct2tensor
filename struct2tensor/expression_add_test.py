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
"""Tests for struct2tensor.expression_add."""

from absl.testing import absltest

from struct2tensor import create_expression
from struct2tensor import expression_add
from struct2tensor import path
from struct2tensor.test import expression_test_util
from struct2tensor.test import prensor_test_util
import tensorflow as tf


class ModifyTest(absltest.TestCase):

  def test_add_paths(self):
    expr = create_expression.create_expression_from_prensor(
        prensor_test_util.create_nested_prensor())
    new_root = expression_add.add_paths(
        expr, {
            path.Path(["user", "friends_copy"]):
                expr.get_descendant_or_error(path.Path(["user", "friends"]))
        })
    new_field = new_root.get_descendant_or_error(
        path.Path(["user", "friends_copy"]))
    self.assertIsNotNone(new_field)
    self.assertTrue(new_field.is_repeated)
    self.assertEqual(new_field.type, tf.string)
    self.assertTrue(new_field.is_leaf)
    leaf_node = expression_test_util.calculate_value_slowly(new_field)
    self.assertEqual(leaf_node.values.dtype, tf.string)
    self.assertEqual(new_field.known_field_names(), frozenset())

  def test_add_to(self):
    root = create_expression.create_expression_from_prensor(
        prensor_test_util.create_nested_prensor())
    root_1 = expression_add.add_paths(
        root, {
            path.Path(["user", "friends_2"]):
                root.get_descendant_or_error(path.Path(["user", "friends"]))
        })
    root_2 = expression_add.add_paths(
        root_1, {
            path.Path(["user", "friends_3"]):
                root_1.get_descendant_or_error(
                    path.Path(["user", "friends_2"]))
        })
    root_3 = expression_add.add_to(root,
                                   {path.Path(["user", "friends_3"]): root_2})

    new_field = root_3.get_descendant_or_error(path.Path(["user", "friends_3"]))
    self.assertIsNotNone(new_field)
    self.assertTrue(new_field.is_repeated)
    self.assertEqual(new_field.type, tf.string)
    leaf_node = expression_test_util.calculate_value_slowly(new_field)
    self.assertEqual(leaf_node.values.dtype, tf.string)

  def test_add_to_already_existing_path(self):
    with self.assertRaises(ValueError):
      root = create_expression.create_expression_from_prensor(
          prensor_test_util.create_nested_prensor())
      root_1 = expression_add.add_paths(
          root, {
              path.Path(["user", "friends_2"]):
                  root.get_descendant_or_error(path.Path(["user", "friends"]))
          })
      root_2 = expression_add.add_paths(
          root_1, {
              path.Path(["user", "friends_3"]):
                  root_1.get_descendant_or_error(
                      path.Path(["user", "friends_2"]))
          })
      expression_add.add_to(root, {path.Path(["user", "friends"]): root_2})


if __name__ == "__main__":
  absltest.main()
