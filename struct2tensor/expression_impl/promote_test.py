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
"""Tests for struct2tensor.promote."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from struct2tensor import create_expression
from struct2tensor import path
from struct2tensor.expression_impl import promote
from struct2tensor.test import expression_test_util
from struct2tensor.test import prensor_test_util
import tensorflow as tf

import unittest
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import
from tensorflow_metadata.proto.v0 import schema_pb2


class PromoteTest(unittest.TestCase):

  def assertLen(self, arr, expected_len):
    self.assertEqual(len(arr), expected_len)  # pylint:disable=g-generic-assert

  def test_promote_anonymous(self):
    expr = create_expression.create_expression_from_prensor(
        prensor_test_util.create_nested_prensor())
    new_root, new_field = promote.promote_anonymous(
        expr, path.Path(["user", "friends"]))
    new_field = new_root.get_descendant_or_error(new_field)
    self.assertTrue(new_field.is_repeated)
    self.assertEqual(new_field.type, tf.string)
    self.assertTrue(new_field.is_leaf)
    self.assertFalse(new_field.calculation_is_identity())
    self.assertTrue(new_field.calculation_equal(new_field))
    self.assertFalse(new_field.calculation_equal(expr))
    leaf_node = expression_test_util.calculate_value_slowly(new_field)
    self.assertEqual(leaf_node.values.dtype, tf.string)
    self.assertEqual(new_field.known_field_names(), frozenset())

    sources = new_field.get_source_expressions()
    self.assertLen(sources, 2)
    self.assertIs(
        expr.get_descendant_or_error(path.Path(["user", "friends"])),
        sources[0])
    self.assertIs(expr.get_child_or_error("user"), sources[1])

  def test_promote_with_schema(self):
    expr = create_expression.create_expression_from_prensor(
        prensor_test_util.create_big_prensor()).apply_schema(
            prensor_test_util.create_big_prensor_schema())

    new_root, new_field = promote.promote_anonymous(
        expr, path.Path(["user", "friends"]))
    new_field = new_root.get_descendant_or_error(new_field)
    new_schema_feature = new_field.schema_feature
    self.assertIsNotNone(new_schema_feature)
    self.assertEqual(new_schema_feature.string_domain.value[0], "a")

  def test_promote_with_schema_dense_parent(self):
    s = prensor_test_util.create_big_prensor_schema()
    feature_dict = {feature.name: feature for feature in s.feature}
    user_feature = feature_dict["user"]
    user_feature.value_count.min = 3
    user_feature.value_count.max = 3
    user_feature.presence.min_fraction = 1
    user_feature.lifecycle_stage = schema_pb2.LifecycleStage.ALPHA

    user_dict = {
        feature.name: feature for feature in user_feature.struct_domain.feature
    }
    friends_feature = user_dict["friends"]
    friends_feature.value_count.min = 2
    friends_feature.value_count.max = 2
    friends_feature.presence.min_fraction = 1
    friends_feature.presence.min_count = 10
    friends_feature.lifecycle_stage = schema_pb2.LifecycleStage.BETA
    friends_feature.distribution_constraints.min_domain_mass = 0.5

    expr = create_expression.create_expression_from_prensor(
        prensor_test_util.create_big_prensor()).apply_schema(s)

    new_root, new_field = promote.promote_anonymous(
        expr, path.Path(["user", "friends"]))
    new_field = new_root.get_descendant_or_error(new_field)
    new_schema_feature = new_field.schema_feature
    self.assertIsNotNone(new_schema_feature)
    self.assertEqual(new_schema_feature.string_domain.value[0], "a")
    self.assertEqual(new_schema_feature.value_count.max, 6)
    self.assertEqual(new_schema_feature.value_count.min, 6)
    self.assertEqual(new_schema_feature.presence.min_fraction, 1)
    self.assertEqual(new_schema_feature.presence.min_count, 3)
    self.assertEqual(new_schema_feature.lifecycle_stage,
                     schema_pb2.LifecycleStage.ALPHA)
    self.assertEqual(
        new_schema_feature.distribution_constraints.min_domain_mass, 0.5)

  def test_lifecycle_stage(self):
    # Stages have the following priority, from lowest to highest:
    #   schema_pb2.LifecycleStage.DEPRECATED
    #   schema_pb2.LifecycleStage.PLANNED,
    #   schema_pb2.LifecycleStage.ALPHA
    #   schema_pb2.LifecycleStage.DEBUG_ONLY,
    #   None
    #   schema_pb2.LifecycleStage.UNKNOWN_STAGE,
    #   schema_pb2.LifecycleStage.BETA
    #   schema_pb2.LifecycleStage.PRODUCTION
    def _check_lifecycle_stage(a, b):
      s = prensor_test_util.create_big_prensor_schema()
      feature_dict = {feature.name: feature for feature in s.feature}
      user_feature = feature_dict["user"]
      if a is not None:
        user_feature.lifecycle_stage = a

      user_dict = {
          feature.name: feature
          for feature in user_feature.struct_domain.feature
      }
      friends_feature = user_dict["friends"]
      if b is not None:
        friends_feature.lifecycle_stage = b

      expr = create_expression.create_expression_from_prensor(
          prensor_test_util.create_big_prensor()).apply_schema(s)

      new_root, new_field = promote.promote_anonymous(
          expr, path.Path(["user", "friends"]))
      new_field = new_root.get_descendant_or_error(new_field)
      return new_field.schema_feature.lifecycle_stage

    self.assertEqual(
        schema_pb2.LifecycleStage.DEPRECATED,
        _check_lifecycle_stage(schema_pb2.LifecycleStage.DEPRECATED,
                               schema_pb2.LifecycleStage.PLANNED))
    self.assertEqual(
        schema_pb2.LifecycleStage.DEPRECATED,
        _check_lifecycle_stage(schema_pb2.LifecycleStage.PLANNED,
                               schema_pb2.LifecycleStage.DEPRECATED))

    self.assertEqual(
        schema_pb2.LifecycleStage.PLANNED,
        _check_lifecycle_stage(schema_pb2.LifecycleStage.PLANNED,
                               schema_pb2.LifecycleStage.ALPHA))
    self.assertEqual(
        schema_pb2.LifecycleStage.PLANNED,
        _check_lifecycle_stage(schema_pb2.LifecycleStage.ALPHA,
                               schema_pb2.LifecycleStage.PLANNED))

    self.assertEqual(
        schema_pb2.LifecycleStage.ALPHA,
        _check_lifecycle_stage(schema_pb2.LifecycleStage.DEBUG_ONLY,
                               schema_pb2.LifecycleStage.ALPHA))
    self.assertEqual(
        schema_pb2.LifecycleStage.ALPHA,
        _check_lifecycle_stage(schema_pb2.LifecycleStage.ALPHA,
                               schema_pb2.LifecycleStage.DEBUG_ONLY))

    self.assertEqual(
        schema_pb2.LifecycleStage.DEBUG_ONLY,
        _check_lifecycle_stage(schema_pb2.LifecycleStage.DEBUG_ONLY, None))

    self.assertEqual(
        schema_pb2.LifecycleStage.DEBUG_ONLY,
        _check_lifecycle_stage(None, schema_pb2.LifecycleStage.DEBUG_ONLY))

    # None looks like UNKNOWN_STAGE.
    self.assertEqual(
        schema_pb2.LifecycleStage.UNKNOWN_STAGE,
        _check_lifecycle_stage(None, schema_pb2.LifecycleStage.UNKNOWN_STAGE))
    self.assertEqual(
        schema_pb2.LifecycleStage.UNKNOWN_STAGE,
        _check_lifecycle_stage(schema_pb2.LifecycleStage.UNKNOWN_STAGE, None))

    self.assertEqual(
        schema_pb2.LifecycleStage.UNKNOWN_STAGE,
        _check_lifecycle_stage(schema_pb2.LifecycleStage.BETA,
                               schema_pb2.LifecycleStage.UNKNOWN_STAGE))
    self.assertEqual(
        schema_pb2.LifecycleStage.UNKNOWN_STAGE,
        _check_lifecycle_stage(schema_pb2.LifecycleStage.UNKNOWN_STAGE,
                               schema_pb2.LifecycleStage.BETA))

    self.assertEqual(
        schema_pb2.LifecycleStage.BETA,
        _check_lifecycle_stage(schema_pb2.LifecycleStage.BETA,
                               schema_pb2.LifecycleStage.PRODUCTION))
    self.assertEqual(
        schema_pb2.LifecycleStage.BETA,
        _check_lifecycle_stage(schema_pb2.LifecycleStage.PRODUCTION,
                               schema_pb2.LifecycleStage.BETA))

  def test_promote_with_schema_dense_fraction(self):
    """Test when min_fraction is not 1."""
    s = prensor_test_util.create_big_prensor_schema()
    feature_dict = {feature.name: feature for feature in s.feature}
    user_feature = feature_dict["user"]
    user_feature.value_count.min = 3
    user_feature.value_count.max = 3
    user_feature.presence.min_fraction = 1

    user_dict = {
        feature.name: feature for feature in user_feature.struct_domain.feature
    }
    friends_feature = user_dict["friends"]
    friends_feature.presence.min_fraction = 0.9

    expr = create_expression.create_expression_from_prensor(
        prensor_test_util.create_big_prensor()).apply_schema(s)

    new_root, new_field = promote.promote_anonymous(
        expr, path.Path(["user", "friends"]))
    new_field = new_root.get_descendant_or_error(new_field)
    new_schema_feature = new_field.schema_feature
    self.assertIsNotNone(new_schema_feature)
    self.assertEqual(new_schema_feature.presence.min_fraction, 0.3)

  def test_promote_optional_child_of_repeated(self):
    expr = create_expression.create_expression_from_prensor(
        prensor_test_util.create_nested_prensor())
    new_root, new_field = promote.promote_anonymous(
        expr, path.Path(["doc", "keep_me"]))
    new_expr = new_root.get_descendant_or_error(new_field)
    self.assertTrue(new_expr.is_repeated)

  def test_promote(self):
    """Tests promote.promote(...), and indirectly tests set_path."""
    expr = create_expression.create_expression_from_prensor(
        prensor_test_util.create_nested_prensor())
    new_root = promote.promote(expr, path.Path(["user", "friends"]),
                               "new_field")
    new_field = new_root.get_child_or_error("new_field")
    self.assertIsNotNone(new_field)
    self.assertTrue(new_field.is_repeated)
    self.assertEqual(new_field.type, tf.string)
    self.assertTrue(new_field.is_leaf)
    leaf_node = expression_test_util.calculate_value_slowly(new_field)
    self.assertEqual(leaf_node.values.dtype, tf.string)
    self.assertEqual(new_field.known_field_names(), frozenset())


@test_util.run_all_in_graph_and_eager_modes
class PromoteValuesTest(tf.test.TestCase):

  def test_promote_and_calculate(self):
    """Tests get_sparse_tensors on a deep tree."""
    expr = create_expression.create_expression_from_prensor(
        prensor_test_util.create_nested_prensor())
    new_root, new_path = promote.promote_anonymous(
        expr, path.Path(["user", "friends"]))
    new_field = new_root.get_descendant_or_error(new_path)
    leaf_node = expression_test_util.calculate_value_slowly(new_field)
    self.assertAllEqual(leaf_node.parent_index, [0, 1, 1, 1, 2])
    self.assertAllEqual(leaf_node.values, [b"a", b"b", b"c", b"d", b"e"])


if __name__ == "__main__":
  unittest.main()
