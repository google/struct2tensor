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

from struct2tensor import create_expression
from struct2tensor import path
from struct2tensor.expression_impl import promote
from struct2tensor.test import expression_test_util
from struct2tensor.test import prensor_test_util
import tensorflow as tf

from absl.testing import absltest
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import
from tensorflow_metadata.proto.v0 import schema_pb2


class PromoteTest(absltest.TestCase):

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
    #   schema_pb2.LifecycleStage.DISABLED
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
                               schema_pb2.LifecycleStage.DISABLED))
    self.assertEqual(
        schema_pb2.LifecycleStage.DEPRECATED,
        _check_lifecycle_stage(schema_pb2.LifecycleStage.DISABLED,
                               schema_pb2.LifecycleStage.DEPRECATED))

    self.assertEqual(
        schema_pb2.LifecycleStage.DISABLED,
        _check_lifecycle_stage(schema_pb2.LifecycleStage.PLANNED,
                               schema_pb2.LifecycleStage.DISABLED))

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

  def test_promote_substructure(self):
    """Tests promote.promote(...) of substructure."""
    expr = create_expression.create_expression_from_prensor(
        prensor_test_util.create_deep_prensor())
    new_root = promote.promote(expr, path.Path(["event", "doc"]), "new_field")

    new_field = new_root.get_child_or_error("new_field")
    self.assertIsNotNone(new_field)
    self.assertTrue(new_field.is_repeated)
    self.assertEqual(new_field.known_field_names(),
                     frozenset(["bar", "keep_me"]))

    bar_expr = new_field.get_child_or_error("bar")
    self.assertIsNotNone(bar_expr)
    self.assertTrue(bar_expr.is_repeated)
    self.assertEqual(bar_expr.type, tf.string)
    self.assertTrue(bar_expr.is_leaf)

    keep_me_expr = new_field.get_child_or_error("keep_me")
    self.assertIsNotNone(keep_me_expr)
    self.assertFalse(keep_me_expr.is_repeated)
    self.assertEqual(keep_me_expr.type, tf.bool)
    self.assertTrue(keep_me_expr.is_leaf)

    child_node = expression_test_util.calculate_value_slowly(new_field)
    self.assertEqual(child_node.size, 3)
    self.assertTrue(child_node.is_repeated)

    bar_node = expression_test_util.calculate_value_slowly(bar_expr)
    self.assertEqual(bar_node.values.dtype, tf.string)

    keep_me_node = expression_test_util.calculate_value_slowly(keep_me_expr)
    self.assertEqual(keep_me_node.values.dtype, tf.bool)

  def test_promote_substructure_then_leaf(self):
    """Tests expr.promote(...) of substructure and then a leaf."""
    expr = create_expression.create_expression_from_prensor(
        prensor_test_util.create_deep_prensor())
    new_root = (expr
                .promote(path.Path(["event", "doc"]), "new_field")
                .promote(path.Path(["new_field", "bar"]), "new_bar"))

    new_bar = new_root.get_child_or_error("new_bar")
    self.assertIsNotNone(new_bar)
    self.assertTrue(new_bar.is_repeated)
    self.assertEqual(new_bar.type, tf.string)
    self.assertTrue(new_bar.is_leaf)

    new_field_bar = new_root.get_descendant_or_error(
        path.Path(["new_field", "bar"]))
    self.assertIsNotNone(new_field_bar)
    self.assertTrue(new_bar.is_repeated)
    self.assertEqual(new_bar.type, tf.string)
    self.assertTrue(new_bar.is_leaf)

    new_field_keep_me = new_root.get_descendant_or_error(
        path.Path(["new_field", "keep_me"]))
    self.assertIsNotNone(new_field_keep_me)
    self.assertFalse(new_field_keep_me.is_repeated)
    self.assertEqual(new_field_keep_me.type, tf.bool)
    self.assertTrue(new_field_keep_me.is_leaf)

    bar_node = expression_test_util.calculate_value_slowly(new_bar)
    self.assertEqual(bar_node.values.dtype, tf.string)

    new_field_bar_node = expression_test_util.calculate_value_slowly(
        new_field_bar)
    self.assertEqual(new_field_bar_node.values.dtype, tf.string)

    new_field_keep_me_node = expression_test_util.calculate_value_slowly(
        new_field_keep_me)
    self.assertEqual(new_field_keep_me_node.values.dtype, tf.bool)

  def test_promote_leaf_then_substructure(self):
    """Tests expr.promote(...) of leaf and then a substructure."""
    expr = create_expression.create_expression_from_prensor(
        prensor_test_util.create_four_layer_prensor())
    new_root = (
        expr
        .promote(path.Path(["event", "doc", "nested_child", "bar"]), "new_bar")
        .promote(path.Path(["event", "doc"]), "new_doc"))

    new_doc = new_root.get_child_or_error("new_doc")
    self.assertIsNotNone(new_doc)
    self.assertTrue(new_doc.is_repeated)
    self.assertEqual(new_doc.known_field_names(),
                     frozenset(["nested_child", "new_bar"]))

    new_bar_expr = new_doc.get_child_or_error("new_bar")
    self.assertIsNotNone(new_bar_expr)
    self.assertTrue(new_bar_expr.is_repeated)
    self.assertEqual(new_bar_expr.type, tf.string)
    self.assertTrue(new_bar_expr.is_leaf)

    nested_child_expr = new_doc.get_child_or_error("nested_child")
    self.assertIsNotNone(nested_child_expr)
    self.assertTrue(nested_child_expr.is_repeated)
    self.assertEqual(nested_child_expr.known_field_names(),
                     frozenset(["bar", "keep_me"]))

    bar_expr = nested_child_expr.get_child_or_error("bar")
    self.assertIsNotNone(bar_expr)
    self.assertTrue(bar_expr.is_repeated)
    self.assertEqual(bar_expr.type, tf.string)
    self.assertTrue(bar_expr.is_leaf)

    keep_me_expr = nested_child_expr.get_child_or_error("keep_me")
    self.assertIsNotNone(keep_me_expr)
    self.assertFalse(keep_me_expr.is_repeated)
    self.assertEqual(keep_me_expr.type, tf.bool)
    self.assertTrue(keep_me_expr.is_leaf)

    bar_node = expression_test_util.calculate_value_slowly(new_bar_expr)
    self.assertEqual(bar_node.values.dtype, tf.string)


@test_util.run_all_in_graph_and_eager_modes
class PromoteValuesTest(tf.test.TestCase):

  def test_promote_and_calculate(self):
    """Tests promoting a leaf on a nested tree."""
    expr = create_expression.create_expression_from_prensor(
        prensor_test_util.create_nested_prensor())
    new_root, new_path = promote.promote_anonymous(
        expr, path.Path(["user", "friends"]))
    new_field = new_root.get_descendant_or_error(new_path)
    leaf_node = expression_test_util.calculate_value_slowly(new_field)
    self.assertAllEqual(leaf_node.parent_index, [0, 1, 1, 1, 2])
    self.assertAllEqual(leaf_node.values, [b"a", b"b", b"c", b"d", b"e"])

  def test_promote_and_calculate_substructure(self):
    """Tests promoting substructure on a tree with depth of 4."""
    expr = create_expression.create_expression_from_prensor(
        prensor_test_util.create_four_layer_prensor())
    new_root, new_path = promote.promote_anonymous(
        expr, path.Path(["event", "doc", "nested_child"]))
    new_nested_child = new_root.get_descendant_or_error(new_path)
    bar_expr = new_root.get_descendant_or_error(new_path.get_child("bar"))
    keep_me_expr = new_root.get_descendant_or_error(
        new_path.get_child("keep_me"))

    # the promoted nested_child's parent index is changed.
    nested_child_node = expression_test_util.calculate_value_slowly(
        new_nested_child)
    self.assertAllEqual(nested_child_node.parent_index, [0, 1, 1, 1])
    self.assertTrue(nested_child_node.is_repeated)

    # bar's parent index should be unchanged.
    bar_node = expression_test_util.calculate_value_slowly(bar_expr)
    self.assertAllEqual(bar_node.parent_index, [0, 1, 1, 2])
    self.assertAllEqual(bar_node.values, [b"a", b"b", b"c", b"d"])
    self.assertTrue(bar_node.is_repeated)

    # keep_me's parent index should be unchanged.
    keep_me_node = expression_test_util.calculate_value_slowly(keep_me_expr)
    self.assertAllEqual(keep_me_node.parent_index, [0, 1])
    self.assertAllEqual(keep_me_node.values, [False, True])
    self.assertFalse(keep_me_node.is_repeated)

  def test_promote_and_calculate_substructure_then_leaf(self):
    """Tests promoting of substructure and then a leaf."""
    expr = create_expression.create_expression_from_prensor(
        prensor_test_util.create_four_layer_prensor())
    new_root, new_nested_child_path = promote.promote_anonymous(
        expr, path.Path(["event", "doc", "nested_child"]))
    new_root, new_bar_path = promote.promote_anonymous(
        new_root, new_nested_child_path.get_child("bar"))

    # the promoted nested_child's parent index is changed.
    new_nested_child = new_root.get_descendant_or_error(new_nested_child_path)
    nested_child_node = expression_test_util.calculate_value_slowly(
        new_nested_child)
    self.assertAllEqual(nested_child_node.parent_index, [0, 1, 1, 1])
    self.assertTrue(nested_child_node.is_repeated)

    # promoted bar's parent index is changed.
    new_bar = new_root.get_descendant_or_error(new_bar_path)
    bar_node = expression_test_util.calculate_value_slowly(new_bar)
    self.assertAllEqual(bar_node.parent_index, [0, 1, 1, 1])
    self.assertAllEqual(bar_node.values, [b"a", b"b", b"c", b"d"])
    self.assertTrue(bar_node.is_repeated)

    # bar's parent index should be unchanged.
    nested_child_bar = new_root.get_descendant_or_error(
        new_nested_child_path.get_child("bar"))
    nested_child_bar_node = expression_test_util.calculate_value_slowly(
        nested_child_bar)
    self.assertAllEqual(nested_child_bar_node.parent_index, [0, 1, 1, 2])
    self.assertAllEqual(nested_child_bar_node.values, [b"a", b"b", b"c", b"d"])
    self.assertTrue(nested_child_bar_node.is_repeated)

    # keep_me's parent index should be unchanged.
    nested_child_keep_me = new_root.get_descendant_or_error(
        new_nested_child_path.get_child("keep_me"))
    nested_child_keep_me_node = expression_test_util.calculate_value_slowly(
        nested_child_keep_me)
    self.assertAllEqual(nested_child_keep_me_node.parent_index, [0, 1])
    self.assertAllEqual(nested_child_keep_me_node.values, [False, True])
    self.assertFalse(nested_child_keep_me_node.is_repeated)

  def test_promote_and_calculate_leaf_then_substructure(self):
    """Tests promoting of leaf and then a substructure."""
    expr = create_expression.create_expression_from_prensor(
        prensor_test_util.create_four_layer_prensor())
    new_root, new_bar_path = promote.promote_anonymous(
        expr, path.Path(["event", "doc", "nested_child", "bar"]))
    new_root, new_path = promote.promote_anonymous(new_root,
                                                   path.Path(["event", "doc"]))

    new_doc = new_root.get_descendant_or_error(new_path)
    new_bar = new_root.get_descendant_or_error(
        new_path.concat(new_bar_path.suffix(2)))
    bar_expr = new_root.get_descendant_or_error(
        new_path.concat(path.Path(["nested_child", "bar"])))
    keep_me_expr = new_root.get_descendant_or_error(
        new_path.concat(path.Path(["nested_child", "keep_me"])))

    new_doc_node = expression_test_util.calculate_value_slowly(new_doc)
    self.assertAllEqual(new_doc_node.parent_index, [0, 1, 1])
    self.assertTrue(new_doc_node.is_repeated)

    # new_bar's parent index is changed (from the first promote).
    # The second promote should not change new_bar's parent index.
    new_bar_node = expression_test_util.calculate_value_slowly(new_bar)
    self.assertAllEqual(new_bar_node.parent_index, [0, 1, 1, 1])
    self.assertAllEqual(new_bar_node.values, [b"a", b"b", b"c", b"d"])
    self.assertTrue(new_bar_node.is_repeated)

    # bar's parent index should be unchanged.
    bar_node = expression_test_util.calculate_value_slowly(bar_expr)
    self.assertAllEqual(bar_node.parent_index, [0, 1, 1, 2])
    self.assertAllEqual(bar_node.values, [b"a", b"b", b"c", b"d"])
    self.assertTrue(bar_node.is_repeated)

    # keep_me's parent index should be unchanged.
    keep_me_node = expression_test_util.calculate_value_slowly(keep_me_expr)
    self.assertAllEqual(keep_me_node.parent_index, [0, 1])
    self.assertAllEqual(keep_me_node.values, [False, True])
    self.assertFalse(keep_me_node.is_repeated)

  def test_promote_substructure_with_schema(self):
    expr = create_expression.create_expression_from_prensor(
        prensor_test_util.create_deep_prensor()).apply_schema(
            prensor_test_util.create_deep_prensor_schema())

    original_schema = expr.get_descendant_or_error(path.Path(["event", "doc"
                                                             ])).schema_feature

    new_root, new_field_path = promote.promote_anonymous(
        expr, path.Path(["event", "doc"]))
    new_field = new_root.get_descendant_or_error(new_field_path)
    new_schema_feature = new_field.schema_feature
    self.assertIsNotNone(new_schema_feature)

    # The struct_domain of this feature should not be changed.
    self.assertProtoEquals(new_schema_feature.struct_domain,
                           original_schema.struct_domain)

    bar_schema = new_root.get_descendant_or_error(
        new_field_path.concat(path.Path(["bar"]))).schema_feature
    self.assertIsNotNone(bar_schema)
    self.assertEqual(bar_schema.string_domain.value[0], "a")

    keep_me_schema = new_root.get_descendant_or_error(
        new_field_path.concat(path.Path(["keep_me"]))).schema_feature
    self.assertIsNotNone(keep_me_schema)
    self.assertEqual(keep_me_schema.presence.min_count, 1)


if __name__ == "__main__":
  absltest.main()
