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
"""Tests for struct2tensor.expression_impl.map_prensor_to_prensor."""

from absl.testing import absltest
from struct2tensor import calculate
from struct2tensor import create_expression
from struct2tensor import path
from struct2tensor import prensor
# For tf.Session.Run against a Prensor
from struct2tensor import prensor_value  # pylint: disable=unused-import
from struct2tensor.expression_impl import map_prensor_to_prensor
from struct2tensor.test import prensor_test_util
import tensorflow as tf

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import
from tensorflow_metadata.proto.v0 import schema_pb2


@test_util.run_all_in_graph_and_eager_modes
class MapPrensorToPrensorTest(tf.test.TestCase):

  def test_map_prensor_to_prensor(self):
    original = create_expression.create_expression_from_prensor(
        prensor_test_util.create_nested_prensor())

    def my_prensor_op(original_prensor):
      # Note that we are copying over the original root prensor node. The root
      # node is ignored in the result.
      return prensor.create_prensor_from_descendant_nodes({
          path.Path([]):
              original_prensor.node,
          path.Path(["bar2"]):
              original_prensor.get_child_or_error("bar").node,
          path.Path(["keep_me2"]):
              original_prensor.get_child_or_error("keep_me").node
      })

    # Since the top node is actually a child node, we use the child schema.
    my_output_schema = map_prensor_to_prensor.create_schema(
        is_repeated=True,
        children={
            "bar2": {
                "is_repeated": True,
                "dtype": tf.string
            },
            "keep_me2": {
                "is_repeated": False,
                "dtype": tf.bool
            }
        })

    result = map_prensor_to_prensor.map_prensor_to_prensor(
        root_expr=original,
        source=path.Path(["doc"]),
        paths_needed=[path.Path(["bar"]),
                      path.Path(["keep_me"])],
        prensor_op=my_prensor_op,
        output_schema=my_output_schema)

    doc_result = result.get_child_or_error("doc")
    bar_result = doc_result.get_child_or_error("bar")
    keep_me_result = doc_result.get_child_or_error("keep_me")
    bar2_result = doc_result.get_child_or_error("bar2")
    keep_me2_result = doc_result.get_child_or_error("keep_me2")
    self.assertIsNone(doc_result.get_child("missing_field"))
    self.assertTrue(bar_result.is_repeated)
    self.assertTrue(bar2_result.is_repeated)
    self.assertEqual(bar_result.type, tf.string)
    self.assertEqual(bar2_result.type, tf.string)
    self.assertFalse(keep_me_result.is_repeated)
    self.assertFalse(keep_me2_result.is_repeated)
    self.assertEqual(keep_me_result.type, tf.bool)
    self.assertEqual(keep_me2_result.type, tf.bool)

    [prensor_result] = calculate.calculate_prensors([result])

    doc_value = prensor_result.get_child_or_error("doc")
    self.assertAllEqual([0, 1, 1], doc_value.node.parent_index)
    bar2_value = doc_value.get_child_or_error("bar2")
    self.assertAllEqual([0, 1, 1, 2], bar2_value.node.parent_index)
    self.assertAllEqual([b"a", b"b", b"c", b"d"], bar2_value.node.values)
    keep_me2_value = doc_value.get_child_or_error("keep_me2")
    self.assertAllEqual([0, 1], keep_me2_value.node.parent_index)
    self.assertAllEqual([False, True], keep_me2_value.node.values)

  def test_map_prensor_to_prensor_with_schema(self):
    original = create_expression.create_expression_from_prensor(
        prensor_test_util.create_nested_prensor())

    def my_prensor_op(original_prensor):
      # Note that we are copying over the original root prensor node. The root
      # node is ignored in the result.
      return prensor.create_prensor_from_descendant_nodes({
          path.Path([]):
              original_prensor.node,
          path.Path(["bar2"]):
              original_prensor.get_child_or_error("bar").node,
          path.Path(["keep_me2"]):
              original_prensor.get_child_or_error("keep_me").node
      })

    bar2_feature = schema_pb2.Feature()
    bar2_feature.value_count.max = 7
    keep_me2_feature = schema_pb2.Feature()
    keep_me2_feature.value_count.max = 10

    # Since the top node is actually a child node, we use the child schema.
    my_output_schema = map_prensor_to_prensor.create_schema(
        is_repeated=True,
        children={
            "bar2": {
                "is_repeated": True,
                "dtype": tf.string,
                "schema_feature": bar2_feature
            },
            "keep_me2": {
                "is_repeated": False,
                "dtype": tf.bool,
                "schema_feature": keep_me2_feature
            }
        })

    result = map_prensor_to_prensor.map_prensor_to_prensor(
        root_expr=original,
        source=path.Path(["doc"]),
        paths_needed=[path.Path(["bar"]),
                      path.Path(["keep_me"])],
        prensor_op=my_prensor_op,
        output_schema=my_output_schema)

    doc_result = result.get_child_or_error("doc")
    bar2_result = doc_result.get_child_or_error("bar2")
    self.assertEqual(bar2_result.schema_feature.value_count.max, 7)

    keep_me2_result = doc_result.get_child_or_error("keep_me2")
    self.assertEqual(keep_me2_result.schema_feature.value_count.max, 10)


if __name__ == "__main__":
  absltest.main()
