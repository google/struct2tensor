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
"""Tests for struct2tensor.expression_impl.placeholder."""

from absl.testing import absltest
from struct2tensor import calculate
from struct2tensor import path
from struct2tensor import prensor
from struct2tensor.expression_impl import map_prensor_to_prensor as mpp
from struct2tensor.expression_impl import placeholder
from struct2tensor.expression_impl import project
from struct2tensor.expression_impl import promote
from struct2tensor.test import prensor_test_util
import tensorflow as tf

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


@test_util.run_all_in_graph_and_eager_modes
class PlaceholderTest(tf.test.TestCase):

  def testPlaceholderExpression(self):
    pren = prensor_test_util.create_nested_prensor()
    expected_pren = prensor.create_prensor_from_descendant_nodes({
        path.Path([]):
            prensor.RootNodeTensor(tf.constant(3, dtype=tf.int64)),
        path.Path(["new_friends"]):
            prensor.LeafNodeTensor(
                tf.constant([0, 1, 1, 1, 2], dtype=tf.int64),
                tf.constant(["a", "b", "c", "d", "e"], dtype=tf.string), True)
    })

    root_schema = mpp.create_schema(
        is_repeated=True,
        children={
            "doc": {
                "is_repeated": True,
                "children": {
                    "bar": {
                        "is_repeated": True,
                        "dtype": tf.string
                    },
                    "keep_me": {
                        "is_repeated": False,
                        "dtype": tf.bool
                    }
                }
            },
            "user": {
                "is_repeated": True,
                "children": {
                    "friends": {
                        "is_repeated": True,
                        "dtype": tf.string
                    }
                }
            }
        })

    exp = placeholder.create_expression_from_schema(root_schema)
    promote_exp = promote.promote(exp, path.Path(["user", "friends"]),
                                  "new_friends")
    project_exp = project.project(promote_exp, [path.Path(["new_friends"])])
    new_friends_exp = project_exp.get_descendant(path.Path(["new_friends"]))

    result = calculate.calculate_values([new_friends_exp],
                                        feed_dict={exp: pren})

    res_node = result[0]
    exp_node = expected_pren.get_descendant(path.Path(["new_friends"])).node

    self.assertAllEqual(res_node.is_repeated, exp_node.is_repeated)
    self.assertAllEqual(res_node.values, exp_node.values)
    self.assertAllEqual(res_node.parent_index, exp_node.parent_index)

  def testCreateExpressionFromSchema(self):
    root_schema = mpp.create_schema(is_repeated=True, children={})
    exp = placeholder.create_expression_from_schema(root_schema)
    pren = prensor.create_prensor_from_descendant_nodes(
        {path.Path([]): prensor.RootNodeTensor(tf.constant(1, dtype=tf.int64))})
    result = calculate.calculate_values([exp], feed_dict={exp: pren})
    res_node = result[0]
    exp_node = pren.get_descendant(path.Path([])).node

    self.assertAllEqual(res_node.is_repeated, exp_node.is_repeated)
    self.assertAllEqual(res_node.size, exp_node.size)

  def testPlaceholderRootExpressionRequiresSideInfo(self):
    root_schema = mpp.create_schema(is_repeated=True, children={})
    exp = placeholder.create_expression_from_schema(root_schema)
    with self.assertRaisesRegex(
        ValueError, "_PlaceholderRootExpression requires side_info"):
      calculate.calculate_values([exp], feed_dict={exp: None})


if __name__ == "__main__":
  absltest.main()
