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

from absl.testing import absltest
from struct2tensor import calculate
from struct2tensor import create_expression
from struct2tensor import path
from struct2tensor import prensor
# For tf.Session.Run against a Prensor
from struct2tensor import prensor_value  # pylint: disable=unused-import
from struct2tensor.expression_impl import filter_expression
from struct2tensor.expression_impl import proto_test_util
from struct2tensor.test import expression_test_util
from struct2tensor.test import prensor_test_util
from struct2tensor.test import test_pb2
import tensorflow as tf


from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


def _create_slice_and_project_example():
  r"""Creates an example for test_slice_and_project.

  Returns:

               ------*-----------------
              /                        \
        ---session0----             session1
       /      \        \             /      \
    event0  event1   event2      event3   event4
    /    \    |  \     |  \       /       /  |  \
  act0 act1 act2 act3 act4 act5 act6   act7 act8 act9
    |   |     |        |    |     |      |    |   |
    a   b     c        e    f     g      h    i   j

  This also adds an action_mask that is false for the zeroeth action.
  """
  return proto_test_util.text_to_expression([
      """
        event:{
          action_mask: [false, true]
          action:{
            doc_id:"a"
          }
          action:{
            doc_id:"b"
          }
        }
        event:{
          action_mask: [false, true]
          action:{
            doc_id:"c"
          }
          action:{
          }
        }
        event:{
          action_mask: [false, true]
          action:{
            doc_id:"e"
          }
          action:{
            doc_id:"f"
          }
        }""", """
        event:{
          action_mask: [false]
          action:{
            doc_id:"g"
          }
        }
        event:{
          action_mask: [false, true, true]
          action:{
            doc_id:"h"
          }
          action:{
            doc_id:"i"
          }
          action:{
            doc_id:"j"
          }
        }"""
  ], test_pb2.Session)


def _create_nested_prensor():
  r"""Creates a prensor representing a list of nested protocol buffers.

       -----*----------------------------------------------------
      /                       \                                  \
   root0                    root1-----------------------      root2 (empty)
    /   \                   /    \               \      \
    |  keep_my_sib0:False  |  keep_my_sib1:True   | keep_my_sib2:False
  doc0-----               doc1---------------    doc2--------
   |       \                \           \    \               \
  bar:"a"  keep_me:False    bar:"b" bar:"c" keep_me:True      bar:"d"

  Returns:
    a prensor expression representing:
    {doc:[{bar:["a"], keep_me:False}], keep_my_sib:False}
    {doc:[{bar:["b","c"], keeo}, {bar:["d"]}],
     keep_me:True}
    {}
  """
  return prensor.create_prensor_from_descendant_nodes({
      path.Path([]):
          prensor_test_util.create_root_node(3),
      path.Path(["doc"]):
          prensor_test_util.create_child_node([0, 1, 1], True),
      path.Path(["keep_my_sib"]):
          prensor_test_util.create_repeated_leaf_node([0, 1, 1],
                                                      [False, True, False]),
      path.Path(["doc", "bar"]):
          prensor_test_util.create_repeated_leaf_node([0, 1, 1, 2],
                                                      ["a", "b", "c", "d"]),
      path.Path(["doc", "keep_me"]):
          prensor_test_util.create_optional_leaf_node([0, 1], [False, True])
  })


def _create_nested_prensor_2():
  r"""Creates a prensor representing a list of nested protocol buffers.

  keep_me no longer has a value in doc0.

       -----*----------------------------------------------------
      /                       \                                  \
   root0                    root1-----------------------      root2 (empty)
    /   \                   /    \               \      \
    |  keep_my_sib0:False  |  keep_my_sib1:True   | keep_my_sib2:False
  doc0                    doc1---------------    doc2--------
   |                        \           \    \               \
  bar:"a"                   bar:"b" bar:"c" keep_me:True      bar:"d"

  Returns:
    a prensor expression representing:
    {doc:[{bar:["a"], keep_me:False}], keep_my_sib:False}
    {doc:[{bar:["b","c"], keeo}, {bar:["d"]}],
     keep_me:True}
    {}
  """
  return prensor.create_prensor_from_descendant_nodes({
      path.Path([]):
          prensor_test_util.create_root_node(3),
      path.Path(["doc"]):
          prensor_test_util.create_child_node([0, 1, 1], True),
      path.Path(["keep_my_sib"]):
          prensor_test_util.create_repeated_leaf_node([0, 1, 1],
                                                      [False, True, False]),
      path.Path(["doc", "bar"]):
          prensor_test_util.create_repeated_leaf_node([0, 1, 1, 2],
                                                      ["a", "b", "c", "d"]),
      path.Path(["doc", "keep_me"]):
          prensor_test_util.create_optional_leaf_node([1], [True])
  })


@test_util.run_all_in_graph_and_eager_modes
class FilterExpressionTest(tf.test.TestCase):

  def test_filter_by_child(self):
    """Tests filter_by_child."""
    root = create_expression.create_expression_from_prensor(
        prensor_test_util.create_big_prensor())
    root_2 = filter_expression.filter_by_child(root, path.create_path("doc"),
                                               "keep_me", "new_doc")
    [result] = calculate.calculate_prensors([root_2])
    self.assertAllEqual(
        result.get_descendant_or_error(path.Path(["new_doc"
                                                 ])).node.parent_index, [1])
    self.assertAllEqual(
        result.get_descendant_or_error(path.Path(["new_doc", "keep_me"
                                                 ])).node.parent_index, [0])
    self.assertAllEqual(
        result.get_descendant_or_error(path.Path(["new_doc",
                                                  "keep_me"])).node.values,
        [True])
    self.assertAllEqual(
        result.get_descendant_or_error(path.Path(["new_doc",
                                                  "bar"])).node.parent_index,
        [0, 0])
    self.assertAllEqual(
        result.get_descendant_or_error(path.Path(["new_doc",
                                                  "bar"])).node.values,
        [b"b", b"c"])

  def test_filter_by_child_create_nested_prensor(self):
    """Tests filter_by_child."""
    root = create_expression.create_expression_from_prensor(
        _create_nested_prensor())
    root_2 = filter_expression.filter_by_child(root, path.create_path("doc"),
                                               "keep_me", "new_doc")
    [result] = calculate.calculate_prensors([root_2])
    self.assertAllEqual(
        result.get_descendant_or_error(path.Path(["new_doc"
                                                 ])).node.parent_index, [1])
    self.assertAllEqual(
        result.get_descendant_or_error(path.Path(["new_doc", "keep_me"
                                                 ])).node.parent_index, [0])
    self.assertAllEqual(
        result.get_descendant_or_error(path.Path(["new_doc",
                                                  "keep_me"])).node.values,
        [True])
    self.assertAllEqual(
        result.get_descendant_or_error(path.Path(["new_doc",
                                                  "bar"])).node.parent_index,
        [0, 0])
    self.assertAllEqual(
        result.get_descendant_or_error(path.Path(["new_doc",
                                                  "bar"])).node.values,
        [b"b", b"c"])

  def test_filter_by_child_create_nested_prensor_2(self):
    """Tests filter_by_child.

    In particular, it checks for the case where parent_index != self index.
    """
    root = create_expression.create_expression_from_prensor(
        _create_nested_prensor_2())
    root_2 = filter_expression.filter_by_child(root, path.create_path("doc"),
                                               "keep_me", "new_doc")
    [result] = calculate.calculate_prensors([root_2])
    self.assertAllEqual(
        result.get_descendant_or_error(path.Path(["new_doc"
                                                 ])).node.parent_index, [1])
    self.assertAllEqual(
        result.get_descendant_or_error(path.Path(["new_doc", "keep_me"
                                                 ])).node.parent_index, [0])
    self.assertAllEqual(
        result.get_descendant_or_error(path.Path(["new_doc",
                                                  "keep_me"])).node.values,
        [True])
    self.assertAllEqual(
        result.get_descendant_or_error(path.Path(["new_doc",
                                                  "bar"])).node.parent_index,
        [0, 0])
    self.assertAllEqual(
        result.get_descendant_or_error(path.Path(["new_doc",
                                                  "bar"])).node.values,
        [b"b", b"c"])

  def test_filter_by_sibling(self):
    r"""Tests filter_by_sibling.

    Beginning with the struct:
         -----*----------------------------------------------------
        /                       \                                  \
     root0                    root1-----------------------      root2 (empty)
      /   \                   /    \               \      \
      |  keep_my_sib0:False  |  keep_my_sib1:True   | keep_my_sib2:False
    doc0-----               doc1---------------    doc2--------
     |       \                \           \    \               \
    bar:"a"  keep_me:False    bar:"b" bar:"c" keep_me:True      bar:"d"

    Filter doc with keep_my_sib:

    End with the struct (suppressing original doc):
         -----*----------------------------------------------------
        /                       \                                  \
    root0                    root1------------------        root2 (empty)
        \                   /    \                  \
        keep_my_sib0:False  |  keep_my_sib1:True   keep_my_sib2:False
                           new_doc0-----------
                             \           \    \
                             bar:"b" bar:"c" keep_me:True

    """
    root = create_expression.create_expression_from_prensor(
        _create_nested_prensor())
    root_2 = filter_expression.filter_by_sibling(root, path.create_path("doc"),
                                                 "keep_my_sib", "new_doc")
    [result] = calculate.calculate_prensors([root_2])
    self.assertAllEqual(
        result.get_descendant_or_error(path.Path(["new_doc"
                                                 ])).node.parent_index, [1])
    self.assertAllEqual(
        result.get_descendant_or_error(path.Path(["new_doc", "keep_me"
                                                 ])).node.parent_index, [0])
    self.assertAllEqual(
        result.get_descendant_or_error(path.Path(["new_doc",
                                                  "keep_me"])).node.values,
        [True])
    self.assertAllEqual(
        result.get_descendant_or_error(path.Path(["new_doc",
                                                  "bar"])).node.parent_index,
        [0, 0])
    self.assertAllEqual(
        result.get_descendant_or_error(path.Path(["new_doc",
                                                  "bar"])).node.values,
        [b"b", b"c"])

  def test_slice_and_project_mini(self):
    """Testing a part of query_test.test_slice_and_project.

    Originally, there was an error with query_test.test_slice_and_project,
    caused by filter_expression. I used this unit test to find and ultimately
    fix the error.
    """
    root = _create_slice_and_project_example()

    root_2 = filter_expression.filter_by_sibling(root,
                                                 path.Path(["event", "action"]),
                                                 "action_mask", "taction")
    calculate_value = expression_test_util.calculate_value_slowly(
        root_2.get_descendant_or_error(path.Path(["event", "taction"])))
    value_indices = calculate_value.parent_index
    self.assertAllEqual(value_indices, [0, 1, 2, 4, 4])

  def test_indices_where_true(self):
    input_prensor_node = prensor_test_util.create_repeated_leaf_node(
        [0, 0, 1, 1, 2, 2, 3, 4, 4, 4],
        [False, True, False, True, False, True, False, False, True, True])
    tensor_result = filter_expression._self_indices_where_true(
        input_prensor_node)
    self.assertAllEqual(tensor_result, [1, 3, 5, 8, 9])


if __name__ == "__main__":
  absltest.main()
