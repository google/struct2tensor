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
"""Tests for struct2tensor.parquet."""

import tensorflow.compat.v2 as tf
from absl.testing import absltest
from pyarrow.lib import ArrowIOError
from tensorflow.python.framework import (
  test_util,  # pylint: disable=g-direct-tensorflow-import
)

from struct2tensor import path, prensor
from struct2tensor.expression_impl import parquet, project, promote

tf.enable_v2_behavior()


@test_util.run_all_in_graph_and_eager_modes
class ParquetDatasetTestBase(tf.test.TestCase):

  def setUp(self):
    super(ParquetDatasetTestBase, self).setUp()
    self._test_filenames = [
        "struct2tensor/testdata/parquet_testdata/dremel_example.parquet"
    ]
    self._rowgroup_test_filenames = [
        "struct2tensor/testdata/parquet_testdata/dremel_example_two_row_groups.parquet"
    ]

  def _assertPrensorEqual(self, result, expected):
    """Traverses prensors level order, to check that the two prensors are equal.

    Args:
      result: the resulting prensor.
      expected: the expected prensor
    """
    res_node = result.node
    exp_node = expected.node

    self.assertEqual(type(res_node), type(exp_node))

    if result.is_leaf:
      self.assertAllEqual(res_node.parent_index, exp_node.parent_index)
      self.assertAllEqual(res_node.values, exp_node.values)
    else:
      if isinstance(res_node, prensor.RootNodeTensor):
        self.assertEqual(res_node.size, exp_node.size)
      else:
        self.assertAllEqual(res_node.parent_index, exp_node.parent_index)
      res_children = result.get_children()
      exp_children = expected.get_children()

      for child_step in res_children:
        self._assertPrensorEqual(res_children[child_step],
                                 exp_children[child_step])


class ParquetDatasetOutputsPrensorTest(ParquetDatasetTestBase):
  """This tests the public facing API, ParquetDataset()."""

  def testInvalidFile(self):
    """Tests exception is thrown for invalid file."""
    with self.assertRaisesRegex(ArrowIOError, "Failed to open"):
      parquet.ParquetDataset(
          filenames=["invalid"], value_paths=["DocId"], batch_size=1)

  def testInvalidColumnName(self):
    """Tests exception is thrown for invalid column name."""
    with self.assertRaisesRegex(ValueError, "path does not exist in the file."):
      parquet.ParquetDataset(
          filenames=self._test_filenames,
          value_paths=["invalid_path"],
          batch_size=1)

  def testPrensorOutput(self):
    """Tests that the dataset outputs a prensor."""
    pq_ds = parquet.ParquetDataset(
        self._test_filenames,
        value_paths=["Name.Language.Country", "DocId", "Name.Language.Code"],
        batch_size=1)

    for (i, pren) in enumerate(pq_ds):
      doc_id = pren.get_descendant_or_error(path.Path(["DocId"])).node
      self.assertAllEqual(doc_id.parent_index, [0])
      self.assertAllEqual(doc_id.values, [(i + 1) * 10])
      code = pren.get_descendant_or_error(
          path.Path(["Name", "Language", "Code"])).node
      if i == 0:
        self.assertAllEqual(code.parent_index, [0, 1, 2])
        self.assertAllEqual(code.values, [b"en-us", b"en", b"en-gb"])
      else:
        self.assertAllEqual(code.parent_index, [])
        self.assertAllEqual(code.values, [])

  def testMultipleColumnsTwoRowGroupsAndEqualBatchSize_OutputsPrensor(self):
    """Tests that the correct prensor for three columns is outputted."""
    pq_ds = parquet.ParquetDataset(
        filenames=self._rowgroup_test_filenames,
        value_paths=["DocId", "Name.Language.Code", "Name.Language.Country"],
        batch_size=2)
    expected_prensor = prensor.create_prensor_from_descendant_nodes({
        path.Path([]):
            prensor.RootNodeTensor(tf.constant(2, dtype=tf.int64)),
        path.Path(["DocId"]):
            prensor.LeafNodeTensor(
                tf.constant([0, 1], dtype=tf.int64),
                tf.constant([10, 20], dtype=tf.int64), True),
        path.Path(["Name"]):
            prensor.ChildNodeTensor(
                tf.constant([0, 0, 0, 1], dtype=tf.int64), True),
        path.Path(["Name", "Language"]):
            prensor.ChildNodeTensor(
                tf.constant([0, 0, 2], dtype=tf.int64), True),
        path.Path(["Name", "Language", "Code"]):
            prensor.LeafNodeTensor(
                tf.constant([0, 1, 2], dtype=tf.int64),
                tf.constant([b"en-us", b"en", b"en-gb"]), True),
        path.Path(["Name", "Language", "Country"]):
            prensor.LeafNodeTensor(
                tf.constant([0, 2], dtype=tf.int64), tf.constant([b"us",
                                                                  b"gb"]), True)
    })

    for i, pren in enumerate(pq_ds):
      if i == 0:
        self._assertPrensorEqual(pren, expected_prensor)


class ParquetDatasetWithExpressionTest(ParquetDatasetTestBase):
  """This tests the public facing API, using the placeholder expression."""

  def testPromoteAndProjectExpression(self):
    filenames = [
        "struct2tensor/testdata/parquet_testdata/dremel_example.parquet"
    ]
    batch_size = 2
    exp = parquet.create_expression_from_parquet_file(filenames)
    new_exp = promote.promote(exp, path.Path(["Name", "Language", "Code"]),
                              "new_code")
    new_code_project_exp = project.project(new_exp,
                                           [path.Path(["Name", "new_code"])])
    docid_project_exp = project.project(exp, [path.Path(["DocId"])])

    pqds = parquet.calculate_parquet_values(
        [new_code_project_exp, docid_project_exp], exp, filenames, batch_size)

    new_code_expected = prensor.create_prensor_from_descendant_nodes({
        path.Path([]):
            prensor.RootNodeTensor(tf.constant(2, dtype=tf.int64)),
        path.Path(["Name"]):
            prensor.ChildNodeTensor(
                tf.constant([0, 0, 0, 1], dtype=tf.int64), True),
        path.Path(["Name", "new_code"]):
            prensor.LeafNodeTensor(
                tf.constant([0, 0, 2], dtype=tf.int64),
                tf.constant([b"en-us", b"en", b"en-gb"]), True)
    })

    docid_expected = prensor.create_prensor_from_descendant_nodes({
        path.Path([]):
            prensor.RootNodeTensor(tf.constant(2, dtype=tf.int64)),
        path.Path(["DocId"]):
            prensor.LeafNodeTensor(
                tf.constant([0, 1], dtype=tf.int64),
                tf.constant([10, 20], dtype=tf.int64), False)
    })

    for ele in pqds:
      new_code_pren = ele[0]
      docid_pren = ele[1]

      self._assertPrensorEqual(new_code_pren, new_code_expected)
      self._assertPrensorEqual(docid_pren, docid_expected)


if __name__ == "__main__":
  absltest.main()
