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

import tensorflow as tf
from absl.testing import absltest
from tensorflow.python.framework import (
  test_util,  # pylint: disable=g-direct-tensorflow-import
)

from struct2tensor.expression_impl import parquet


class ParquetDatasetTestBase(tf.test.TestCase):
  """Test base for testing _RawParquetDataset.

  This wraps tensorflow dataset's iterator and get_next, so that it can run both
  eagerly and in graph mode.
  """

  def setUp(self):
    super().setUp()
    self._test_filenames = [
        "struct2tensor/testdata/parquet_testdata/dremel_example.parquet"
    ]
    self._datatype_test_filenames = [
        "struct2tensor/testdata/parquet_testdata/all_data_types.parquet"
    ]
    self._rowgroup_test_filenames = [
        "struct2tensor/testdata/parquet_testdata/dremel_example_two_row_groups.parquet"
    ]

  def _assertTensorsEqual(self, result, expected, assert_items_equal):
    """Checks that two list of prensors (collection of tensors) are equal.

    Args:
      result: first list of prensors
      expected: second list of prensors
      assert_items_equal: determines if order of result and expected matters
    """
    if assert_items_equal:
      self.assertCountEqual(result, expected)
      return

    for pren1, pren2 in zip(result, expected):
      self.assertEqual(len(pren1), len(pren2))
      # we know the first ele is always scalar
      self.assertEqual(pren1[0], pren2[0])
      for (tensor1, tensor2) in zip(pren1, pren2):
        self.assertAllEqual(tensor1, tensor2)

  def _getNext(self, dataset, requires_initialization=False):
    """Returns a callable that returns the next element of the dataset.

    Example use:
    ```python
    # In both graph and eager modes
    dataset = ...
    get_next = self._getNext(dataset)
    result = self.evaluate(get_next())
    ```

    Args:
      dataset: A dataset whose elements will be returned.
      requires_initialization: Indicates that when the test is executed in graph
        mode, it should use an initializable iterator to iterate through the
        dataset (e.g. when it contains stateful nodes). Defaults to False.

    Returns:
      A callable that returns the next element of `dataset`. Any `TensorArray`
      objects `dataset` outputs are stacked.
    """

    def ta_wrapper(gn):

      def _wrapper():
        r = gn()
        if isinstance(r, tf.TensorArray):
          return r.stack()
        else:
          return r

      return _wrapper

    building_function = tf.compat.v1.get_default_graph().building_function
    if tf.executing_eagerly() or building_function:
      iterator = iter(dataset)
      return ta_wrapper(iterator._next_internal)
    else:
      if requires_initialization:
        iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
        self.evaluate(iterator.initializer)
      else:
        iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
      get_next = iterator.get_next()
      return ta_wrapper(lambda: get_next)

  def assertDatasetProduces(self,
                            dataset,
                            expected_output=None,
                            expected_shapes=None,
                            expected_error=None,
                            requires_initialization=False,
                            num_test_iterations=1,
                            assert_items_equal=False,
                            expected_error_iter=1):
    """Asserts that a dataset produces the expected output / error.

    Args:
      dataset: A dataset to check for the expected output / error.
      expected_output: A list of elements that the dataset is expected to
        produce.
      expected_shapes: A list of TensorShapes which is expected to match
        output_shapes of dataset.
      expected_error: A tuple `(type, message)` identifying the expected error
        `dataset` should raise. The `type` should match the expected exception
        type, while `message` should either be a regular expression that is
        expected to match the error message partially.
      requires_initialization: Indicates that when the test is executed in graph
        mode, it should use an initializable iterator to iterate through the
        dataset (e.g. when it contains stateful nodes). Defaults to False.
      num_test_iterations: Number of times `dataset` will be iterated. Defaults
        to 2.
      assert_items_equal: Tests expected_output has (only) the same elements
        regardless of order.
      expected_error_iter: How many times to iterate before expecting an error,
        if an error is expected.
    """
    self.assertTrue(
        expected_error is not None or expected_output is not None,
        "Exactly one of expected_output or expected error should be provided.")
    if expected_error:
      self.assertIsNone(
          expected_output,
          "Exactly one of expected_output or expected error should be provided."
      )
      with self.assertRaisesRegex(expected_error[0], expected_error[1]):
        get_next = self._getNext(
            dataset, requires_initialization=requires_initialization)
        for _ in range(expected_error_iter):
          self.evaluate(get_next())
      return
    if expected_shapes:
      self.assertEqual(expected_shapes, tf.data.get_output_shapes(dataset))
    self.assertGreater(num_test_iterations, 0)
    for _ in range(num_test_iterations):
      get_next = self._getNext(
          dataset, requires_initialization=requires_initialization)
      result = []
      for _ in range(len(expected_output)):
        res = self.evaluate(get_next())
        result.append(res)
      self._assertTensorsEqual(result, expected_output, assert_items_equal)
      with self.assertRaises(tf.errors.OutOfRangeError):
        self.evaluate(get_next())
      with self.assertRaises(tf.errors.OutOfRangeError):
        self.evaluate(get_next())


@test_util.run_all_in_graph_and_eager_modes
class ParquetDatasetForTestingOpTest(ParquetDatasetTestBase):
  """Tests for constructing parquet datasets and reading columns.

  Input file schema:
    message Document {
      required int64 DocId;
      optional group Links {
        repeated int64 Backward;
        repeated int64 Forward;
      }
      repeated group Name {
        repeated group Language {
          required binary Code (UTF8);
          optional binary Country (UTF8);
        }
        optional binary Url (UTF8);
      }
    }

  Input file contents:
    Document
      DocId: 10
      Links
        Forward: 20
        Forward: 40
        Forward: 60
      Name
        Language
          Code: 'en-us'
          Country: 'us'
        Language
          Code: 'en'
        Url: 'http://A'
      Name
        Url: 'http://B'
      Name
        Language
          Code: 'en-gb'
          Country: 'gb'
    Document
      DocId: 20
      Links
        Backward: 10
        Backward: 30
        Forward: 80
      Name
        Url: 'http://C'
  """

  def testInvalidParentIndexPaths(self):
    """Tests that wrong parent_index_paths order will throw an error."""
    with self.assertRaisesRegex(
        tf.errors.InvalidArgumentError,
        "parent_index_paths is not aligned with value_paths"):
      pq_ds = parquet._RawParquetDataset(
          filenames=self._test_filenames,
          value_paths=["DocId", "Name.Language.Code", "Name.Language.Country"],
          value_dtypes=(
              tf.int64,
              tf.string,
              tf.string,
          ),
          parent_index_paths=[
              "Name.Language.Code", "Name.Language.Code", "Name.Language.Code",
              "DocId", "Name.Language.Country"
          ],
          path_index=[0, 1, 2, 0, 2],
          batch_size=1)
      get_next = self._getNext(pq_ds, True)
      self.evaluate(get_next())

  def testCreate(self):
    """Tests the creation of a dataset with valid parameters."""
    parquet._RawParquetDataset(
        filenames=self._test_filenames,
        value_paths=["DocId"],
        value_dtypes=(tf.int64,),
        parent_index_paths=["DocId"],
        path_index=[0],
        batch_size=1)

  def testFirstParentIndexOnly(self):
    """Tests only reqruest one parent index.

    Even when the path contains more than one. i.e. Name.Language.Code should
    have 4 parent indices (including the root).
    """
    pq_ds = parquet._RawParquetDataset(
        filenames=self._test_filenames,
        value_paths=["Name.Language.Code"],
        value_dtypes=(tf.string,),
        parent_index_paths=["Name.Language.Code"],
        path_index=[0],
        batch_size=1)
    self.assertDatasetProduces(
        pq_ds,
        expected_output=[(1, [0, 0, 0], [b"en-us", b"en", b"en-gb"]),
                         (1, [0], [])])

  def testDatasetShuffle(self):
    """Tests that the dataset supports shuffling of the dataset."""
    pq_ds = parquet._RawParquetDataset(
        filenames=self._test_filenames,
        value_paths=["DocId"],
        value_dtypes=(tf.int64,),
        parent_index_paths=["DocId"],
        path_index=[0],
        batch_size=1)
    pq_ds = pq_ds.shuffle(10)
    self.assertDatasetProduces(
        pq_ds,
        expected_output=[(1, [0], [10]), (1, [0], [20])],
        assert_items_equal=True)

  def testBatchEvenlyDivisible_ReadsTwoMessages(self):
    """Tests batch size that evenly divides the total number of messages."""
    pq_ds = parquet._RawParquetDataset(
        filenames=self._test_filenames,
        value_paths=["DocId"],
        value_dtypes=(tf.int64,),
        parent_index_paths=["DocId"],
        path_index=[0],
        batch_size=2)
    self.assertDatasetProduces(pq_ds, expected_output=[(2, [0, 1], [10, 20])])

  def testBatchLargerThanTotal_ReadsTwoMessages(self):
    """Tests batch size that is larger than the total number of messages.

    Since the batch size is larger, the output would be less than batch size.
    """
    pq_ds = parquet._RawParquetDataset(
        filenames=self._test_filenames,
        value_paths=["DocId"],
        value_dtypes=(tf.int64,),
        parent_index_paths=["DocId"],
        path_index=[0],
        batch_size=5)
    self.assertDatasetProduces(pq_ds, expected_output=[(2, [0, 1], [10, 20])])

  def testBatchEvenlyDivisibleContainsNones_ReadsTwoMessages(self):
    """Tests batch size that evenly divides the total number of messages.

    And that the messages contains None values.
    """
    pq_ds = parquet._RawParquetDataset(
        filenames=self._test_filenames,
        value_paths=["Name.Language.Country"],
        value_dtypes=(tf.string,),
        parent_index_paths=[
            "Name.Language.Country", "Name.Language.Country",
            "Name.Language.Country"
        ],
        path_index=[0, 1, 2],
        batch_size=2)
    self.assertDatasetProduces(
        pq_ds,
        expected_output=[(2, [0, 0, 0, 1], [0, 0, 2], [0, 2], [b"us", b"gb"])])

  def testBatchLargerThanTotalContainsNones_ReadsTwoMessages(self):
    """Tests batch size that is larger than the total number of messages.

    And that the messages contains None values.
    """
    pq_ds = parquet._RawParquetDataset(
        filenames=self._test_filenames,
        value_paths=["Name.Language.Country"],
        value_dtypes=(tf.string,),
        parent_index_paths=[
            "Name.Language.Country", "Name.Language.Country",
            "Name.Language.Country"
        ],
        path_index=[0, 1, 2],
        batch_size=5)
    self.assertDatasetProduces(
        pq_ds,
        expected_output=[(2, [0, 0, 0, 1], [0, 0, 2], [0, 2], [b"us", b"gb"])])

  def testMultipleColumns(self):
    """Tests that the dataset supports multiple columns."""
    pq_ds = parquet._RawParquetDataset(
        filenames=self._test_filenames,
        value_paths=["DocId", "Name.Language.Code", "Name.Language.Country"],
        value_dtypes=(
            tf.int64,
            tf.string,
            tf.string,
        ),
        parent_index_paths=[
            "DocId", "Name.Language.Code", "Name.Language.Code",
            "Name.Language.Code", "Name.Language.Country"
        ],
        path_index=[0, 0, 1, 2, 2],
        batch_size=1)
    self.assertDatasetProduces(
        pq_ds,
        expected_output=[(1, [0], [10], [0, 0, 0], [0, 0, 2], [0, 1, 2],
                          [b"en-us", b"en", b"en-gb"], [0, 2], [b"us", b"gb"]),
                         (1, [0], [20], [0], [], [], [], [], [])])

  def testMultipleFiles(self):
    """Tests that the dataset supports multiple files."""
    pq_ds = parquet._RawParquetDataset(
        filenames=[self._test_filenames[0], self._rowgroup_test_filenames[0]],
        value_paths=["DocId"],
        value_dtypes=(tf.int64,),
        parent_index_paths=["DocId"],
        path_index=[0],
        batch_size=1)
    self.assertDatasetProduces(
        pq_ds,
        expected_output=[(1, [0], [10]), (1, [0], [20]), (1, [0], [10]),
                         (1, [0], [20]), (1, [0], [30]), (1, [0], [40])])

  def testMultipleFilesLargeBatchSize(self):
    """Tests that the dataset supports multiple files and large batch size."""
    # TODO(andylou) We don't want this behavior in the future.
    # We eventually want the dataset to grab batches across files.
    pq_ds = parquet._RawParquetDataset(
        filenames=[self._test_filenames[0], self._rowgroup_test_filenames[0]],
        value_paths=["DocId"],
        value_dtypes=(tf.int64,),
        parent_index_paths=["DocId"],
        path_index=[0],
        batch_size=5)
    self.assertDatasetProduces(
        pq_ds,
        expected_output=[(2, [0, 1], [10, 20]),
                         (4, [0, 1, 2, 3], [10, 20, 30, 40])])


@test_util.run_all_in_graph_and_eager_modes
class ParquetDatasetForTestingOpDataTypesTest(ParquetDatasetTestBase):
  """Tests for testing that all types are handled in tensorflow.

  Input file schema:
    message Document {
      required boolean bool;
      required int32 int32;
      required int64 int64;
      required float float;
      required double double;
      required binary byte_array (UTF8);
    }
  Input file contents:
    Document
      bool: false
      int32: 10
      int64: 20
      float: 30.0
      double: 40.0
      byte_array: "fifty"
  """

  def testGetNext_HandlesBool(self):
    """Tests that bool is translated from parquet file to tensor."""
    pq_ds = parquet._RawParquetDataset(
        filenames=self._datatype_test_filenames,
        value_paths=["bool"],
        value_dtypes=(tf.bool,),
        parent_index_paths=["bool"],
        path_index=[0],
        batch_size=1)
    self.assertDatasetProduces(pq_ds, expected_output=[(1, [0], [False])])

  def testGetNext_HandlesInt32(self):
    """Tests that int32 is translated from parquet file to tensor."""
    pq_ds = parquet._RawParquetDataset(
        filenames=self._datatype_test_filenames,
        value_paths=["int32"],
        value_dtypes=(tf.int32,),
        parent_index_paths=["int32"],
        path_index=[0],
        batch_size=1)
    self.assertDatasetProduces(pq_ds, expected_output=[(1, [0], [10])])

  def testGetNext_HandlesInt64(self):
    """Tests that int64 is translated from parquet file to tensor."""
    pq_ds = parquet._RawParquetDataset(
        filenames=self._datatype_test_filenames,
        value_paths=["int64"],
        value_dtypes=(tf.int64,),
        parent_index_paths=["int64"],
        path_index=[0],
        batch_size=1)
    self.assertDatasetProduces(pq_ds, expected_output=[(1, [0], [20])])

  def testGetNext_HandlesFloat(self):
    """Tests that float is translated from parquet file to tensor."""
    pq_ds = parquet._RawParquetDataset(
        filenames=self._datatype_test_filenames,
        value_paths=["float"],
        value_dtypes=(tf.float32,),
        parent_index_paths=["float"],
        path_index=[0],
        batch_size=1)
    self.assertDatasetProduces(pq_ds, expected_output=[(1, [0], [30.0])])

  def testGetNext_HandlesDouble(self):
    """Tests that double is translated from parquet file to tensor."""
    pq_ds = parquet._RawParquetDataset(
        filenames=self._datatype_test_filenames,
        value_paths=["double"],
        value_dtypes=(tf.double,),
        parent_index_paths=["double"],
        path_index=[0],
        batch_size=1)
    self.assertDatasetProduces(pq_ds, expected_output=[(1, [0], [40.0])])

  def testGetNext_HandlesString(self):
    """Tests that string is translated from parquet file to tensor."""
    pq_ds = parquet._RawParquetDataset(
        filenames=self._datatype_test_filenames,
        value_paths=["byte_array"],
        value_dtypes=(tf.string,),
        parent_index_paths=["byte_array"],
        path_index=[0],
        batch_size=1)
    self.assertDatasetProduces(pq_ds, expected_output=[(1, [0], [b"fifty"])])


@test_util.run_all_in_graph_and_eager_modes
class ParquetDatasetForTestingOpRowGroupTest(ParquetDatasetTestBase):
  """Tests that multiple row groups are handled.

  This file is the dremel_example.parquet with 2x of the same values
  Input file schema:
    message Document {
      required int64 DocId;
      optional group Links {
        repeated int64 Backward;
        repeated int64 Forward;
      }
      repeated group Name {
        repeated group Language {
          required binary Code (UTF8);
          optional binary Country (UTF8);
        }
        optional binary Url (UTF8);
      }
    }
  Input file contents:
  RowGroup1:
    Document
      DocId: 10
      Links
        Forward: 20
        Forward: 40
        Forward: 60
      Name
        Language
          Code: 'en-us'
          Country: 'us'
        Language
          Code: 'en'
        Url: 'http://A'
      Name
        Url: 'http://B'
      Name
        Language
          Code: 'en-gb'
          Country: 'gb'
    Document
      DocId: 20
      Links
        Backward: 10
        Backward: 30
        Forward: 80
      Name
        Url: 'http://C'
  RowGroup2:
    Document
      DocId: 30
      Links
        Forward: 200
        Forward: 400
        Forward: 600
      Name
        Language
          Code: 'en-us2'
          Country: 'us2'
        Language
          Code: 'en2'
        Url: 'http://A2'
      Name
        Url: 'http://B2'
      Name
        Language
          Code: 'en-gb2'
          Country: 'gb2'
    Document
      DocId: 40
      Links
        Backward: 100
        Backward: 300
        Forward: 800
      Name
        Url: 'http://C2'
  """

  def testTwoRowGroupsAndDefaultBatchSize(self):
    """Tests default batch size with two row groups.

    Input:
    Rowgroup0:
      Document
        DocId: 10
      Document
        DocId: 20
    RowGroup1:
      Document
        DocId: 30
      Document
        DocId: 40
    """
    pq_ds = parquet._RawParquetDataset(
        filenames=self._rowgroup_test_filenames,
        value_paths=["DocId"],
        value_dtypes=(tf.int64,),
        parent_index_paths=["DocId"],
        path_index=[0],
        batch_size=1)
    self.assertDatasetProduces(
        pq_ds,
        expected_output=[(1, [0], [10]), (1, [0], [20]), (1, [0], [30]),
                         (1, [0], [40])])

  def testTwoRowGroupsAndDefaultBatchSizeContainsNones(self):
    """Tests default batch size with two row groups with None values.

    Input:
    RowGroup0:
      Document
        Name
          Language
            Code: 'en-us'
          Language
            Code: 'en'
        Name
        Name
          Language
            Code: 'en-gb'
      Document
        Name
    RowGroup1:
      Document
        Name
          Language
            Code: 'en-us2'
          Language
            Code: 'en2'
        Name
        Name
          Language
            Code: 'en-gb2'
      Document
        Name
    """
    pq_ds = parquet._RawParquetDataset(
        filenames=self._rowgroup_test_filenames,
        value_paths=["Name.Language.Code"],
        value_dtypes=(tf.string,),
        parent_index_paths=[
            "Name.Language.Code", "Name.Language.Code", "Name.Language.Code"
        ],
        path_index=[0, 1, 2],
        batch_size=1)
    self.assertDatasetProduces(
        pq_ds,
        expected_output=[
            (1, [0, 0, 0], [0, 0, 2], [0, 1, 2], [b"en-us", b"en", b"en-gb"]),
            (1, [0], [], [], []),
            (1, [0, 0, 0], [0, 0, 2], [0, 1, 2], [b"en-us2", b"en2",
                                                  b"en-gb2"]),
            (1, [0], [], [], [])
        ])

  def testTwoRowGroupsAndEqualBatchSize(self):
    """Tests batch size == row group size, with two row groups.

    Input:
    Rowgroup0:
      Document
        DocId: 10
      Document
        DocId: 20
    RowGroup1:
      Document
        DocId: 30
      Document
        DocId: 40
    """
    pq_ds = parquet._RawParquetDataset(
        filenames=self._rowgroup_test_filenames,
        value_paths=["DocId"],
        value_dtypes=(tf.int64,),
        parent_index_paths=["DocId"],
        path_index=[0],
        batch_size=2)
    self.assertDatasetProduces(
        pq_ds, expected_output=[(2, [0, 1], [10, 20]), (2, [0, 1], [30, 40])])

  def testTwoRowGroupsAndEqualBatchSizeContainsNones(self):
    """Tests batch size == row group size, with two row groups with None values.

    Input:
    RowGroup0:
      Document
        Name
          Language
            Code: 'en-us'
          Language
            Code: 'en'
        Name
        Name
          Language
            Code: 'en-gb'
      Document
        Name
    RowGroup1:
      Document
        Name
          Language
            Code: 'en-us2'
          Language
            Code: 'en2'
        Name
        Name
          Language
            Code: 'en-gb2'
      Document
        Name
    """
    pq_ds = parquet._RawParquetDataset(
        filenames=self._rowgroup_test_filenames,
        value_paths=["Name.Language.Code"],
        value_dtypes=(tf.string,),
        parent_index_paths=[
            "Name.Language.Code", "Name.Language.Code", "Name.Language.Code"
        ],
        path_index=[0, 1, 2],
        batch_size=2)
    self.assertDatasetProduces(
        pq_ds,
        expected_output=[(2, [0, 0, 0,
                              1], [0, 0, 2], [0, 1,
                                              2], [b"en-us", b"en", b"en-gb"]),
                         (2, [0, 0, 0,
                              1], [0, 0,
                                   2], [0, 1,
                                        2], [b"en-us2", b"en2", b"en-gb2"])])

  def testTwoRowGroupsAndLargerBatchSize(self):
    """Tests batch size > row group size, with two row groups.

    Input:
    Rowgroup0:
      Document
        DocId: 10
      Document
        DocId: 20
    RowGroup1:
      Document
        DocId: 30
      Document
        DocId: 40
    """
    pq_ds = parquet._RawParquetDataset(
        filenames=self._rowgroup_test_filenames,
        value_paths=["DocId"],
        value_dtypes=(tf.int64,),
        parent_index_paths=["DocId"],
        path_index=[0],
        batch_size=3)
    self.assertDatasetProduces(
        pq_ds, expected_output=[(3, [0, 1, 2], [10, 20, 30]), (1, [0], [40])])

  def testTwoRowGroupsAndLargerBatchSizeContainsNones(self):
    """Tests batch size > row group size, with two row groups with None values.

    Input:
    RowGroup0:
      Document
        Name
          Language
            Code: 'en-us'
          Language
            Code: 'en'
        Name
        Name
          Language
            Code: 'en-gb'
      Document
        Name
    RowGroup1:
      Document
        Name
          Language
            Code: 'en-us2'
          Language
            Code: 'en2'
        Name
        Name
          Language
            Code: 'en-gb2'
      Document
        Name
    """
    pq_ds = parquet._RawParquetDataset(
        filenames=self._rowgroup_test_filenames,
        value_paths=["Name.Language.Code"],
        value_dtypes=(tf.string,),
        parent_index_paths=[
            "Name.Language.Code", "Name.Language.Code", "Name.Language.Code"
        ],
        path_index=[0, 1, 2],
        batch_size=3)
    self.assertDatasetProduces(
        pq_ds,
        expected_output=[
            (3, [0, 0, 0, 1, 2, 2, 2], [0, 0, 2, 4, 4, 6], [0, 1, 2, 3, 4, 5],
             [b"en-us", b"en", b"en-gb", b"en-us2", b"en2", b"en-gb2"]),
            (1, [0], [], [], [])
        ])

  def testTwoRowGroupsAndDefaultBatchSizeFirstValueIsNone(self):
    """Tests batch size < row group size, where the first value is None.

    This tests that the buffer is still used properly, even if ther value
    cached is None.
    Input:
    Rowgroup0:
      Document
        Links
      Document
        Links
          Backward: 10
          Backward: 30
    RowGroup1:
      Document
        Links
      Document
        Links
          Backward: 100
          Backward: 300
    """
    pq_ds = parquet._RawParquetDataset(
        filenames=self._rowgroup_test_filenames,
        value_paths=["Links.Backward"],
        value_dtypes=(tf.int64,),
        parent_index_paths=["Links.Backward", "Links.Backward"],
        path_index=[0, 1],
        batch_size=1)
    self.assertDatasetProduces(
        pq_ds,
        expected_output=[(1, [0], [], []), (1, [0], [0, 0], [10, 30]),
                         (1, [0], [], []), (1, [0], [0, 0], [100, 300])])

  def testTwoRowGroupsAndEqualBatchSizeFirstValueIsNone(self):
    """Tests batch size == row group size, where the first value is None.

    This tests that the buffer is still used properly, even if the value
    cached is None.
    Input:
    Rowgroup0:
      Document
        Links
      Document
        Links
          Backward: 10
          Backward: 30
    RowGroup1:
      Document
        Links
      Document
        Links
          Backward: 100
          Backward: 300
    """
    pq_ds = parquet._RawParquetDataset(
        filenames=self._rowgroup_test_filenames,
        value_paths=["Links.Backward"],
        value_dtypes=(tf.int64,),
        parent_index_paths=["Links.Backward", "Links.Backward"],
        path_index=[0, 1],
        batch_size=2)
    self.assertDatasetProduces(
        pq_ds,
        expected_output=[(2, [0, 1], [1, 1], [10, 30]),
                         (2, [0, 1], [1, 1], [100, 300])])

  def testTwoRowGroupsAndLargerBatchSizeFirstValueIsNone(self):
    """Tests batch size > row group size, where the first value is None.

    This tests that the buffer is still used properly, even if the value
    cached is None.
    Input:
    Rowgroup0:
      Document
        Links
      Document
        Links
          Backward: 10
          Backward: 30
    RowGroup1:
      Document
        Links
      Document
        Links
          Backward: 100
          Backward: 300
    """
    pq_ds = parquet._RawParquetDataset(
        filenames=self._rowgroup_test_filenames,
        value_paths=["Links.Backward"],
        value_dtypes=(tf.int64,),
        parent_index_paths=["Links.Backward", "Links.Backward"],
        path_index=[0, 1],
        batch_size=3)
    self.assertDatasetProduces(
        pq_ds,
        expected_output=[(3, [0, 1, 2], [1, 1], [10, 30]),
                         (1, [0], [0, 0], [100, 300])])

  def testTwoRowGroupsAndDefaultBatchSizeLargeFirstMessage(self):
    """Tests batch size < row group size, where the first message is large.

    Input:
    Rowgroup0:
      Document
        Links
          Forward: 20
          Forward: 40
          Forward: 60
      Document
        Links
          Forward: 80
    RowGroup1:
      Document
        Links
          Forward: 200
          Forward: 400
          Forward: 600
      Document
        Links
          Forward: 800
    """
    pq_ds = parquet._RawParquetDataset(
        filenames=self._rowgroup_test_filenames,
        value_paths=["Links.Forward"],
        value_dtypes=(tf.int64,),
        parent_index_paths=["Links.Forward", "Links.Forward"],
        path_index=[0, 1],
        batch_size=1)
    self.assertDatasetProduces(
        pq_ds,
        expected_output=[(1, [0], [0, 0, 0], [20, 40, 60]), (1, [0], [0], [80]),
                         (1, [0], [0, 0, 0], [200, 400, 600]),
                         (1, [0], [0], [800])])

  def testTwoRowGroupsAndEqualBatchSizeLargeFirstMessage(self):
    """Tests batch size == row group size, where the first message is large.

    Input:
    Rowgroup0:
      Document
        Links
          Forward: 20
          Forward: 40
          Forward: 60
      Document
        Links
          Forward: 80
    RowGroup1:
      Document
        Links
          Forward: 200
          Forward: 400
          Forward: 600
      Document
        Links
          Forward: 800
    """
    pq_ds = parquet._RawParquetDataset(
        filenames=self._rowgroup_test_filenames,
        value_paths=["Links.Forward"],
        value_dtypes=(tf.int64,),
        parent_index_paths=["Links.Forward", "Links.Forward"],
        path_index=[0, 1],
        batch_size=2)
    self.assertDatasetProduces(
        pq_ds,
        expected_output=[(2, [0, 1], [0, 0, 0, 1], [20, 40, 60, 80]),
                         (2, [0, 1], [0, 0, 0, 1], [200, 400, 600, 800])])

  def testTwoRowGroupsAndLargerBatchSizeLargeFirstMessage(self):
    """Tests batch size > row group size, where the first message is large.

    Input:
    Rowgroup0:
      Document
        Links
          Forward: 20
          Forward: 40
          Forward: 60
      Document
        Links
          Forward: 80
    RowGroup1:
      Document
        Links
          Forward: 200
          Forward: 400
          Forward: 600
      Document
        Links
          Forward: 800
    """
    pq_ds = parquet._RawParquetDataset(
        filenames=self._rowgroup_test_filenames,
        value_paths=["Links.Forward"],
        value_dtypes=(tf.int64,),
        parent_index_paths=["Links.Forward", "Links.Forward"],
        path_index=[0, 1],
        batch_size=3)
    self.assertDatasetProduces(
        pq_ds,
        expected_output=[(3, [0, 1, 2], [0, 0, 0, 1, 2, 2,
                                         2], [20, 40, 60, 80, 200, 400, 600]),
                         (1, [0], [0], [800])])

  def testMultipleColumns_TwoRowGroupsAndEqualBatchSize(self):
    """Tests that the dataset supports multiple columns."""
    pq_ds = parquet._RawParquetDataset(
        filenames=self._rowgroup_test_filenames,
        value_paths=["DocId", "Name.Language.Code", "Name.Language.Country"],
        value_dtypes=(
            tf.int64,
            tf.string,
            tf.string,
        ),
        parent_index_paths=[
            "DocId", "Name.Language.Code", "Name.Language.Code",
            "Name.Language.Code", "Name.Language.Country"
        ],
        path_index=[0, 0, 1, 2, 2],
        batch_size=2)
    self.assertDatasetProduces(
        pq_ds,
        expected_output=[(2, [0, 1], [10, 20], [0, 0, 0, 1], [0, 0, 2],
                          [0, 1, 2], [b"en-us", b"en",
                                      b"en-gb"], [0, 2], [b"us", b"gb"]),
                         (2, [0, 1], [30, 40], [0, 0, 0, 1], [0, 0, 2],
                          [0, 1, 2], [b"en-us2", b"en2",
                                      b"en-gb2"], [0, 2], [b"us2", b"gb2"])])


if __name__ == "__main__":
  absltest.main()
