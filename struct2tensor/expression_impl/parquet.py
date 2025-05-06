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
"""Apache Parquet Dataset.

Example usage:

```
  exp = create_expression_from_parquet_file(filenames)
  docid_project_exp = project.project(exp, [path.Path(["DocId"])])
  pqds = parquet_dataset.calculate_parquet_values([docid_project_exp], exp,
                                                  filenames, batch_size)

  for prensors in pqds:
    doc_id_prensor = prensors[0]
```

"""

import collections
from typing import Any, Dict, List, Optional, Tuple, Union

import pyarrow as pa
import pyarrow.parquet as pq
from struct2tensor import calculate
from struct2tensor import calculate_options
from struct2tensor import expression
from struct2tensor import path
from struct2tensor import prensor
from struct2tensor.expression_impl import map_prensor_to_prensor as mpp
from struct2tensor.expression_impl import placeholder
from struct2tensor.ops import gen_parquet_dataset
import tensorflow as tf


def create_expression_from_parquet_file(
    filenames: List[str]) -> placeholder._PlaceholderRootExpression:  # pylint: disable=protected-access
  """Creates a placeholder expression from a parquet file.

  Args:
    filenames: A list of parquet files.

  Returns:
    A PlaceholderRootExpression that should be used as the root of an expression
    graph.
  """

  metadata = pq.ParquetFile(filenames[0]).metadata
  parquet_schema = metadata.schema
  arrow_schema = parquet_schema.to_arrow_schema()

  root_schema = mpp.create_schema(
      is_repeated=True,
      children=_create_children_from_arrow_fields(
          [arrow_schema.field_by_name(name) for name in arrow_schema.names]))

  # pylint: disable=protected-access
  return placeholder._PlaceholderRootExpression(root_schema)


def calculate_parquet_values(
    expressions: List[expression.Expression],
    root_exp: placeholder._PlaceholderRootExpression,  # pylint: disable=protected-access
    filenames: List[str],
    batch_size: int,
    options: Optional[calculate_options.Options] = None):
  """Calculates expressions and returns a parquet dataset.

  Args:
    expressions: A list of expressions to calculate.
    root_exp: The root placeholder expression to use as the feed dict.
    filenames: A list of parquet files.
    batch_size: The number of messages to batch.
    options: calculate options.

  Returns:
    A parquet dataset.
  """
  pqds = _ParquetDatasetWithExpression(expressions, root_exp, filenames,
                                       batch_size, options)
  return pqds.map(pqds._calculate_prensor)  # pylint: disable=protected-access


class _RawParquetDataset(tf.compat.v1.data.Dataset):
  """A dataset which reads columns from parquet and outputs a vector of tensors.

  A ParquetDataset is a Dataset of batches of messages (records).
  Every leaf field field of the messages in each batch has its own values tensor
  and parent indices tensors (which encodes the structural information).

  The user has control over which parent indices of which fields in a path to
  read, and is determined by parent_index_paths and path_index.

  View //struct2tensor/ops/parquet_dataset_op.cc
  for a better understanding of what format the vector of tensors is in.
  """

  def __init__(self, filenames: List[str], value_paths: List[str],
               value_dtypes: List[tf.DType], parent_index_paths: List[str],
               path_index: List[int], batch_size: int):
    """Creates a ParquetDataset.

    Args:
      filenames: A list containing the name(s) of the file(s) to be read.
      value_paths: A list of strings of the dotstring path(s) of each leaf
        path(s).
      value_dtypes: value_dtypes[i] is the Tensorflow data type value_paths[i]
        would be of.
      parent_index_paths: A list of strings of the dotstring path(s) of the
        path(s) to be read.
      path_index: A list containing the index of each field to get the parent
        index of. This will have the same length as parent_index_paths.
      batch_size: An int that determines how many messages are parsed into one
        prensor tree in an iteration. If there are fewer than batch_size
        remaining messages, then all remaining messages will be returned.

    Raises:
      ValueError: if the column does not exist in the parquet schema.
      ValueError: if the column dtype does not match the value_dtype passed in.
    """
    self._filenames = filenames
    self._value_paths = value_paths
    self._value_dtypes = tuple(value_dtypes)
    self._parent_index_paths = parent_index_paths
    self._path_index = path_index
    self._batch_size = batch_size

    super().__init__()

  def _get_column_path_to_index_mapping(self, metadata_file) -> Dict[str, int]:
    """Gets the column index of every column.

    Args:
      metadata_file: the file to be used as the metadata. If there is no
        metadata_file, any file from file_names will suffice.

    Returns:
      A dictionary mapping path name (str) to column index (int).
    """
    metadata = pq.ParquetFile(metadata_file).metadata

    path_to_column_index = {
        metadata.schema.column(index).path: index
        for index in range(metadata.num_columns)
    }

    return path_to_column_index

  def _parquet_to_tf_type(self, parquet_type: str) -> Union[tf.DType, None]:
    """Maps tensorflow datatype to a parquet datatype.

    Args:
      parquet_type: a string representing the parquet datatype.

    Returns:
      the tensorflow datatype equivalent of a parquet datatype.
    """
    return {
        "BOOLEAN": tf.bool,
        "INT32": tf.int32,
        "INT64": tf.int64,
        "FLOAT": tf.float32,
        "DOUBLE": tf.double,
        "BYTE_ARRAY": tf.string
    }.get(parquet_type)

  def _as_variant_tensor(self):
    return gen_parquet_dataset.parquet_dataset(
        self._filenames,
        value_paths=self._value_paths,
        value_dtypes=self._value_dtypes,
        parent_index_paths=self._parent_index_paths,
        path_index=self._path_index,
        batch_size=self._batch_size)

  def _inputs(self):
    return []

  @property
  def output_types(self):
    res = []
    column_counter = 0
    prev = self._parent_index_paths[0]
    res.append(tf.int64)
    for i in range(1, len(self._parent_index_paths)):
      curr = self._parent_index_paths[i]
      res.append(tf.int64)
      if curr != prev:
        res.append(self._value_dtypes[column_counter])
        column_counter += 1
        prev = curr
    res.append(tf.int64)
    res.append(self._value_dtypes[column_counter])
    self.output_dtypes = tuple(res)
    return self.output_dtypes

  @property
  def output_shapes(self):
    return (tf.TensorShape([]),) + tuple(
        tf.TensorShape([None]) for i in range(1, len(self.output_dtypes)))

  @property
  def output_classes(self):
    return tuple(tf.Tensor for i in range(len(self.output_dtypes)))


class ParquetDataset(_RawParquetDataset):
  """A dataset which reads columns from a parquet file and returns a prensor.

  The prensor will have a PrensorTypeSpec, which is created based on
  value_paths.

  Note: In tensorflow v1 this dataset will not return a prensor. The output will
  be the same format as _RawParquetDataset's output (a vector of tensors).
  The following is a workaround in v1:
    pq_ds = ParquetDataset(...)
    type_spec = pq_ds.element_spec
    tensors = pq_ds.make_one_shot_iterator().get_next()
    prensor = type_spec.from_components(tensors)
    session.run(prensor)
  """

  def __init__(self, filenames: List[str], value_paths: List[str],
               batch_size: int):
    """Creates a ParquetDataset.

    Args:
      filenames: A list containing the name(s) of the file(s) to be read.
      value_paths: A list of strings of the dotstring path(s) of each leaf
        path(s).
      batch_size: An int that determines how many messages are parsed into one
        prensor tree in an iteration. If there are fewer than batch_size
        remaining messages, then all remaining messages will be returned.

    Raises:
      ValueError: if the column does not exist in the parquet schema.
    """
    self._filenames = filenames
    self._value_paths = value_paths
    self._batch_size = batch_size

    for filename in filenames:
      self._validate_file(filename, value_paths)

    self._value_dtypes = self._get_column_dtypes(filenames[0], value_paths)

    self._parent_index_paths = []
    self._path_index = []

    self.element_structure = self._create_prensor_spec()
    self._create_parent_index_paths_and_index_from_type_spec(
        self.element_structure, 0, 0)

    super(ParquetDataset,
          self).__init__(filenames, self._value_paths, self._value_dtypes,
                         self._parent_index_paths, self._path_index, batch_size)

  def _get_column_dtypes(
      self, metadata_file: str,
      value_paths: List[str]) -> List[Union[tf.DType, None]]:
    """Returns a list of tensorflow datatypes for each column.

    Args:
      metadata_file: the file to be used as the metadata. If there is no
        metadata_file, any file from file_names will suffice.
      value_paths: A list of strings of the dotstring path(s).

    Returns:
      A list of tensorflow datatypes for each column. This list aligns with
      value_paths.
    """
    path_to_column_index = self._get_column_path_to_index_mapping(metadata_file)
    metadata = pq.ParquetFile(metadata_file).metadata

    value_dtypes = []
    for column in value_paths:
      col = metadata.schema.column(path_to_column_index[column])
      parquet_type = col.physical_type
      value_dtypes.append(self._parquet_to_tf_type(parquet_type))
    return value_dtypes

  def _validate_file(self, filename: str, value_paths: List[str]):
    """Checks if each requested path exists in the parquet file.

    Args:
      filename: The parquet filename.
      value_paths: A list of strings of the dotstring path(s).

    Raises:
      ValueError: if a path does not exist in the parquet file's schema.
    """
    metadata = pq.ParquetFile(filename).metadata

    paths = {}
    for i in range(metadata.num_columns):
      col = metadata.schema.column(i)
      p = (col.path)
      paths[p] = col.physical_type

    for i, p in enumerate(value_paths):
      if p not in paths:
        raise ValueError("path " + p + " does not exist in the file.")

  def _create_children_spec(
      self, field: pa.lib.Field, index_and_paths: List[Tuple[int,
                                                             List[path.Step]]]
  ) -> Tuple[path.Step, prensor._PrensorTypeSpec]:
    """Creates the _PrensorTypeSpec for children and leaves.

    Args:
      field: a pyarrow field.
      index_and_paths: a list of tuple(index, list[step]), where index is the
        column index this step belongs to, and list[step] are children steps of
        the passed in step arg. The reason index is needed is because we need to
        keep track of which column this step belongs to, to populate
        parent_index_paths and path_index.

    Returns:
      a child or leaf _PrensorTypeSpec.
    """

    # pylint: disable=protected-access
    curr_steps_as_set = collections.OrderedDict()
    # Construct the dictionary of paths we need.
    if len(index_and_paths) >= 1 and len(index_and_paths[0][1]) >= 1:
      for p in index_and_paths:
        index = p[0]
        p = p[1]
        curr_step = p[0]
        if p:
          if curr_step in curr_steps_as_set:
            curr_steps_as_set[curr_step].append((index, p[1:]))
          else:
            curr_steps_as_set[curr_step] = [(index, p[1:])]

    field_type = field.type
    if isinstance(field_type, pa.lib.ListType):
      field_type = field_type.value_type
      is_repeated = True
    else:
      is_repeated = False
    if isinstance(field_type, pa.lib.StructType):
      node_type = prensor._PrensorTypeSpec._NodeType.CHILD
      dtype = tf.int64
      children = [
          self._create_children_spec(field_type[step], curr_steps_as_set[step])
          for step in curr_steps_as_set
      ]
    else:
      node_type = prensor._PrensorTypeSpec._NodeType.LEAF
      dtype = tf.dtypes.as_dtype(field_type)
      children = []

    return (field.name,
            prensor._PrensorTypeSpec(is_repeated, node_type, dtype, children))

  def _create_prensor_spec(self) -> prensor._PrensorTypeSpec:  # pylint: disable=protected-access
    """Creates the prensor type spec based on value_paths.

    Returns:
      a root _PrensorTypeSpec.
    """

    metadata = pq.ParquetFile(self._filenames[0]).metadata
    parquet_schema = metadata.schema
    arrow_schema = parquet_schema.to_arrow_schema()

    # pylint: disable=protected-access
    # Sort the paths by number of fields.
    paths = [path.create_path(p) for p in self._value_paths]
    mapped = zip(paths, self._value_paths, self._value_dtypes)
    sorted_mapped = sorted(mapped, key=lambda x: len(x[0].field_list))
    paths, self._value_paths, self._value_dtypes = zip(*sorted_mapped)

    # Creates an ordered dictionary mapping step to a list of children fields.
    # This will allow us to find paths that share a parent.
    curr_steps_as_set = collections.OrderedDict()
    for (i, p) in enumerate(paths):
      step = p.field_list[0]
      if step in curr_steps_as_set:
        curr_steps_as_set[step].append((i, p.field_list[1:]))
      else:
        curr_steps_as_set[step] = [(i, p.field_list[1:])]

    return prensor._PrensorTypeSpec(
        None, prensor._PrensorTypeSpec._NodeType.ROOT, tf.int64, [
            self._create_children_spec(
                arrow_schema.field(step), curr_steps_as_set[step])
            for step in curr_steps_as_set
        ])

  def _create_parent_index_paths_and_index_from_type_spec(
      self, type_spec, index, level):
    """Populates self._parent_index_paths and self.path_index from the typespec.

    It traverses the prensor type spec to get index and level. It then uses
    index to get the correct path from self._value_paths.

    This assumes that self._value_paths is sorted alphabetically, and thus the
    prensor type spec has the same order of paths as self._value_paths.

    Args:
      type_spec: A Prensor type spec.
      index: The index of self._value_paths. It is incremented each time we
        reach a leaf, ie we have a new path.
      level: the step number in a path. It is incremented each time we go to a
        spec's child. It is then decremented when exiting the child spec.
    """
    fields = type_spec._children_specs  # pylint: disable=protected-access

    for field_tuple in fields:
      spec = field_tuple[1]
      self._parent_index_paths.append(self._value_paths[index])
      self._path_index.append(level)
      level += 1
      self._create_parent_index_paths_and_index_from_type_spec(
          spec, index, level)
      level -= 1
      index += 1

  @property
  def element_spec(self):
    return self.element_structure


def _create_children_from_arrow_fields(
    fields: pa.lib.Field) -> Dict[str, Dict[Any, Any]]:
  """Creates a dictionary of children schema for a pyarrow field.

  Args:
    fields: A list of pyarrow fields.

  Returns:
    A dictionary of children. Key is field name. Value is a dictionary
    representing a schema.
  """
  children = {}
  for field in fields:
    field_type = field.type
    if isinstance(field_type, pa.lib.ListType):
      sub_field_type = field_type.value_type
      if isinstance(sub_field_type, pa.lib.StructType):
        children[field.name] = {
            "is_repeated":
                True,
            "children":
                _create_children_from_arrow_fields(
                    [subfield for subfield in sub_field_type])
        }
      elif isinstance(sub_field_type, pa.lib.DataType):
        children[field.name] = {
            "is_repeated": True,
            "dtype": tf.dtypes.as_dtype(sub_field_type)
        }
      else:
        print("this should never be printed")
    elif isinstance(field_type, pa.lib.StructType):
      children[field.name] = {
          "is_repeated":
              False,
          "children":
              _create_children_from_arrow_fields(
                  [subfield for subfield in field_type])
      }
    else:
      children[field.name] = {
          "is_repeated": False,
          "dtype": tf.dtypes.as_dtype(field_type)
      }
  return children


class _ParquetDatasetWithExpression(ParquetDataset):
  """A dataset which reads columns from a parquet file based on the expressions.

  The data read from the parquet file will then have the expression queries
  applied to it, creating a new prensor.

  This dataset should not be created by the user, call
  parquet_dataset.calculate_parquet_values() to get this dataset instead.
  """

  def __init__(self, exprs: List[expression.Expression],
               root_expr: placeholder._PlaceholderRootExpression,
               filenames: List[str], batch_size: int,
               options: Optional[calculate_options.Options]):
    self._exprs = exprs
    self._root_expr = root_expr
    self._filesnames = filenames
    self._batch_size = batch_size
    self._options = options

    # pylint: disable=protected-access
    self._subtrees = [x.get_known_descendants() for x in self._exprs]
    self._all_expressions = []
    for tree in self._subtrees:
      self._all_expressions.extend(tree.values())

    expression_graph = calculate.OriginalExpressionGraph(self._all_expressions)
    self._canonical_graph = calculate.CanonicalExpressionGraph(expression_graph)
    paths = placeholder.get_placeholder_paths_from_graph(self._canonical_graph)

    parquet_paths = [".".join(p.field_list) for p in paths]

    super(_ParquetDatasetWithExpression,
          self).__init__(filenames, parquet_paths, batch_size)

  def _calculate_prensor(self, pren) -> List[prensor.Prensor]:
    """Function for applying expression queries to a prensor.

    This function should be passed into dataset.map().

    Args:
      pren: The prensor that will be used to bind to the root expression.

    Returns:
      A list of modified prensor that have the expression queries applied.
    """
    self._canonical_graph.calculate_values(
        options=self._options, feed_dict={self._root_expr: pren})
    values = [
        self._canonical_graph.get_value_or_die(x) for x in self._all_expressions
    ]

    expr_to_value_map = {
        id(expr): value for expr, value in zip(self._all_expressions, values)
    }

    # pylint: disable=protected-access
    return [
        calculate._get_prensor(subtree, expr_to_value_map)
        for subtree in self._subtrees
    ]
