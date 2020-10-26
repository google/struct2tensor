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
"""Functions for creating new size or has expression.

Given a field "foo.bar",

```
root = size(expr, path.Path(["foo","bar"]), "bar_size")
```

creates a new expression root that has an optional field "foo.bar_size", which
is always present, and contains the number of bar in a particular foo.

```
root_2 = has(expr, path.Path(["foo","bar"]), "bar_has")
```

creates a new expression root that has an optional field "foo.bar_has", which
is always present, and is true if there are one or more bar in foo.
"""
from typing import Optional, Sequence, Tuple

from struct2tensor import calculate_options
from struct2tensor import expression
from struct2tensor import expression_add
from struct2tensor import path
from struct2tensor import prensor
from struct2tensor.expression_impl import map_values
import tensorflow as tf


def size_anonymous(root: expression.Expression, source_path: path.Path
                  ) -> Tuple[expression.Expression, path.Path]:
  """Calculate the size of a field, and store it as an anonymous sibling.

  Args:
    root: the original expression.
    source_path: the source path to measure. Cannot be root.

  Returns:
    The new expression and the new field as a pair.
  """
  return _size_impl(root, source_path, path.get_anonymous_field())


def size(root: expression.Expression, source_path: path.Path,
         new_field_name: path.Step) -> expression.Expression:
  """Get the size of a field as a new sibling field.

  Args:
    root: the original expression.
    source_path: the source path to measure. Cannot be root.
    new_field_name: the name of the sibling field.

  Returns:
    The new expression.
  """
  return _size_impl(root, source_path, new_field_name)[0]


def has(root: expression.Expression, source_path: path.Path,
        new_field_name: path.Step) -> expression.Expression:
  """Get the has of a field as a new sibling field.

  Args:
    root: the original expression.
    source_path: the source path to measure. Cannot be root.
    new_field_name: the name of the sibling field.

  Returns:
    The new expression.
  """
  new_root, size_p = size_anonymous(root, source_path)
  # TODO(martinz): consider using copy_over to "remove" the size field
  # from the result.
  return map_values.map_values(
      new_root, size_p, lambda x: tf.greater(x, tf.constant(0, dtype=tf.int64)),
      tf.bool, new_field_name)


class SizeExpression(expression.Leaf):
  """Size of the given expression.

  SizeExpression is intended to be a sibling of origin.
  origin_parent should be the parent of origin.

  """

  def __init__(self, origin: expression.Expression,
               origin_parent: expression.Expression):
    super().__init__(False, tf.int64)
    self._origin = origin
    self._origin_parent = origin_parent

  def get_source_expressions(self) -> Sequence[expression.Expression]:
    return [self._origin, self._origin_parent]

  def calculate(
      self,
      sources: Sequence[prensor.NodeTensor],
      destinations: Sequence[expression.Expression],
      options: calculate_options.Options,
      side_info: Optional[prensor.Prensor] = None) -> prensor.NodeTensor:

    [origin_value, origin_parent_value] = sources
    if not isinstance(origin_value,
                      (prensor.LeafNodeTensor, prensor.ChildNodeTensor)):
      raise ValueError(
          "origin_value must be a LeafNodeTensor or a ChildNodeTensor, "
          "but was a " + str(type(origin_value)))

    if not isinstance(origin_parent_value,
                      (prensor.ChildNodeTensor, prensor.RootNodeTensor)):
      raise ValueError("origin_parent_value must be a ChildNodeTensor "
                       "or a RootNodeTensor, but was a " +
                       str(type(origin_parent_value)))

    parent_index = origin_value.parent_index
    num_parent_protos = origin_parent_value.size
    # A vector of 1s of the same size as the parent_index.
    updates = tf.ones(tf.shape(parent_index), dtype=tf.int64)
    indices = tf.expand_dims(parent_index, 1)
    # This is incrementing the size by 1 for each element.
    # Obviously, not the fastest way to do this.
    values = tf.scatter_nd(indices, updates, tf.reshape(num_parent_protos, [1]))

    # Need to create a new_parent_index = 0,1,2,3,4...n.
    new_parent_index = tf.range(num_parent_protos, dtype=tf.int64)
    return prensor.LeafNodeTensor(new_parent_index, values, False)

  def calculation_is_identity(self) -> bool:
    return False

  def calculation_equal(self, expr: expression.Expression) -> bool:
    return isinstance(expr, SizeExpression)


def _size_impl(
    root: expression.Expression, source_path: path.Path,
    new_field_name: path.Step) -> Tuple[expression.Expression, path.Path]:
  if not source_path:
    raise ValueError("Cannot get the size of the root.")
  if root.get_descendant(source_path) is None:
    raise ValueError("Path not found: {}".format(str(source_path)))
  parent_path = source_path.get_parent()
  new_path = parent_path.get_child(new_field_name)
  return expression_add.add_paths(
      root, {
          new_path:
              SizeExpression(
                  root.get_descendant_or_error(source_path),
                  root.get_descendant_or_error(parent_path))
      }), new_path
