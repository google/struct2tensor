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
"""get_positional_index and get_index_from_end methods.

The parent_index identifies the index of the parent of each element. These
methods take the parent_index to determine the relationship with respect to
other elements.

Given:

```
session: {
  event: {
    val: 111
  }
  event: {
    val: 121
    val: 122
  }
}

session: {
  event: {
    val: 10
    val: 7
  }
  event: {
    val: 1
  }
}
```

```
get_positional_index(expr, path.Path(["event","val"]), "val_index")
```

yields:

```
session: {
  event: {
    val: 111
    val_index: 0
  }
  event: {
    val: 121
    val: 122
    val_index: 0
    val_index: 1
  }
}

session: {
  event: {
    val: 10
    val: 7
    val_index: 0
    val_index: 1
  }
  event: {
    val: 1
    val_index: 0
  }
}
```

```
get_index_from_end(expr, path.Path(["event","val"]), "neg_val_index")
```
yields:

```
session: {
  event: {
    val: 111
    neg_val_index: -1
  }
  event: {
    val: 121
    val: 122
    neg_val_index: -2
    neg_val_index: -1
  }
}

session: {
  event: {
    val: 10
    val: 7
    neg_val_index: 2
    neg_val_index: -1
  }
  event: {
    val: 1
    neg_val_index: -1
  }
}
```

These methods are useful when you want to depend upon the index of a field.
For example, if you want to filter examples based upon their index, or
cogroup two fields by index, then first creating the index is useful.

Note that while the parent indices of these fields seem like overhead, they
are just references to the parent indices of other fields, and are therefore
take little memory or CPU.
"""

from typing import Optional, Sequence, Tuple

from struct2tensor import calculate_options
from struct2tensor import expression
from struct2tensor import expression_add
from struct2tensor import path
from struct2tensor import prensor
from struct2tensor.expression_impl import size
import tensorflow as tf


def get_positional_index(expr: expression.Expression, source_path: path.Path,
                         new_field_name: path.Step
                        ) -> Tuple[expression.Expression, path.Path]:
  """Gets the positional index.

  Given a field with parent_index [0,1,1,2,3,4,4], this returns:
  parent_index [0,1,1,2,3,4,4] and value [0,0,1,0,0,0,1]

  Args:
    expr: original expression
    source_path: path in expression to get index of.
    new_field_name: the name of the new field.

  Returns:
    The new expression and the new path as a pair.
  """
  new_path = source_path.get_parent().get_child(new_field_name)
  return expression_add.add_paths(
      expr, {
          new_path:
              _PositionalIndexExpression(
                  expr.get_descendant_or_error(source_path))
      }), new_path


def get_index_from_end(t: expression.Expression, source_path: path.Path,
                       new_field_name: path.Step
                      ) -> Tuple[expression.Expression, path.Path]:
  """Gets the number of steps from the end of the array.

  Given an array ["a", "b", "c"], with indices [0, 1, 2], the result of this
  is [-3,-2,-1].

  Args:
    t: original expression
    source_path: path in expression to get index of.
    new_field_name: the name of the new field.

  Returns:
    The new expression and the new path as a pair.
  """
  new_path = source_path.get_parent().get_child(new_field_name)
  work_expr, positional_index_path = get_positional_index(
      t, source_path, path.get_anonymous_field())
  work_expr, size_path = size.size_anonymous(work_expr, source_path)
  work_expr = expression_add.add_paths(
      work_expr, {
          new_path:
              _PositionalIndexFromEndExpression(
                  work_expr.get_descendant_or_error(positional_index_path),
                  work_expr.get_descendant_or_error(size_path))
      })
  # Removing the intermediate anonymous nodes.
  result = expression_add.add_to(t, {new_path: work_expr})
  return result, new_path


class _PositionalIndexExpression(expression.Leaf):
  """The positional index for the origin.

  _PositionalIndexExpression is intended to be a sibling of source.
  The operation will return a field that has the same parent index as source.
  """

  def __init__(self, origin: expression.Expression):
    super().__init__(origin.is_repeated, tf.int64)
    self._origin = origin

  def get_source_expressions(self) -> Sequence[expression.Expression]:
    return [self._origin]

  def calculate(
      self,
      sources: Sequence[prensor.NodeTensor],
      destinations: Sequence[expression.Expression],
      options: calculate_options.Options,
      side_info: Optional[prensor.Prensor] = None) -> prensor.NodeTensor:
    [origin] = sources
    if isinstance(origin, (prensor.LeafNodeTensor, prensor.ChildNodeTensor)):
      return prensor.LeafNodeTensor(
          origin.parent_index,
          origin.get_positional_index(),
          self.is_repeated)
    raise ValueError("Cannot calculate the positional index of the root")

  def calculation_is_identity(self) -> bool:
    return False

  def calculation_equal(self, expr: expression.Expression) -> bool:
    return isinstance(expr, _PositionalIndexExpression)


class _PositionalIndexFromEndExpression(expression.Leaf):
  """The positional index from the end.

  _PositionalIndexFromEndExpression is intended to be a sibling of
  positional_index.
  The operation will return a field that has the same parent index as source.
  """

  def __init__(self, positional_index: expression.Expression,
               size_inp: expression.Expression):
    """Create a new expression.

    Note that while positional_index and size are both siblings of the
    original field, positional_index is index-aligned with the field,
    whereas size is a "required" field.

    Args:
      positional_index: the positional index of the field (from
        get_positional_index).
      size_inp: the size of the field (from size.size).
    """
    super(_PositionalIndexFromEndExpression,
          self).__init__(positional_index.is_repeated, tf.int64)
    self._positional_index = positional_index
    self._size = size_inp

  def get_source_expressions(self) -> Sequence[expression.Expression]:
    return [self._positional_index, self._size]

  def calculate(
      self,
      sources: Sequence[prensor.NodeTensor],
      destinations: Sequence[expression.Expression],
      options: calculate_options.Options,
      side_info: Optional[prensor.Prensor] = None) -> prensor.NodeTensor:
    [positional_index, size_value] = sources
    if not isinstance(positional_index, prensor.LeafNodeTensor):
      raise ValueError("positional_index must be a LeafNodeTensor")
    if not isinstance(size_value, prensor.LeafNodeTensor):
      raise ValueError("size_value must be a LeafNodeTensor")

    size_per_index = tf.gather(size_value.values, positional_index.parent_index)
    return prensor.LeafNodeTensor(positional_index.parent_index,
                                  positional_index.values - size_per_index,
                                  self.is_repeated)

  def calculation_is_identity(self) -> bool:
    return False

  def calculation_equal(self, expr: expression.Expression) -> bool:
    return isinstance(expr, _PositionalIndexFromEndExpression)
