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
"""Implementation of slice.


The slice operation is meant to replicate the slicing of a list in python.

Slicing a list in python is done by specifying a beginning and ending.
The resulting list consists of all elements in the range.

For example:

```
>>> x = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
>>> print(x[2:5]) # all elements between index 2 inclusive and index 5 exclusive
['c', 'd', 'e']
>>> print(x[2:]) # all elements between index 2 and the end.
['c', 'd', 'e', 'f', 'g']
>>> print(x[:4]) # all elements between the beginning and index 4 (exclusive).
['a', 'b', 'c', 'd']
>>> print(x[-3:-1]) # all elements starting three from the end.
>>>                 # until one from the end (exclusive).
['e', 'f']
>>> print(x[-3:6]) # all elements starting three from the end
                   # until index 6 exclusive.
['e', 'f', 'g']
```

TODO(martinz): there is a third argument to slice, which allows one to step
over the elements (e.g. x[2:6:2]=['c', 'e'], giving you every other element.
This is not implemented.


A prensor can be considered to be interleaved lists and dictionaries.
E.g.:

```
my_expression = [{
  "foo":[
    {"bar":[
      {"baz":["a","b","c", "d"]},
      {"baz":["d","e","f"]}
      ]
    },
    {"bar":[
      {"baz":["g","h","i"]},
      {"baz":["j","k","l", ]}
      {"baz":["m"]}
    ]
    }]
}]
```

```
result_1 = slice_expression.slice_expression(
  my_expression, "foo.bar", "new_bar",begin=1, end=3)

result_1 = [{
  "foo":[
    {"bar":[
      {"baz":["a","b","c", "d"]},
      {"baz":["d","e","f"]}
      ],
     "new_bar":[
      {"baz":["d","e","f"]}
      ]
    },
    {"bar":[
      {"baz":["g","h","i"]},
      {"baz":["j","k","l", ]}
      {"baz":["m", ]}
     ],
    "new_bar":[
      {"baz":["j","k","l", ]}
      {"baz":["m", ]}
    ]
    }]
}]
```

```
result_2 = slice_expression.slice_expression(
  my_expression, "foo.bar.baz", "new_baz",begin=1, end=3)

result_2 = [{
  "foo":[
    {"bar":[
      {"baz":["a","b","c", "d"],
       "new_baz":["b","c"],
      },
      {"baz":["d","e","f"], "new_baz":["e","f"]}
      ]
    },
    {"bar":[
      {"baz":["g","h","i"], "new_baz":["h","i"]},
      {"baz":["j","k","l"], "new_baz":["k","l"]},
      {"baz":["m", ]}
      ]
    }]
}]
```

"""

from typing import Callable, Optional, Tuple

from struct2tensor import expression
from struct2tensor import expression_add
from struct2tensor import path
from struct2tensor.expression_impl import filter_expression
from struct2tensor.expression_impl import index
from struct2tensor.expression_impl import map_values
import tensorflow as tf

IndexValue = expression.IndexValue


def slice_expression(expr: expression.Expression, p: path.Path,
                     new_field_name: path.Step, begin: Optional[IndexValue],
                     end: Optional[IndexValue]) -> expression.Expression:
  """Creates a new subtree with a sliced expression.

  This follows the pattern of python slice() method.
  See module-level comments for examples.

  Args:
    expr: the original root expression
    p: the path to the source to be sliced.
    new_field_name: the name of the new subtree.
    begin: beginning index
    end: end index.

  Returns:
    A new root expression.
  """
  work_expr, mask_anonymous_path = _get_slice_mask(expr, p, begin, end)
  work_expr = filter_expression.filter_by_sibling(
      work_expr, p, mask_anonymous_path.field_list[-1], new_field_name)
  new_path = p.get_parent().get_child(new_field_name)
  # We created a lot of anonymous fields and intermediate expressions. Just grab
  # the final result (and its children).
  return expression_add.add_to(expr, {new_path: work_expr})


def _get_mask(t: expression.Expression, p: path.Path, threshold: IndexValue,
              relation: Callable[[tf.Tensor, IndexValue], tf.Tensor]
             ) -> Tuple[expression.Expression, path.Path]:
  """Gets a mask based on a relation of the index to a threshold.

  If the threshold is non-negative, then we create a mask that is true if
  the relation(index, threshold) is true.

  If the threshold is negative, then we create a mask that is true if
  the relation(size - index, threshold) is true.

  Args:
    t: expression to add the field to.
    p: path to create the mask for.
    threshold: the cutoff threshold.
    relation: tf.less or tf.greater_equal.

  Returns:
    A boolean mask on the fields to keep on the model.

  Raises:
    ValueError: if p is not in t.
  """
  if t.get_descendant(p) is None:
    raise ValueError("Path not found: {}".format(str(p)))
  work_expr, index_from_end = index.get_index_from_end(
      t, p, path.get_anonymous_field())
  work_expr, mask_for_negative_threshold = map_values.map_values_anonymous(
      work_expr,
      index_from_end, lambda x: relation(x, tf.cast(threshold, tf.int64)),
      tf.bool)

  work_expr, positional_index = index.get_positional_index(
      work_expr, p, path.get_anonymous_field())
  work_expr, mask_for_non_negative_threshold = map_values.map_values_anonymous(
      work_expr,
      positional_index, lambda x: relation(x, tf.cast(threshold, tf.int64)),
      tf.bool)

  if isinstance(threshold, int):
    if threshold >= 0:
      return work_expr, mask_for_non_negative_threshold
    return work_expr, mask_for_negative_threshold
  else:

    def tf_cond_on_threshold(a, b):
      return tf.cond(tf.greater_equal(threshold, 0), a, b)

    return map_values.map_many_values(work_expr, p.get_parent(), [
        x.field_list[-1]
        for x in [mask_for_non_negative_threshold, mask_for_negative_threshold]
    ], tf_cond_on_threshold, tf.bool, path.get_anonymous_field())


def _get_begin_mask(expr: expression.Expression, p: path.Path, begin: IndexValue
                   ) -> Tuple[expression.Expression, path.Path]:
  """Get a boolean mask of what indices to retain for slice given begin."""
  return _get_mask(expr, p, begin, tf.greater_equal)


def _get_end_mask(t: expression.Expression, p: path.Path,
                  end: IndexValue) -> Tuple[expression.Expression, path.Path]:
  """Get a boolean mask of what indices to retain for slice given end."""
  return _get_mask(t, p, end, tf.less)


def _get_slice_mask(
    expr: expression.Expression, p: path.Path, begin: Optional[IndexValue],
    end: Optional[IndexValue]) -> Tuple[expression.Expression, path.Path]:
  """Gets a mask for slicing a path.

  One way to consider the elements of a path "foo.bar" is as a list of list of
  list of elements. Slicing a path slices this doubly nested list of elements,
  based upon positions in its parent list. Each parent list has a size, and
  there is a beginning and end relative to the elements in that list.

  At each path p, there is conceptually a list of...list of elements.

  For example, given:
  an index with respect to its parent
  The range is specified with beginning and an end.
  1. If begin is not present, begin_index is implied to be zero.
  2. If begin is negative, begin_index is the size of a particular
      list + begin
  3. If end is not present, end_index is the length of the list + 1.
  4. If end is negative, end_index is the length of the list + end
  5. If end is non-negative, end_index is end.
  The mask is positive for all elements in range(begin_index, end_index), and
  negative elsewhere.

  The mask returned is a sibling of path p, where for every element in p, there
  is a corresponding element in the mask.

  Args:
    expr: the root expression
    p: the path to be sliced
    begin: the beginning index
    end: the ending index

  Returns:
    An expression,path pair, where the expression contains all the children in
    `expr` and an anonymous field of the mask and the path points to
    the mask field.
  """
  if begin is None:
    if end is None:
      raise ValueError("Must specify begin or end.")
    return _get_end_mask(expr, p, end)
  else:
    if end is None:
      return _get_begin_mask(expr, p, begin)
    work_expr, begin_mask = _get_begin_mask(expr, p, begin)
    work_expr, end_mask = _get_end_mask(work_expr, p, end)
    return map_values.map_many_values(
        work_expr, p.get_parent(),
        [x.field_list[-1] for x in [begin_mask, end_mask]], tf.logical_and,
        tf.bool, path.get_anonymous_field())
