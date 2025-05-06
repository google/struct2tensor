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
"""Placeholder expression.

A placeholder expression represents prensor nodes, however a prensor is not
needed until calculate is called. This allows the user to apply expression
queries to a placeholder expression before having an actual prensor object.
When calculate is called on a placeholder expression (or a descendant of a
placeholder expression), the feed_dict will need to be passed in. Then calculate
will bind the prensor with the appropriate placeholder expression.

Sample usage:

```
placeholder_exp = placeholder.create_expression_from_schema(schema)
new_exp = expression_queries(placeholder_exp, ..)
result = calculate.calculate_values([new_exp],
                                    feed_dict={placeholder_exp: pren})
# placeholder_exp requires a feed_dict to be passed in when calculating
```

"""

import typing
from typing import FrozenSet, List, Optional, Sequence, Union

from struct2tensor import calculate
from struct2tensor import calculate_options
from struct2tensor import expression
from struct2tensor import path
from struct2tensor import prensor
from struct2tensor.expression_impl import map_prensor_to_prensor as mpp


def create_expression_from_schema(
    schema: mpp.Schema) -> "_PlaceholderRootExpression":
  """Creates a placeholder expression from a parquet schema.

  Args:
    schema: The schema that describes the prensor tree that this placeholder
      represents.

  Returns:
    A PlaceholderRootExpression that should be used as the root of an expression
    graph.
  """

  return _PlaceholderRootExpression(schema)


def _is_placeholder_expression(expr: expression.Expression) -> bool:
  """Returns true if an expression is a ParquetExpression."""
  return isinstance(expr,
                    (_PlaceholderRootExpression, _PlaceholderChildExpression))


def get_placeholder_paths_from_graph(
    graph: calculate.ExpressionGraph) -> List[path.Path]:
  """Gets all placeholder paths from an expression graph.

  This finds all leaf placeholder expressions in an expression graph, and gets
  the path of these expressions.

  Args:
    graph: expression graph

  Returns:
    a list of paths of placeholder expressions
  """
  expressions = [
      x for x in graph.get_expressions_needed()
      if (_is_placeholder_expression(x) and x.is_leaf)
  ]
  expressions = typing.cast(List[_PlaceholderExpression], expressions)
  return [e.get_path() for e in expressions]


class _PlaceholderChildExpression(expression.Expression):
  """A child or leaf parquet expression."""

  # pylint: disable=protected-access
  def __init__(self, parent: "_PlaceholderExpression", step: path.Step,
               schema: mpp.Schema):
    super().__init__(
        schema.is_repeated, schema.type, schema_feature=schema.schema_feature)
    self._parent = parent
    self._step = step
    self._schema = schema

  @property
  def schema(self):
    return self._schema

  @property
  def is_leaf(self) -> bool:
    return not self._schema._children

  def get_path(self) -> path.Path:
    return self._parent.get_path().get_child(self._step)

  def get_source_expressions(self) -> Sequence[expression.Expression]:
    return [self._parent]

  def calculate(  # pytype: disable=signature-mismatch  # overriding-parameter-type-checks
      self,
      source_tensors: Sequence[mpp._TreeAsNode],
      destinations: Sequence[expression.Expression],
      options: calculate_options.Options,
      side_info: Optional[prensor.Prensor] = None) -> mpp._TreeAsNode:
    if side_info:
      return mpp._tree_as_node(side_info)
    [parent] = source_tensors
    if not isinstance(
        parent, (mpp._PrensorAsLeafNodeTensor, mpp._PrensorAsChildNodeTensor,
                 mpp._PrensorAsRootNodeTensor)):
      raise ValueError("calculate() of Parent did not return a "
                       "_PrensorAsNodeTensor")
    my_pren = parent.prensor.get_child(self._step)
    if not my_pren:
      raise ValueError("step " + self._step + " does not exist in prensor: " +
                       str(parent.prensor))
    return mpp._tree_as_node(my_pren)

  def calculation_is_identity(self) -> bool:
    return False

  def calculation_equal(self, expr: expression.Expression) -> bool:
    return self is expr

  def _get_child_impl(self,
                      field_name: path.Step) -> Optional[expression.Expression]:
    if field_name not in self._schema.known_field_names():
      return None
    child_schema = self._schema.get_child(field_name)
    return _PlaceholderChildExpression(self, field_name, child_schema)

  def known_field_names(self) -> FrozenSet[path.Step]:
    return self._schema.known_field_names()


class _PlaceholderRootExpression(expression.Expression):
  """An expression that calculates to the side_info passed in at calculate()."""

  def __init__(self, schema: mpp.Schema):
    """Initializes the root of the placeholder expression.

    Args:
      schema: the schema that represents what the expression tree looks like.
    """
    super().__init__(True, None)
    self._schema = schema

  @property
  def schema(self):
    return self._schema

  @property
  def is_leaf(self) -> bool:
    return False

  def get_path(self) -> path.Path:
    return path.Path([])

  def get_source_expressions(self) -> Sequence[expression.Expression]:
    return []

  def calculate(  # pytype: disable=signature-mismatch  # overriding-parameter-type-checks
      self, source_tensors: Sequence[prensor.NodeTensor],
      destinations: Sequence[expression.Expression],
      options: calculate_options.Options,
      side_info: prensor.Prensor) -> mpp._TreeAsNode:
    if source_tensors:
      raise ValueError("_PlaceholderRootExpression has no sources")
    if side_info:
      return mpp._tree_as_node(side_info)  # pylint: disable=protected-access
    else:
      raise ValueError("_PlaceholderRootExpression requires side_info")

  def calculation_is_identity(self) -> bool:
    return False

  def calculation_equal(self, expr: expression.Expression) -> bool:
    return self is expr

  def _get_child_impl(self,
                      field_name: path.Step) -> Optional[expression.Expression]:
    if field_name not in self._schema.known_field_names():
      return None
    child_schema = self._schema.get_child(field_name)
    return _PlaceholderChildExpression(self, field_name, child_schema)

  def __str__(self) -> str:
    return "_PlaceholderRootExpression: {}".format(str(self._schema))

  def known_field_names(self) -> FrozenSet[path.Step]:
    return self._schema.known_field_names()


_PlaceholderExpression = Union[_PlaceholderRootExpression,
                               _PlaceholderChildExpression]
