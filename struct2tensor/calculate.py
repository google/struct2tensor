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
"""Calculate the value of a list of expressions.

This library contains functions for calculating the value of expressions.

calculate_values will, given a list of expressions, calculate the NodeTensor
of each expression.

calculate_prensors will, given a list of expressions, calculate the prensors
associated with each expression (i.e., the NodeTensor of each expression, but
also the value of the descendants of each expression). This is the most common
method here.

calculate_values_with_graph and calculate_prensors_with_graph will, in addition,
also return the ExpressionGraph used to calculate the result. This is useful if
you want to know what other expressions were used to calculate a value (e.g.,
if you want to know what fields in the original protobuf tensor were parsed).


All of this code does a variety of optimizations:

1. Eliminating identity operations. Identity operations are removed from the
   graph.

2. Eliminating common subexpression. If two sub-expressions are identical, then
   they are only executed once.

3. Calculating values based upon dependencies. For example, for protobufs,
   this will only parse fields based upon what is needed later in the
   calculation.

"""

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from struct2tensor import calculate_options
from struct2tensor import expression
from struct2tensor import path
from struct2tensor import prensor
import tensorflow as tf

# type(id(...)), disambiguated for clarity
IDExpression = int
IDNodeTensor = int


def calculate_values_with_graph(
    expressions: List[expression.Expression],
    options: Optional[calculate_options.Options] = None,
    feed_dict: Optional[Dict[expression.Expression, prensor.Prensor]] = None
) -> Tuple[List[prensor.NodeTensor], "ExpressionGraph"]:
  """Calculates the values of the expressions, and the graph used.

  Note that this does not return prensors, but instead a list of NodeTensors.

  Args:
    expressions: a list of expressions to calculate.
    options: options for calculations, passed to calculate(...).
    feed_dict: a dictionary, mapping expression to prensor that will be used
      as the initial expression in the expression graph.

  Returns:
    the list of values and the graph used to calculate them.
  """
  if options is None:
    options = calculate_options.get_default_options()
  expression_graph = _create_graph(expressions, options, feed_dict=feed_dict)
  return ([expression_graph.get_value_or_die(x) for x in expressions],
          expression_graph)


def calculate_values(
    expressions: List[expression.Expression],
    options: Optional[calculate_options.Options] = None,
    feed_dict: Optional[Dict[expression.Expression, prensor.Prensor]] = None
) -> List[prensor.NodeTensor]:
  """Calculates the values of the expressions.

  Note that this does not return prensors, but instead a list of NodeTensors.

  Args:
    expressions: A list of expressions to calculate.
    options: options for calculate(...) operations.
    feed_dict: a dictionary, mapping expression to prensor that will be used
      as the initial expression in the expression graph.

  Returns:
    A list of NodeTensor values.
  """
  return calculate_values_with_graph(
      expressions, options=options, feed_dict=feed_dict)[0]


def calculate_prensors_with_graph(
    expressions: Sequence[expression.Expression],
    options: Optional[calculate_options.Options] = None,
    feed_dict: Optional[Dict[expression.Expression, prensor.Prensor]] = None
) -> Tuple[Sequence[prensor.Prensor], "ExpressionGraph"]:
  """Gets the prensor value of the expressions and the graph used.

  This method is useful for getting information like the protobuf fields parsed
  to create an expression.

  Args:
    expressions: expressions to calculate prensors for.
    options: options for calculate(...) methods.
    feed_dict: a dictionary, mapping expression to prensor that will be used
      as the initial expression in the expression graph.

  Returns:
    a list of prensors, and the graph used to calculate them.
  """
  subtrees = [x.get_known_descendants() for x in expressions]
  all_expressions = []
  for tree in subtrees:
    all_expressions.extend(tree.values())
  values, graph = calculate_values_with_graph(
      all_expressions, options=options, feed_dict=feed_dict)
  expr_value_pairs = zip(all_expressions, values)
  value_map = {}
  for expr, value in expr_value_pairs:
    if id(expr) not in value_map:
      value_map[id(expr)] = value
  return ([_get_prensor(subtree, value_map) for subtree in subtrees], graph)


def calculate_prensors(
    expressions: Sequence[expression.Expression],
    options: Optional[calculate_options.Options] = None,
    feed_dict: Optional[Dict[expression.Expression, prensor.Prensor]] = None
) -> Sequence[prensor.Prensor]:
  """Gets the prensor value of the expressions.

  Args:
    expressions: expressions to calculate prensors for.
    options: options for calculate(...).
    feed_dict: a dictionary, mapping expression to prensor that will be used
      as the initial expression in the expression graph.

  Returns:
    a list of prensors.
  """

  return calculate_prensors_with_graph(
      expressions, options=options, feed_dict=feed_dict)[0]


# TODO(martinz): Create an option to create the original expression graph.
def _create_graph(
    expressions: List[expression.Expression],
    options: calculate_options.Options,
    feed_dict: Optional[Dict[expression.Expression, prensor.Prensor]] = None
) -> "ExpressionGraph":
  """Create graph and calculate expressions."""
  expression_graph = OriginalExpressionGraph(expressions)
  canonical_graph = CanonicalExpressionGraph(expression_graph)
  canonical_graph.calculate_values(options, feed_dict=feed_dict)
  return canonical_graph


def _get_prensor(subtree: Mapping[path.Path, expression.Expression],
                 values: Mapping[IDNodeTensor, prensor.NodeTensor]
                ) -> prensor.Prensor:
  """Gets the prensor tree value of the subtree.

  Args:
    subtree: a mapping paths to expressions returned by
      expression.get_known_descendants.
    values: a mapping from expression ids to node values.

  Returns:
    a prensor tree.
  """
  return prensor.create_prensor_from_descendant_nodes(
      {k: values[id(v)] for k, v in subtree.items()})


def _get_earliest_equal_calculation(expr: expression.Expression):
  """Finds an expression with an equal value.

  Recursively traverses sources while expressions are the identity.
  Args:
    expr: the expression to find an equal value.

  Returns:
    An expression with an equal value to the input.
  """
  result = expr
  while result.calculation_is_identity():
    result = result.get_source_expressions()[0]
  return result


def _fancy_type_str(is_repeated: bool, dtype: Optional[tf.DType]):
  return "{} {}".format("repeated" if is_repeated else "optional", dtype)


def _node_type_str(node_tensor: prensor.NodeTensor):
  if isinstance(node_tensor, prensor.LeafNodeTensor):
    return _fancy_type_str(node_tensor.is_repeated, node_tensor.values.dtype)
  else:
    return _fancy_type_str(node_tensor.is_repeated, None)


class _ExpressionNode(object):
  """A node representing an expression in the ExpressionGraph."""

  def __init__(self, expr: expression.Expression):
    """Construct a node in the graph.

    Args:
      expr: must be the result of _get_earliest_equal_calculation(...)
    """

    self.expression = expr
    self.sources = [
        _get_earliest_equal_calculation(x)
        for x in expr.get_source_expressions()
    ]
    self.destinations = []  # type: List[_ExpressionNode]
    self.value = None

  def __eq__(self, node: "_ExpressionNode") -> bool:
    """Test if this node is equal to the other.

    Requires that all sources are already canonical.

    Args:
      node: Another node to compare to.

    Returns:
      True if the nodes are guaranteed to have equal value.
    """
    if not self.expression.calculation_equal(node.expression):
      return False
    # This assumes the sources are canonical, so we check for identity equality
    # as opposed to semantic equality.
    return all([a is b for a, b in zip(self.sources, node.sources)])

  def __str__(self) -> str:
    return ("expression: {expression} sources: {sources} destinations: "
            "{destinations} value: {value}").format(
                expression=str(self.expression),
                sources=str(self.sources),
                destinations=str(self.destinations),
                value=str(self.value))

  def _create_value_error(self) -> ValueError:
    """Creates a ValueError, assuming there should be one for this node."""
    return ValueError("Expression {} returned the wrong type:"
                      " expected: {}"
                      " actual: {}.".format(
                          self.expression,
                          _fancy_type_str(self.expression.is_repeated,
                                          self.expression.type),
                          _node_type_str(self.value)))

  def calculate(self,
                source_values: Sequence[prensor.NodeTensor],
                options: calculate_options.Options,
                side_info: Optional[prensor.Prensor]) -> None:
    """Calculate the value of the node, and store it in self.value."""
    self.value = self.expression.calculate(
        source_values, [x.expression for x in self.destinations],
        options,
        side_info=side_info)
    if self.value.is_repeated != self.expression.is_repeated:
      raise self._create_value_error()
    expected_type = self.expression.type
    actual_value = self.value
    if expected_type is None:
      if not (isinstance(actual_value, prensor.RootNodeTensor) or
              isinstance(actual_value, prensor.ChildNodeTensor)):
        raise self._create_value_error()
    elif isinstance(actual_value, prensor.LeafNodeTensor):
      if expected_type != actual_value.values.dtype:
        raise self._create_value_error()
    else:
      raise self._create_value_error()

  def __hash__(self) -> int:
    """This assumes all sources are canonical."""
    return hash(tuple([id(x) for x in self.sources]))


class ExpressionGraph(object):
  """A graph representing the computation of a list of expressions."""

  def __init__(self):
    self._node = {}  # type: Dict[IDExpression, _ExpressionNode]
    # An ordered list of nodes.
    self._ordered_node_list = []  # type: List[_ExpressionNode]

  @property
  def ordered_node_list(self):
    """Do not add or delete elements."""
    return self._ordered_node_list

  def _get_node(self, expr: expression.Expression) -> Optional[_ExpressionNode]:
    """Gets a node corresponding to the expression.

    Args:
      expr: the expression to look up.

    Returns:
      A node for an expression that returns the same values as the
      expression.
    """
    earliest = _get_earliest_equal_calculation(expr)
    return self._node.get(id(earliest))

  def get_value(self,
                expr: expression.Expression) -> Optional[prensor.NodeTensor]:
    node = self._get_node(expr)
    return None if node is None else node.value

  def get_value_or_die(self, expr: expression.Expression) -> prensor.NodeTensor:
    result = self.get_value(expr)
    if result is None:
      raise ValueError("Could not find expression's value")
    return result

  def _find_destinations(self, nodes: Sequence[_ExpressionNode]) -> None:
    """Initialize destinations of nodes.

    For all nodes u,v in nodes:
      if u is a source of v, make v a destination of u.

    Args:
      nodes: a list containing every node in the graph exactly once.
    """
    for node in nodes:
      for source in node.sources:
        source_node = self._get_node(source)
        if source_node is None:
          raise ValueError("Could not find source of node")
        source_node.destinations.append(node)

  def calculate_values(
      self,
      options: calculate_options.Options,
      feed_dict: Optional[Dict[expression.Expression, prensor.Prensor]] = None
  ) -> None:
    for node in self.ordered_node_list:
      source_values = [self._node[id(x)].value for x in node.sources]
      side_info = feed_dict[node.expression] if feed_dict and (
          node.expression in feed_dict) else None
      node.calculate(source_values, options, side_info=side_info)

  def get_expressions_needed(self) -> Sequence[expression.Expression]:
    return [x.expression for x in self.ordered_node_list]

  def __str__(self):
    return str([str(x) for x in self.ordered_node_list])


class OriginalExpressionGraph(ExpressionGraph):
  """A graph representing the computation of a list of expressions.

  This can directly consume a list of expressions, and create a graph.
  In terms of calculating the values, the primary purpose here is to enumerate
  all expressions.
  """

  def __init__(self, expressions: Sequence[expression.Expression]):
    super().__init__()
    self._add_expressions(expressions)
    original_nodes = list(self._node.values())
    self._find_destinations(original_nodes)
    self._order_nodes()

  def _add_expressions(self,
                       expressions: Sequence[expression.Expression]) -> None:
    """Add expressions to the graph."""
    if self.ordered_node_list:
      raise ValueError("Only call once during construction")
    to_add = list(expressions)
    while to_add:
      expr = to_add.pop()
      if self._get_node(expr) is None:
        earliest_expr = _get_earliest_equal_calculation(expr)
        node = _ExpressionNode(earliest_expr)
        self._node[id(earliest_expr)] = node
        to_add.extend(node.sources)

  def _order_nodes(self) -> None:
    """Topologically sorts the nodes and puts them in ordered_node_list."""
    nodes_to_process = []  # type: List[_ExpressionNode]
    if self.ordered_node_list:
      raise ValueError("Only call once during construction")
    expr_id_to_count = {
        expr_id: len(node.sources) for expr_id, node in self._node.items()
    }
    for k, v in sorted(expr_id_to_count.items()):
      if v == 0:
        nodes_to_process.append(self._node[k])
    while nodes_to_process:
      node = nodes_to_process.pop()
      self._ordered_node_list.append(node)
      for dest in node.destinations:
        count = expr_id_to_count[id(dest.expression)] - 1
        expr_id_to_count[id(dest.expression)] = count
        if count == 0:
          nodes_to_process.append(dest)


class CanonicalExpressionGraph(ExpressionGraph):
  """A graph representing the computation of a list of expressions.

  In the construction of the graph, if two expressions have inputs which can be
  proven to have equal outputs, and the two expressions have equal calculations,
  then they are represented by a single "canonical" node.

  This canonicalization is done sequentially through the ordered_node_list of
  the input graph (to avoid Python recursion limits).
  """

  def __init__(self, original: ExpressionGraph):
    super().__init__()
    # Nodes indexed by _ExpressionNode.
    self._node_map = {}  # type: Dict[_ExpressionNode, _ExpressionNode]
    self._add_expressions([x.expression for x in original.ordered_node_list])
    self._find_destinations(self.ordered_node_list)

  def _add_expressions(self,
                       expressions: Sequence[expression.Expression]) -> None:
    """Add expressions to the graph.

    Args:
      expressions: an ordered list, where sources are before expressions, and
        for all x in expressions, x == _get_earliest_equal_calculation(x)
    """
    for expr in expressions:
      maybe_node = self._create_node_if_not_exists(expr)
      if maybe_node is not None:
        self._ordered_node_list.append(maybe_node)

  def _create_node_if_not_exists(self, expr: expression.Expression
                                ) -> Optional[_ExpressionNode]:
    """Creates a canonical node for an expression if none exists.

    This method assumes that the method has already been called on all
    pre-existing nodes.

    After this method is called, self._get_canonical_or_error(expr) will
    succeed.

    Args:
      expr: an expression to be canonicalized.

    Returns:
      a new node, or None if the new node is not created.
    """
    if self._get_node(expr) is not None:
      return None
    maybe_canonical = _ExpressionNode(expr)
    # The sources are already the earliest equal calculation.
    maybe_canonical.sources = [
        self._get_canonical_or_error(x) for x in maybe_canonical.sources
    ]
    if maybe_canonical in self._node_map:
      self._node[id(expr)] = self._node_map[maybe_canonical]
      return None
    else:
      self._node_map[maybe_canonical] = maybe_canonical
      self._node[id(expr)] = maybe_canonical
      return maybe_canonical

  def _get_canonical_or_error(self, expr: expression.Expression
                             ) -> expression.Expression:
    """Gets a canonical expression or dies."""
    node = self._get_node(expr)
    if node is not None:
      return node.expression
    else:
      raise ValueError("Expression not found: " + str(expr))
