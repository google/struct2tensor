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
"""Expressions for adding fields.

add_paths: an unsafe but powerful way of adding fields. If you are writing a
  new Expression subclass, having a single factory method that calls this
  function is useful.

add_to: a way to copy fields back to an earlier tree.

"""

from typing import FrozenSet, Mapping, Optional, Sequence, Tuple

from struct2tensor import expression
from struct2tensor import path
from struct2tensor import prensor
from struct2tensor.calculate_options import Options


def create_subtrees(
    path_map: Mapping[path.Path, expression.Expression]
) -> Tuple[Optional[expression.Expression],
           Mapping[path.Step, Mapping[path.Path, expression.Expression]]]:
  """Breaks a tree into a root node and dictionary of subtrees."""
  subtrees = {
  }  # type:Mapping[path.Step, Mapping[path.Path, expression.Expression]]
  root_expression = None  # type:Optional[expression.Expression]
  for k, v in path_map.items():
    if not k:
      root_expression = v
    else:
      first_step = k.field_list[0]
      suffix = k.suffix(1)
      if first_step not in subtrees:
        subtrees[first_step] = {}  # pytype: disable=unsupported-operands
      subtrees[first_step][suffix] = v  # pytype: disable=unsupported-operands
  return (root_expression, subtrees)


class _AddPathsExpression(expression.Expression):
  """An expression that is the result of add_paths, adding new fields.

  _AddPathsExpression is an overlay on top of the original expression with
  specified paths being added.
  """

  def __init__(
      self, origin: expression.Expression,
      path_map: Mapping[path.Step, Mapping[path.Path, expression.Expression]]):
    super().__init__(
        origin.is_repeated,
        origin.type,
        schema_feature=origin.schema_feature,
        validate_step_format=origin.validate_step_format,
    )
    self._origin = origin
    self._path_map = path_map

  def get_source_expressions(self) -> Sequence[expression.Expression]:
    return [self._origin]

  def calculate(self, source_values: Sequence[prensor.NodeTensor],
                destinations: Sequence[expression.Expression],
                options: Options) -> prensor.NodeTensor:
    if len(source_values) != 1:
      raise ValueError("Expected one source.")
    return source_values[0]

  def calculation_is_identity(self) -> bool:
    return True

  def calculation_equal(self, expr: expression.Expression) -> bool:
    return expr.calculation_is_identity()

  def _get_child_impl(self,
                      field_name: path.Step) -> Optional[expression.Expression]:
    child_from_origin = self._origin.get_child(field_name)
    path_map = self._path_map.get(field_name)
    if path_map is None:
      return child_from_origin
    set_root_expr, subtrees = create_subtrees(path_map)
    if child_from_origin is None:
      if set_root_expr is None:
        raise ValueError("Must have a value in the original if there are paths")
      if subtrees:
        return _AddPathsExpression(set_root_expr, subtrees)
      return set_root_expr
    if set_root_expr is not None:
      raise ValueError("Tried to overwrite an existing expression")
    return _AddPathsExpression(child_from_origin, subtrees)

  def known_field_names(self) -> FrozenSet[path.Step]:
    return self._origin.known_field_names().union(self._path_map.keys())

  def __str__(self) -> str:
    keys_to_add = ",".join([str(k) for k in self._path_map.keys()])
    return "_AddPathsExpression({}, [{}])".format(
        str(self._origin), keys_to_add)


def add_paths(root: expression.Expression,
              path_map: Mapping[path.Path, expression.Expression]):
  """Creates a new expression based on `root` with paths in `path_map` added.

  This operation should be used with care: e.g., there is no guarantee that
  the new expressions make any sense in the place they are put in the tree. It
  is useful when wrapping a new Expression type, but should not be used by
  end users.

  Prefer add_to to add_paths.

  Args:
    root: the root of the tree.
    path_map: a map from a path to the new subtree.

  Returns:
    a new tree with the nodes from the root and the new subtrees.
  """
  for p in path_map.keys():
    if root.get_descendant(p.get_parent()) is None:
      raise ValueError("No parent of {}".format(p))
    if root.get_descendant(p) is not None:
      raise ValueError("Path already set: {}".format(str(p)))
  _, map_of_maps = create_subtrees(path_map)
  return _AddPathsExpression(root, map_of_maps)


def _is_true_source_expression(candidate: expression.Expression,
                               dest: expression.Expression) -> bool:
  """True if dest is an expression derived from candidate through add_paths.

  More precisely, true if dest is the result of zero or more add_paths
  operations on candidate, where the (source of)* dest is candidate.

  A "true source" for a node dest has an identical NodeTensor to dest, and the
  parent of "true source" is a "true source" for the dest. This means if I have
  a child X of dest, I could make a new tree where X is a child of the source.

  See add_to.

  Args:
    candidate: the possible source
    dest: the possible destination.

  Returns:
    True iff source is a true source expression of dest.
  """
  if dest is candidate:
    return True
  if isinstance(dest, _AddPathsExpression):
    return _is_true_source_expression(candidate, dest._origin)  # pylint: disable=protected-access
  return False


def add_to(root: expression.Expression,
           origins: Mapping[path.Path, expression.Expression]):
  """Copies subtrees from the origins to the root.

  This operation can be used to reduce the number of expressions in the graph.
  1. The path must be present in the associated origin.
  2. The root must not have any of the paths already.
  3. The root must already have the parent of the path.
  4. The parent of the path in the root must be a source expression of the
     parent of the path in the origin.

  Args:
    root: the original tree that has new expressions added to it.
    origins: mapping from path to trees that have subtrees at the path.

  Returns:
    A tree with the root and the additional subtrees.
  """
  for p, origin_root in origins.items():
    path_parent = p.get_parent()
    if not _is_true_source_expression(
        root.get_descendant_or_error(path_parent),
        origin_root.get_descendant_or_error(path_parent)):
      raise ValueError("Not a true source for tree with {}".format(str(p)))
    if root.get_descendant(p) is not None:
      raise ValueError("Already contains {}.".format(str(p)))
  path_map = {
      p: origin_root.get_descendant_or_error(p)
      for p, origin_root in origins.items()
  }
  return add_paths(root, path_map)
