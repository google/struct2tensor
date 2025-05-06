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
"""project selects a subtree of an expression.

project is often used right before calculating the value.

Example:

```
expr = ...
new_expr = project.project(expr, [path.Path(["foo","bar"]),
                                  path.Path(["x", "y"])])
[prensor_result] = calculate.calculate_prensors([new_expr])
```

prensor_result now has two paths, "foo.bar" and "x.y".

"""

import collections
from typing import FrozenSet, List, Mapping, Optional, Sequence

from struct2tensor import calculate_options
from struct2tensor import expression
from struct2tensor import path
from struct2tensor import prensor


def project(expr: expression.Expression,
            paths: Sequence[path.Path]) -> expression.Expression:
  """select a subtree.

  Paths not selected are removed.
  Paths that are selected are "known", such that if calculate_prensors is
  called, they will be in the result.

  Args:
    expr: the original expression.
    paths: the paths to include.

  Returns:
    A projected expression.
  """
  missing_paths = [p for p in paths if expr.get_descendant(p) is None]
  if missing_paths:
    raise ValueError("{} Path(s) missing in project: {}".format(
        len(missing_paths), ", ".join([str(x) for x in missing_paths])))
  return _ProjectExpression(expr, paths)


def _group_paths_by_first_step(
    paths: Sequence[path.Path]) -> Mapping[path.Step, List[path.Path]]:
  result = collections.defaultdict(list)
  for p in paths:
    if p:
      first_step = p.field_list[0]
      result[first_step].append(p.suffix(1))
  return result


class _ProjectExpression(expression.Expression):
  """Project all subfields of an expression."""

  def __init__(
      self,
      origin: expression.Expression,
      paths: Sequence[path.Path],
  ):
    super().__init__(
        origin.is_repeated,
        origin.type,
        origin.schema_feature,
        validate_step_format=origin.validate_step_format,
    )
    self._paths_map = _group_paths_by_first_step(paths)
    self._origin = origin

  def get_source_expressions(self) -> Sequence[expression.Expression]:
    return [self._origin]

  def calculate(
      self,
      sources: Sequence[prensor.NodeTensor],
      destinations: Sequence[expression.Expression],
      options: calculate_options.Options,
      side_info: Optional[prensor.Prensor] = None) -> prensor.NodeTensor:
    if len(sources) != 1:
      raise ValueError("Expected one source.")
    return sources[0]

  def calculation_is_identity(self) -> bool:
    return True

  def calculation_equal(self, expr: expression.Expression) -> bool:
    return expr.calculation_is_identity()

  def _get_child_impl(self,
                      field_name: path.Step) -> Optional[expression.Expression]:
    paths = self._paths_map.get(field_name)
    if paths is None:
      return None
    original = self._origin.get_child(field_name)
    if original is None:
      raise ValueError(
          "Project a field that doesn't exist: {}".format(field_name))
    return _ProjectExpression(original, paths)

  def known_field_names(self) -> FrozenSet[path.Step]:
    return frozenset(self._paths_map.keys())
