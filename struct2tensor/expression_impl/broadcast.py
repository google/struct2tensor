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

r"""Methods for broadcasting a path in a tree.

This provides methods for broadcasting a field anonymously (that is used in
promote_and_broadcast), or with an explicitly given name.

Suppose you have an expr representing:

```
+
|
+-session*   (stars indicate repeated)
     |
     +-event*
     |
     +-val*-int64

session: {
  event: {}
  event: {}
  val: 10
  val: 11
}
session: {
  event: {}
  event: {}
  val: 20
}
```

Then:

```
broadcast.broadcast(expr, path.Path(["session","val"]), "event", "nv")
```

becomes:

```
+
|
+---session*   (stars indicate repeated)
       |
       +-event*
       |   |
       |   +---nv*-int64
       |
       +-val*-int64

session: {
  event: {
    nv: 10
    nv:11
  }
  event: {
    nv: 10
    nv:11
  }
  val: 10
  val: 11
}
session: {
  event: {nv: 20}
  event: {nv: 20}
  val: 20
}
```

"""

from typing import Optional, Sequence, Tuple

from struct2tensor import calculate_options
from struct2tensor import expression
from struct2tensor import expression_add
from struct2tensor import path
from struct2tensor import prensor
from struct2tensor.ops import struct2tensor_ops
import tensorflow as tf


class _BroadcastExpression(expression.Leaf):
  """A broadcast field.

  """

  def __init__(self, origin: expression.Expression,
               sibling: expression.Expression):
    super().__init__(origin.is_repeated, origin.type)
    if origin.type is None:
      raise ValueError("Can only broadcast a field")
    self._origin = origin
    self._sibling = sibling

  def get_source_expressions(self) -> Sequence[expression.Expression]:
    return [self._origin, self._sibling]

  def calculate(
      self,
      sources: Sequence[prensor.NodeTensor],
      destinations: Sequence[expression.Expression],
      options: calculate_options.Options,
      side_info: Optional[prensor.Prensor] = None) -> prensor.NodeTensor:
    [origin_value, sibling_value] = sources
    if not isinstance(origin_value, prensor.LeafNodeTensor):
      raise ValueError("origin not a LeafNodeTensor")
    if not isinstance(sibling_value, prensor.ChildNodeTensor):
      raise ValueError("sibling value is not a ChildNodeTensor")
    sibling_to_parent_index = sibling_value.parent_index
    # For each i, for each v, if there exist exactly n values j such that:
    # sibling_to_parent_index[i]==origin_value.parent_index[j]
    # then there exists exactly n values k such that:
    # new_parent_index[k] = i
    # new_values[k] = origin_value.values[j]
    # (Ordering is also preserved).
    [broadcasted_to_sibling_index, index_to_values
    ] = struct2tensor_ops.equi_join_indices(sibling_to_parent_index,
                                            origin_value.parent_index)
    new_values = tf.gather(origin_value.values, index_to_values)
    return prensor.LeafNodeTensor(broadcasted_to_sibling_index, new_values,
                                  self.is_repeated)

  def calculation_is_identity(self) -> bool:
    return False

  def calculation_equal(self, expr: expression.Expression) -> bool:
    return isinstance(expr, _BroadcastExpression)


def _broadcast_impl(
    root: expression.Expression, origin: path.Path, sibling: path.Step,
    new_field_name: path.Step) -> Tuple[expression.Expression, path.Path]:
  sibling_path = origin.get_parent().get_child(sibling)
  new_expr = _BroadcastExpression(
      root.get_descendant_or_error(origin),
      root.get_descendant_or_error(origin.get_parent().get_child(sibling)))
  new_path = sibling_path.get_child(new_field_name)
  return expression_add.add_paths(root, {new_path: new_expr}), new_path


def broadcast_anonymous(
    root: expression.Expression, origin: path.Path,
    sibling: path.Step) -> Tuple[expression.Expression, path.Path]:
  return _broadcast_impl(root, origin, sibling, path.get_anonymous_field())


def broadcast(root: expression.Expression, origin: path.Path,
              sibling_name: path.Step,
              new_field_name: path.Step) -> expression.Expression:
  return _broadcast_impl(root, origin, sibling_name, new_field_name)[0]
