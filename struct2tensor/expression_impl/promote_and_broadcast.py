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
r"""promote_and_broadcast a set of nodes.

For example, suppose an expr represents:

```
+
|
+-session*   (stars indicate repeated)
     |
     +-event*
     |   |
     |   +-val*-int64
     |
     +-user_info? (question mark indicates optional)
           |
           +-age? int64

session: {
  event: {
    val: 1
  }
  event: {
    val: 4
    val: 5
  }
  user_info: {
    age: 25
  }
}

session: {
  event: {
    val: 7
  }
  event: {
    val: 8
    val: 9
  }
  user_info: {
    age: 20
  }
}
```

```
promote_and_broadcast.promote_and_broadcast(
    path.Path(["event"]),{"nage":path.Path(["user_info","age"])})
```

creates:

```
+
|
+-session*   (stars indicate repeated)
     |
     +-event*
     |   |
     |   +-val*-int64
     |   |
     |   +-nage*-int64
     |
     +-user_info? (question mark indicates optional)
           |
           +-age? int64

session: {
  event: {
    nage: 25
    val: 1
  }
  event: {
    nage: 25
    val: 4
    val: 5
  }
  user_info: {
    age: 25
  }
}

session: {
  event: {
    nage: 20
    val: 7
  }
  event: {
    nage: 20
    val: 8
    val: 9
  }
  user_info: {
    age: 20
  }
}
```

"""

from typing import Mapping, Tuple

from struct2tensor import expression
from struct2tensor import expression_add
from struct2tensor import path
from struct2tensor.expression_impl import broadcast
from struct2tensor.expression_impl import promote


def promote_and_broadcast_anonymous(
    root: expression.Expression, origin: path.Path,
    new_parent: path.Path) -> Tuple[expression.Expression, path.Path]:
  """Promotes then broadcasts the origin until its parent is new_parent."""
  least_common_ancestor = origin.get_least_common_ancestor(new_parent)

  new_expr, new_path = root, origin
  while new_path.get_parent() != least_common_ancestor:
    new_expr, new_path = promote.promote_anonymous(new_expr, new_path)

  while new_path.get_parent() != new_parent:
    new_parent_step = new_parent.field_list[len(new_path) - 1]
    new_expr, new_path = broadcast.broadcast_anonymous(new_expr, new_path,
                                                       new_parent_step)

  return new_expr, new_path


def _promote_and_broadcast_name(root: expression.Expression, origin: path.Path,
                                dest_path_parent: path.Path,
                                field_name: path.Step) -> expression.Expression:
  new_root, anonymous_path = promote_and_broadcast_anonymous(
      root, origin, dest_path_parent)
  path_result = dest_path_parent.get_child(field_name)
  return expression_add.add_paths(
      new_root, {path_result: new_root.get_descendant_or_error(anonymous_path)})


def promote_and_broadcast(root: expression.Expression,
                          path_dictionary: Mapping[path.Step, path.Path],
                          dest_path_parent: path.Path) -> expression.Expression:
  """Promote and broadcast a set of paths to a particular location.

  Args:
    root: the original expression.
    path_dictionary: a map from destination fields to origin paths.
    dest_path_parent: a map from destination strings.

  Returns:
    A new expression, where all the origin paths are promoted and broadcast
    until they are children of dest_path_parent.
  """

  result_paths = {}
  # Here, we branch out and create a different tree for each field that is
  # promoted and broadcast.
  for field_name, origin_path in path_dictionary.items():
    result_path = dest_path_parent.get_child(field_name)
    new_root = _promote_and_broadcast_name(root, origin_path, dest_path_parent,
                                           field_name)
    result_paths[result_path] = new_root
  # We create a new tree that has all of the generated fields from the older
  # trees.
  return expression_add.add_to(root, result_paths)
