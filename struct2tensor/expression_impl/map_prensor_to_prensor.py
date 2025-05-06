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
r"""Arbitrary operations from prensors to prensors in an expression.

This is useful if a single op generates an entire structure. In general, it is
better to use the existing expressions framework or design a custom expression
than use this op. So long as any of the output is required, all of the input
is required.

For example, suppose you have an op my_op, that takes a prensor of the form:

```
  event
   / \
 foo   bar
```

and produces a prensor of the form my_result_schema:

```
   event
    / \
 foo2 bar2
```

```
my_result_schema = create_schema(
    is_repeated=True,
    children={"foo2":{is_repeated:True, dtype:tf.int64},
              "bar2":{is_repeated:False, dtype:tf.int64}})
```

If you give it an expression original with the schema:

```
 session
    |
  event
  /  \
foo   bar

result = map_prensor_to_prensor(
  original,
  path.Path(["session","event"]),
  my_op,
  my_result_schema)
```

Result will have the schema:

```
 session
    |
  event--------
  /  \    \    \
foo   bar foo2 bar2
```

"""

from typing import Any, Callable, Dict, FrozenSet, Optional, Sequence, Union

from struct2tensor import calculate_options
from struct2tensor import expression
from struct2tensor import expression_add
from struct2tensor import path
from struct2tensor import prensor
import tensorflow as tf

from tensorflow_metadata.proto.v0 import schema_pb2


class Schema(object):
  """A finite schema for a prensor.

  Effectively, this stores everything for the prensor but the tensors
  themselves.

  Notice that this is slightly different than schema_pb2.Schema, although
  similar in nature. At present, there is no clear way to extract is_repeated
  and dtype from schema_pb2.Schema.

  See create_schema below for constructing a schema.

  Note that for LeafNodeTensor, dtype is not None.
  Also, for ChildNodeTensor and RootNodeTensor, dtype is None. However,
  a ChildNodeTensor or RootNodeTensor could be childless.
  """

  def __init__(self,
               is_repeated: bool = True,
               dtype: Optional[tf.DType] = None,
               schema_feature: Optional[schema_pb2.Feature] = None,
               children: Optional[Dict[path.Step, "Schema"]] = None):
    """Create a new Schema object.

    Args:
      is_repeated: is the root repeated?
      dtype: tf.dtype of the root if the root is a leaf, otherwise None.
      schema_feature: schema_pb2.Feature of the root (no struct_domain
        necessary)
      children: child schemas.
    """
    self._is_repeated = is_repeated
    self._type = dtype
    self._schema_feature = schema_feature
    self._children = children if children is not None else {}
    # Cannot have a type and children.
    assert (self._type is None or not self._children)

  @property
  def is_repeated(self) -> bool:
    return self._is_repeated

  @property
  def type(self) -> Optional[tf.DType]:
    return self._type

  @property
  def schema_feature(self) -> Optional[schema_pb2.Feature]:
    return self._schema_feature

  def get_child(self, key: path.Step):
    return self._children[key]

  def known_field_names(self) -> FrozenSet[path.Step]:
    return frozenset(self._children.keys())

  def __str__(self) -> str:
    return ("Schema(is_repeated:{is_repeated} type:{type}"
            " "
            "schema_feature:({schema_feature})"
            " "
            "children:{children})").format(
                is_repeated=self._is_repeated,
                type=self._type,
                schema_feature=self._schema_feature,
                children=self._children)


def create_schema(is_repeated: bool = True,
                  dtype: Optional[tf.DType] = None,
                  schema_feature: Optional[schema_pb2.Feature] = None,
                  children: Optional[Dict[path.Step, Any]] = None) -> Schema:
  """Create a schema recursively.

  Example:
  my_result_schema = create_schema(
    is_repeated=True,
    children={"foo2":{is_repeated=True, dtype=tf.int64},
              "bar2":{is_repeated=False, dtype=tf.int64}})

  Args:
    is_repeated: whether the root is repeated.
    dtype: the dtype of a leaf (None for non-leaves).
    schema_feature: the schema_pb2.Feature describing this expression. name and
      struct_domain need not be specified.
    children: the child schemas. Note that the value type of children is either
      a Schema or a dictionary of arguments to create_schema.

  Returns:
    a new Schema represented by the inputs.
  """
  children_dict = children or {}
  child_schemas = {
      k: _create_schema_helper(v) for k, v in children_dict.items()
  }
  return Schema(
      is_repeated=is_repeated,
      dtype=dtype,
      schema_feature=schema_feature,
      children=child_schemas)


def _create_schema_helper(my_input) -> Schema:
  """Helper for create_schema.

  If my_input is a Schema, it is just returned.

  Otherwise, my_input should be a dictionary of arguments to create_schema.

  Args:
    my_input: either a Schema or a dictionary which optionally has the keys
      is_repeated, dtype, and children.

  Returns:
    a Schema.
  """
  if isinstance(my_input, Schema):
    return my_input
  return create_schema(**my_input)


def map_prensor_to_prensor(
    root_expr: expression.Expression, source: path.Path,
    paths_needed: Sequence[path.Path],
    prensor_op: Callable[[prensor.Prensor], prensor.Prensor],
    output_schema: Schema) -> expression.Expression:
  r"""Maps an expression to a prensor, and merges that prensor.

  For example, suppose you have an op my_op, that takes a prensor of the form:

    event
     / \
   foo   bar

  and produces a prensor of the form my_result_schema:

     event
      / \
   foo2 bar2

  If you give it an expression original with the schema:

   session
      |
    event
    /  \
  foo   bar

  result = map_prensor_to_prensor(
    original,
    path.Path(["session","event"]),
    my_op,
    my_output_schema)

  Result will have the schema:

   session
      |
    event--------
    /  \    \    \
  foo   bar foo2 bar2

  Args:
    root_expr: the root expression
    source: the path where the prensor op is applied.
    paths_needed: the paths needed for the op.
    prensor_op: the prensor op
    output_schema: the output schema of the op.

  Returns:
    A new expression where the prensor is merged.
  """
  original_child = root_expr.get_descendant_or_error(source).project(
      paths_needed)
  prensor_child = _PrensorOpExpression(original_child, prensor_op,
                                       output_schema)
  paths_map = {
      source.get_child(k): prensor_child.get_child_or_error(k)
      for k in prensor_child.known_field_names()
  }
  result = expression_add.add_paths(root_expr, paths_map)
  return result


##################### Implementation Follows ###################################


def _tree_as_node(prensor_tree: prensor.Prensor) -> "_TreeAsNode":
  """Create a _TreeAsNode, a NodeTensor with a prensor property.

  The root node of the tree is pulled out, the resulting NodeTensor has the same
  properties.

  Args:
    prensor_tree: the original tree, that will be accessible via prensor.

  Returns:
    a _TreeAsNode x with x.prensor==prensor_tree, and x is equivalent
    to prensor_tree.node.
  """
  top_node = prensor_tree.node
  if isinstance(top_node, prensor.RootNodeTensor):
    return _PrensorAsRootNodeTensor(prensor_tree, top_node)
  if isinstance(top_node, prensor.ChildNodeTensor):
    return _PrensorAsChildNodeTensor(prensor_tree, top_node)
  return _PrensorAsLeafNodeTensor(prensor_tree, top_node)


class _PrensorAsRootNodeTensor(prensor.RootNodeTensor):
  """A root node tensor that has a prensor property."""

  def __init__(self, prensor_tree: prensor.Prensor,
               root: prensor.RootNodeTensor):
    """Call _tree_as_node instead."""
    super().__init__(root.size)
    self._prensor = prensor_tree

  @property
  def prensor(self):
    return self._prensor


class _PrensorAsChildNodeTensor(prensor.ChildNodeTensor):
  """A child node tensor that has a prensor property."""

  def __init__(self, prensor_tree: prensor.Prensor,
               child: prensor.ChildNodeTensor):
    """Call _tree_as_node instead."""
    super().__init__(child.parent_index, child.is_repeated)
    self._prensor = prensor_tree

  @property
  def prensor(self):
    return self._prensor


class _PrensorAsLeafNodeTensor(prensor.LeafNodeTensor):
  """A leaf node tensor that has a prensor property."""

  def __init__(self, prensor_tree: prensor.Prensor,
               leaf: prensor.LeafNodeTensor):
    """Call _tree_as_node instead."""
    super(_PrensorAsLeafNodeTensor,
          self).__init__(leaf.parent_index, leaf.values, leaf.is_repeated)
    self._prensor = prensor_tree

  @property
  def prensor(self):
    return self._prensor


_TreeAsNode = Union[_PrensorAsLeafNodeTensor, _PrensorAsChildNodeTensor,
                    _PrensorAsRootNodeTensor]


def _get_schema_or_error(parent: expression.Expression,
                         step: path.Step) -> Schema:
  if not isinstance(parent, (_PrensorOpExpression, _PrensorOpChildExpression)):
    raise ValueError("No parent schema")
  parent_schema = parent.schema
  return parent_schema.get_child(step)


class _PrensorOpChildExpression(expression.Expression):
  """A helper class for PrensorOpExpression, representing its descendants."""

  def __init__(self, parent: expression.Expression, step: path.Step,
               schema: Schema):
    super().__init__(
        schema.is_repeated,
        schema.type,
        schema_feature=schema.schema_feature,
        validate_step_format=parent.validate_step_format,
    )
    self._parent = parent
    self._step = step
    self._schema = schema

  @property
  def schema(self):
    return self._schema

  def get_source_expressions(self) -> Sequence[expression.Expression]:
    return [self._parent]

  def calculate(self,
                source_tensors: Sequence[prensor.NodeTensor],
                destinations: Sequence[expression.Expression],
                options: calculate_options.Options,
                side_info: Optional[prensor.Prensor] = None) -> _TreeAsNode:
    [parent_result] = source_tensors
    if not isinstance(parent_result,
                      (_PrensorAsLeafNodeTensor, _PrensorAsChildNodeTensor,
                       _PrensorAsRootNodeTensor)):
      raise ValueError("Parent did not return _TreeAsNode")
    my_prensor = parent_result.prensor.get_child(self._step)
    return _tree_as_node(my_prensor)

  def calculation_is_identity(self) -> bool:
    return False

  def calculation_equal(self, expr: expression.Expression) -> bool:
    if isinstance(expr, _PrensorOpChildExpression):
      return self._step == expr._step  # pylint: disable=protected-access
    return False

  def _get_child_impl(self,
                      field_name: path.Step) -> Optional[expression.Expression]:
    """Implementation of getting a named child in a subclass.

    This is called and cached inside get_child().

    Args:
      field_name: the field accessed.

    Returns:
      The expression of the field, or None if the field does not exist.
    """
    if field_name not in self._schema.known_field_names():
      return None
    child_schema = self._schema.get_child(field_name)
    return _PrensorOpChildExpression(self, field_name, child_schema)

  def known_field_names(self) -> FrozenSet[path.Step]:
    return self._schema.known_field_names()


class _PrensorOpExpression(expression.Expression):
  """An expression generated by a callable that returns a prensor.

  Note that it is expected that project() is called on the origin before
  this method, as is generally advised before calling get_known_descendants().
  """

  def __init__(self, origin: expression.Expression,
               operation: Callable[[prensor.Prensor], prensor.Prensor],
               schema: Schema):
    """Creates a new expression.

    Conceptually, this expression = operation applied to origin, where the
    result is tagged with schema.

    Args:
      origin: the expression which is used to generate the input prensor.
      operation: the operation applied to the origin.
      schema: the schema of the result. If a path is not in the schema, it is
        not calculated.
    """
    super().__init__(
        schema.is_repeated,
        schema.type,
        schema_feature=schema.schema_feature,
        validate_step_format=origin.validate_step_format,
    )

    self._origin = origin
    self._operation = operation
    self._schema = schema

  @property
  def schema(self):
    return self._schema

  def _get_source_paths(self) -> Sequence[path.Path]:
    """Returns the source paths in a deterministic order."""
    result = [k for k in self._origin.get_known_descendants().keys()]
    # In order to make certain that the source_paths are in a deterministic
    # order, we sort them here.
    result.sort()
    return result

  def get_source_expressions(self) -> Sequence[expression.Expression]:
    subtree = self._origin.get_known_descendants()
    source_paths = self._get_source_paths()
    return [subtree[k] for k in source_paths]

  def calculate(self,
                sources: Sequence[prensor.NodeTensor],
                destinations: Sequence[expression.Expression],
                options: calculate_options.Options,
                side_info: Optional[prensor.Prensor] = None) -> _TreeAsNode:
    source_tree = prensor.create_prensor_from_descendant_nodes(
        {k: v for k, v in zip(self._get_source_paths(), sources)})
    result_tree = self._operation(source_tree)
    # TODO(martinz): consider a full type check on result_tree. This
    # can be done outside of the GraphDef.
    return _tree_as_node(result_tree)

  def calculation_is_identity(self) -> bool:
    return False

  def calculation_equal(self, expr: expression.Expression) -> bool:
    return self is expr

  def _get_child_impl(self,
                      field_name: path.Step) -> Optional[expression.Expression]:
    if field_name not in self._schema.known_field_names():
      return None
    child_schema = self._schema.get_child(field_name)
    return _PrensorOpChildExpression(self, field_name, child_schema)

  def known_field_names(self) -> FrozenSet[path.Step]:
    return self._schema.known_field_names()
