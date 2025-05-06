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
"""Expressions to parse a proto.

These expressions return values with more information than standard node values.
Specifically, each node calculates additional tensors that are used as inputs
for its children.
"""

import abc
from typing import Callable, FrozenSet, Mapping, Optional, Sequence, Set, Tuple, Union, cast

from struct2tensor import calculate_options
from struct2tensor import expression
from struct2tensor import expression_add
from struct2tensor import path
from struct2tensor import prensor
from struct2tensor.expression_impl import parse_message_level_ex
from struct2tensor.ops import struct2tensor_ops
import tensorflow as tf

from google.protobuf import descriptor_pb2
from google.protobuf import descriptor
from google.protobuf.descriptor_pool import DescriptorPool

# To the best of my knowledge, ProtoFieldNames ARE strings.
# Also includes extensions, encoded in a parentheses like (foo.bar.Baz).
ProtoFieldName = str
ProtoFullName = str

# A string representing a step in a path.
StrStep = str


def is_proto_expression(expr: expression.Expression) -> bool:
  """Returns true if an expression is a ProtoExpression."""
  return isinstance(
      expr, (_ProtoRootExpression, _ProtoChildExpression, _ProtoLeafExpression))


def create_expression_from_file_descriptor_set(
    tensor_of_protos: tf.Tensor,
    proto_name: ProtoFullName,
    file_descriptor_set: descriptor_pb2.FileDescriptorSet,
    message_format: str = "binary") -> expression.Expression:
  """Create an expression from a 1D tensor of serialized protos.

  Args:
    tensor_of_protos: 1D tensor of serialized protos.
    proto_name: fully qualified name (e.g. "some.package.SomeProto") of the
      proto in `tensor_of_protos`.
    file_descriptor_set: The FileDescriptorSet proto containing `proto_name`'s
      and all its dependencies' FileDescriptorProto. Note that if file1 imports
      file2, then file2's FileDescriptorProto must precede file1's in
      file_descriptor_set.file.
    message_format: Indicates the format of the protocol buffer: is one of
       'text' or 'binary'.

  Returns:
    An expression.
  """

  pool = DescriptorPool()
  for f in file_descriptor_set.file:
    # This method raises if f's dependencies have not been added.
    pool.Add(f)

  # This method raises if proto not found.
  desc = pool.FindMessageTypeByName(proto_name)

  return create_expression_from_proto(tensor_of_protos, desc, message_format)


def create_expression_from_proto(
    tensor_of_protos: tf.Tensor,
    desc: descriptor.Descriptor,
    message_format: str = "binary") -> expression.Expression:
  """Create an expression from a 1D tensor of serialized protos.

  Args:
    tensor_of_protos: 1D tensor of serialized protos.
    desc: a descriptor of protos in tensor of protos.
    message_format: Indicates the format of the protocol buffer: is one of
      'text' or 'binary'.

  Returns:
    An expression.
  """
  return _ProtoRootExpression(desc, tensor_of_protos, message_format)


# The function signature expected by `created_transformed_field`.
# It describes functions of the form:
#
# def transform_fn(parent_indices, values):
#   ...
#   return (transformed_parent_indices, transformed_values).
#
# Where values/transformed_values are serialized protos of the same type
# and parent_indices/transformed_parent_indices are non-decreasing int64
# vectors.  Each pair of indices and values must have the same shape.
TransformFn = Callable[[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]


def create_transformed_field(
    expr: expression.Expression, source_path: path.CoercableToPath,
    dest_field: StrStep, transform_fn: TransformFn) -> expression.Expression:
  """Create an expression that transforms serialized proto tensors.

  The transform_fn argument should take the form:

  def transform_fn(parent_indices, values):
    ...
    return (transformed_parent_indices, transformed_values)

  Given:
  - parent_indices: an int64 vector of non-decreasing parent message indices.
  - values: a string vector of serialized protos having the same shape as
    `parent_indices`.
  `transform_fn` must return new parent indices and serialized values encoding
  the same proto message as the passed in `values`.  These two vectors must
  have the same size, but it need not be the same as the input arguments.

  Note:
    If CalculateOptions.use_string_view (set at calculate time, thus this
    Expression cannot know beforehand) is True, `values` passed to
    `transform_fn` are string views pointing all the way back to the original
    input tensor (of serialized root protos). And `transform_fn` must maintain
    such views and avoid creating new values that are either not string views
    into the root protos or self-owned strings. This is because downstream
    decoding ops will still produce string views referring into its input
    (which are string views into the root proto) and they will only hold a
    reference to the original, root proto tensor, keeping it alive. So the input
    tensor may get destroyed after the decoding op.

    In short, you can do element-wise transforms to `values`, but can't mutate
    the contents of elements in `values` or create new elements.

    To lift this restriction, a decoding op must be told to hold a reference
    of the input tensors of all its upstream decoding ops.


  Args:
    expr: a source expression containing `source_path`.
    source_path: the path to the field to reverse.
    dest_field: the name of the newly created field. This field will be a
      sibling of the field identified by `source_path`.
    transform_fn: a callable that accepts parent_indices and serialized proto
      values and returns a posibly modified parent_indices and values. Note that
      when CalcuateOptions.use_string_view is set, transform_fn should not have
      any stateful side effecting uses of serialized proto inputs. Doing so
      could cause segfaults as the backing string tensor lifetime is not
      guaranteed when the side effecting operations are run.

  Returns:
    An expression.

  Raises:
    ValueError: if the source path is not a proto message field.
  """
  source_path = path.create_path(source_path)
  source_expr = expr.get_descendant_or_error(source_path)
  if not isinstance(source_expr, _ProtoChildExpression):
    raise ValueError(
        "Expected _ProtoChildExpression for field {}, but found {}.".format(
            str(source_path), source_expr))

  if isinstance(source_expr, _TransformProtoChildExpression):
    # In order to be able to propagate fields needed for parsing, the source
    # expression of _TransformProtoChildExpression must always be the original
    # _ProtoChildExpression before any transformation. This means that two
    # sequentially applied _TransformProtoChildExpression would have the same
    # source and would apply the transformation to the source directly, instead
    # of one transform operating on the output of the other.
    # To work around this, the user supplied transform function is wrapped to
    # first call the source's transform function.
    # The downside of this approach is that the initial transform may be
    # applied redundantly if there are other expressions derived directly
    # from it.
    def final_transform(parent_indices: tf.Tensor,
                        values: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
      parent_indices, values = source_expr.transform_fn(parent_indices, values)
      return transform_fn(parent_indices, values)
  else:
    final_transform = transform_fn

  transformed_expr = _TransformProtoChildExpression(
      parent=source_expr._parent,  # pylint: disable=protected-access
      desc=source_expr._desc,  # pylint: disable=protected-access
      is_repeated=source_expr.is_repeated,
      name_as_field=source_expr.name_as_field,
      transform_fn=final_transform,
      backing_str_tensor=source_expr._backing_str_tensor)  # pylint: disable=protected-access
  dest_path = source_path.get_parent().get_child(dest_field)
  return expression_add.add_paths(expr, {dest_path: transformed_expr})


class _ProtoRootNodeTensor(prensor.RootNodeTensor):
  """The value of the root node.

  This not only contains the normal size information, but also information
  needed by its children.

  In particular:
  1. Any needed regular fields are included.
  2. Any needed extended fields are included.
  3. Any needed map fields are included.
  4. if this is an Any proto, any needed casted fields are included.

  """

  def __init__(self, size: tf.Tensor,
               fields: Mapping[StrStep, struct2tensor_ops._ParsedField]):
    super().__init__(size)
    self.fields = fields


class _ProtoChildNodeTensor(prensor.ChildNodeTensor):
  """The value of a child node.

  This not only contains the normal parent_index information, but also
  information needed by its children.

  In particular:
  1. Any needed regular fields are included.
  2. Any needed extended fields are included.
  3. Any needed map fields are included.
  4. if this is an Any proto, any needed casted fields are included.
  """

  def __init__(self, parent_index: tf.Tensor, is_repeated: bool,
               fields: Mapping[StrStep, struct2tensor_ops._ParsedField]):
    super().__init__(parent_index, is_repeated)
    self.fields = fields


_ParentProtoNodeTensor = Union[_ProtoRootNodeTensor, _ProtoChildNodeTensor]


class _AbstractProtoChildExpression(expression.Expression):
  """A child or leaf proto expression."""

  def __init__(self, parent: "_ParentProtoExpression", name_as_field: StrStep,
               is_repeated: bool, my_type: Optional[tf.DType],
               backing_str_tensor: Optional[tf.Tensor]):
    super().__init__(is_repeated, my_type)
    self._parent = parent
    self._name_as_field = name_as_field

    self._backing_str_tensor = backing_str_tensor

  @property
  def name_as_field(self) -> StrStep:
    return self._name_as_field

  def get_needed_fields(self, expr: expression.Expression) -> Sequence[StrStep]:
    return [self._name_as_field]

  def get_path(self) -> path.Path:
    """Returns the path to the root of the proto."""
    return self._parent.get_path().get_child(self.name_as_field)

  def get_proto_source(self) -> Tuple[tf.Tensor, descriptor.Descriptor]:
    """Returns the proto root."""
    return self._parent.get_proto_source()

  def get_source_expressions(self) -> Sequence[expression.Expression]:
    # In order to parse this proto, you need to parse its parent.
    return [self._parent]

  def calculate(
      self,
      sources: Sequence[prensor.NodeTensor],
      destinations: Sequence[expression.Expression],
      options: calculate_options.Options,
      side_info: Optional[prensor.Prensor] = None) -> prensor.NodeTensor:
    [parent_value] = sources
    if isinstance(parent_value, _ProtoRootNodeTensor) or isinstance(
        parent_value, _ProtoChildNodeTensor):
      parsed_field = parent_value.fields.get(self.name_as_field)
      if parsed_field is None:
        raise ValueError("Cannot find {} in {}".format(
            str(self), str(parent_value)))
      return self.calculate_from_parsed_field(parsed_field, destinations,
                                              options)
    raise ValueError("Not a _ParentProtoNodeTensor: " + str(type(parent_value)))

  @abc.abstractmethod
  def calculate_from_parsed_field(
      self, parsed_field: struct2tensor_ops._ParsedField,  # pylint: disable=protected-access
      destinations: Sequence[expression.Expression],
      options: calculate_options.Options) -> prensor.NodeTensor:
    """Calculate the NodeTensor given the parsed fields requested from a parent.

    Args:
      parsed_field: the parsed field from name_as_field.
      destinations: the destination of the expression.
      options: calculate options.

    Returns:
      A node tensor for this node.
    """
    raise NotImplementedError()

  def calculation_is_identity(self) -> bool:
    return False


class _ProtoLeafExpression(_AbstractProtoChildExpression):
  """Represents parsing a leaf field."""

  def __init__(self, parent: "_ParentProtoExpression",
               desc: descriptor.FieldDescriptor, name_as_field: path.Step):
    """Initialize a proto leaf expression.

    Args:
      parent: the parent of the expression.
      desc: the field descriptor of the expression name_as_field.
      name_as_field: the name of the field.
    """
    super().__init__(parent, name_as_field,
                     desc.label == descriptor.FieldDescriptor.LABEL_REPEATED,
                     struct2tensor_ops._get_dtype_from_cpp_type(desc.cpp_type),
                     None)  # pylint: disable=protected-access
    # TODO(martinz): make _get_dtype_from_cpp_type public.
    self._field_descriptor = desc

  def calculate_from_parsed_field(
      self, parsed_field: struct2tensor_ops._ParsedField,  # pylint: disable=protected-access
      destinations: Sequence[expression.Expression],
      options: calculate_options.Options) -> prensor.NodeTensor:
    return prensor.LeafNodeTensor(parsed_field.index, parsed_field.value,
                                  self.is_repeated)

  def calculation_equal(self, expr: expression.Expression) -> bool:
    # pylint: disable=protected-access
    return (isinstance(expr, _ProtoLeafExpression) and
            self._field_descriptor == expr._field_descriptor and
            self.name_as_field == expr.name_as_field)

  def _get_child_impl(self,
                      field_name: path.Step) -> Optional[expression.Expression]:
    return None

  def known_field_names(self) -> FrozenSet[path.Step]:
    return frozenset()

  def __str__(self) -> str:
    return "_ProtoLeafExpression: {} from {}".format(self.name_as_field,
                                                     self._parent)


class _ProtoChildExpression(_AbstractProtoChildExpression):
  """An expression representing a proto submessage.

  Supports:
    A standard submessage.
    An extension submessage.
    A protobuf.Any submessage.
    A proto map submessage.
    Also supports having fields of the above types.
  """

  def __init__(self, parent: "_ParentProtoExpression",
               desc: descriptor.Descriptor, is_repeated: bool,
               name_as_field: StrStep, backing_str_tensor: Optional[tf.Tensor]):
    """Initialize a _ProtoChildExpression.

    This does not take a field descriptor so it can represent syntactic sugar
    fields such as Any and Maps.
    Args:
      parent: the parent.
      desc: the message descriptor of the submessage represented by this
        expression.
      is_repeated: whether the field is repeated.
      name_as_field: the name of the field.
      backing_str_tensor: a string tensor representing the root serialized
        proto. This is passed to keep string_views of the tensor valid for
        all children of the root expression
    """
    super().__init__(parent, name_as_field, is_repeated, None,
                     backing_str_tensor)
    self._desc = desc

  def calculate_from_parsed_field(
      self, parsed_field: struct2tensor_ops._ParsedField,  # pylint:disable=protected-access
      destinations: Sequence[expression.Expression],
      options: calculate_options.Options) -> prensor.NodeTensor:
    needed_fields = _get_needed_fields(destinations)
    backing_str_tensor = None
    if options.use_string_view:
      backing_str_tensor = self._backing_str_tensor
    fields = parse_message_level_ex.parse_message_level_ex(
        parsed_field.value,
        self._desc,
        needed_fields,
        backing_str_tensor=backing_str_tensor,
        honor_proto3_optional_semantics=options
        .experimental_honor_proto3_optional_semantics)
    return _ProtoChildNodeTensor(parsed_field.index, self.is_repeated, fields)

  def calculation_equal(self, expr: expression.Expression) -> bool:
    # Ensure that we're dealing with the _ProtoChildExpression and not any
    # of its subclasses.
    if type(expr) != _ProtoChildExpression:  # pylint: disable=unidiomatic-typecheck
      return False
    expr = cast(_ProtoChildExpression, expr)  # Keep pytype happy.
    return (self._desc == expr._desc and  # pylint: disable=protected-access
            self.name_as_field == expr.name_as_field)

  def _get_child_impl(self,
                      field_name: path.Step) -> Optional[expression.Expression]:
    return _get_child(self, self._desc, field_name, self._backing_str_tensor)

  def known_field_names(self) -> FrozenSet[path.Step]:
    return _known_field_names_from_descriptor(self._desc)

  def __str__(self) -> str:
    return "_ProtoChildExpression: name_as_field: {} desc: {} from {}".format(
        str(self.name_as_field), str(self._desc.full_name), self._parent)


class _TransformProtoChildExpression(_ProtoChildExpression):
  """Transforms the parent indices and values prior to parsing."""

  def __init__(self, parent: "_ParentProtoExpression",
               desc: descriptor.Descriptor, is_repeated: bool,
               name_as_field: StrStep, transform_fn: TransformFn,
               backing_str_tensor: Optional[tf.Tensor]):
    super(_TransformProtoChildExpression,
          self).__init__(parent, desc, is_repeated, name_as_field,
                         backing_str_tensor)
    self._transform_fn = transform_fn

  @property
  def transform_fn(self):
    return self._transform_fn

  def calculate_from_parsed_field(
      self,
      parsed_field: struct2tensor_ops._ParsedField,  # pylint:disable=protected-access
      destinations: Sequence[expression.Expression],
      options: calculate_options.Options) -> prensor.NodeTensor:
    needed_fields = _get_needed_fields(destinations)
    transformed_parent_indices, transformed_values = self._transform_fn(
        parsed_field.index, parsed_field.value)
    backing_str_tensor = None
    if options.use_string_view:
      backing_str_tensor = self._backing_str_tensor
    fields = parse_message_level_ex.parse_message_level_ex(
        transformed_values,
        self._desc,
        needed_fields,
        backing_str_tensor=backing_str_tensor,
        honor_proto3_optional_semantics=options
        .experimental_honor_proto3_optional_semantics)
    return _ProtoChildNodeTensor(transformed_parent_indices, self.is_repeated,
                                 fields)

  def calculation_equal(self, expr: expression.Expression) -> bool:
    return (isinstance(expr, _TransformProtoChildExpression) and
            self._desc == expr._desc and  # pylint: disable=protected-access
            self.name_as_field == expr.name_as_field
            and self.transform_fn is expr.transform_fn)

  def __str__(self) -> str:
    return ("_TransformProtoChildExpression: name_as_field: {} desc: {} from {}"
            .format(
                str(self.name_as_field), str(self._desc.full_name),
                self._parent))


class _ProtoRootExpression(expression.Expression):
  """The expression representing the parse of the root of a proto.

  This class returns a _ProtoRootNodeTensor, that parses out fields for
  _ProtoChildExpression and _ProtoLeafExpression to consume.
  """

  def __init__(self,
               desc: descriptor.Descriptor,
               tensor_of_protos: tf.Tensor,
               message_format: str = "binary"):
    """Initialize a proto expression.

    Args:
      desc: the descriptor of the expression.
      tensor_of_protos: a 1-D tensor to get the protos from.
      message_format: Indicates the format of the protocol buffer: is one of
       'text' or 'binary'.
    """
    super().__init__(True, None)
    self._descriptor = desc
    self._tensor_of_protos = tensor_of_protos
    self._message_format = message_format

  def get_path(self) -> path.Path:
    """Returns the path to the root of the proto."""
    return path.Path([])

  def get_proto_source(self) -> Tuple[tf.Tensor, descriptor.Descriptor]:
    """Returns the tensor of protos and the original descriptor."""
    return (self._tensor_of_protos, self._descriptor)

  def get_source_expressions(self) -> Sequence[expression.Expression]:
    return []

  def calculate(
      self,
      sources: Sequence[prensor.NodeTensor],
      destinations: Sequence[expression.Expression],
      options: calculate_options.Options,
      side_info: Optional[prensor.Prensor] = None) -> _ProtoRootNodeTensor:
    if sources:
      raise ValueError("_ProtoRootExpression has no sources")
    size = tf.size(self._tensor_of_protos, out_type=tf.int64)
    needed_fields = _get_needed_fields(destinations)
    backing_str_tensor = None
    if options.use_string_view:
      assert self._message_format == "binary", (
          "`options.use_string_view` is only compatible with 'binary' message "
          "format. Please create the root expression with "
          "message_format='binary'.")
      backing_str_tensor = self._tensor_of_protos
    fields = parse_message_level_ex.parse_message_level_ex(
        self._tensor_of_protos,
        self._descriptor,
        needed_fields,
        message_format=self._message_format,
        backing_str_tensor=backing_str_tensor,
        honor_proto3_optional_semantics=options
        .experimental_honor_proto3_optional_semantics)
    return _ProtoRootNodeTensor(size, fields)

  def calculation_is_identity(self) -> bool:
    return False

  def calculation_equal(self, expr: expression.Expression) -> bool:
    # TODO(martinz): In theory, we could check for the equality of the
    # tensor_of_protos and the descriptors.
    return self is expr

  def _get_child_impl(self,
                      field_name: path.Step) -> Optional[expression.Expression]:
    return _get_child(self, self._descriptor, field_name,
                      self._tensor_of_protos)

  def known_field_names(self) -> FrozenSet[path.Step]:
    return _known_field_names_from_descriptor(self._descriptor)

  def __str__(self) -> str:
    return "_ProtoRootExpression: {}".format(str(self._descriptor.full_name))


ProtoExpression = Union[_ProtoRootExpression, _ProtoChildExpression,  # pylint: disable=invalid-name
                        _ProtoLeafExpression]

_ParentProtoExpression = Union[_ProtoChildExpression, _ProtoRootExpression]


def _known_field_names_from_descriptor(
    desc: descriptor.Descriptor) -> FrozenSet[StrStep]:
  return frozenset([field.name for field in desc.fields])


def _get_field_descriptor(
    desc: descriptor.Descriptor,
    field_name: ProtoFieldName) -> Optional[descriptor.FieldDescriptor]:
  if path.is_extension(field_name):
    try:
      return desc.file.pool.FindExtensionByName(
          path.get_raw_extension_name(field_name))
    except KeyError:
      return None
  return desc.fields_by_name.get(field_name)


def _get_any_child(
    parent: Union[_ProtoChildExpression,
                  _ProtoRootExpression], desc: descriptor.Descriptor,
    field_name: ProtoFieldName, backing_str_tensor: Optional[tf.Tensor]
) -> Optional[Union[_ProtoLeafExpression, _ProtoChildExpression]]:
  """Gets the child of an any descriptor."""
  if path.is_extension(field_name):
    full_name_child = parse_message_level_ex.get_full_name_from_any_step(
        field_name)
    if full_name_child is None:
      return None
    field_message = desc.file.pool.FindMessageTypeByName(full_name_child)
    return _ProtoChildExpression(parent, field_message, False, field_name,
                                 backing_str_tensor)
  else:
    return _get_child_helper(parent, desc.fields_by_name.get(field_name),
                             field_name, backing_str_tensor)


def _is_map_field_desc(field_desc: descriptor.FieldDescriptor) -> bool:
  return (field_desc.message_type and
          field_desc.message_type.GetOptions().map_entry)


def _get_map_child(
    parent: Union[_ProtoChildExpression, _ProtoRootExpression],
    desc: descriptor.Descriptor,
    field_name: ProtoFieldName,
    backing_str_tensor: Optional[tf.Tensor],
) -> Optional[Union[_ProtoLeafExpression, _ProtoChildExpression]]:
  """Gets the child given a map field."""
  [map_field_name, _] = path.parse_map_indexing_step(field_name)
  map_field_desc = desc.fields_by_name.get(map_field_name)
  if map_field_desc is None:
    return None
  if not _is_map_field_desc(map_field_desc):
    return None
  map_message_desc = map_field_desc.message_type
  if map_message_desc is None:
    # Note: I don't know if this is reachable. Theoretically, _is_map_field_desc
    # should have already returned false.
    return None
  value_field_desc = map_message_desc.fields_by_name.get("value")
  if value_field_desc is None:
    # Note: I don't know if this is reachable. Theoretically, _is_map_field_desc
    # should have already returned false.
    return None
  # This relies on the fact that the value is an optional field.
  return _get_child_helper(parent, value_field_desc, field_name,
                           backing_str_tensor)


def _get_child_helper(
    parent: Union[_ProtoChildExpression, _ProtoRootExpression],
    field_descriptor: Optional[descriptor.FieldDescriptor],
    field_name: ProtoFieldName, backing_str_tensor: Optional[tf.Tensor]
) -> Optional[Union[_ProtoChildExpression, _ProtoLeafExpression]]:
  """Helper function for _get_child, _get_any_child, and _get_map_child.

  Note that the field_descriptor.field_name is not necessarily equal to
  field_name, especially if this is called from _get_map_child.

  Args:
    parent: the parent expression
    field_descriptor: the field descriptor of the submessage represented by the
      returned expression, if present. If None, this will just return None.
    field_name: the field name of the _AbstractProtoChildExpression returned.
    backing_str_tensor: a string tensor representing the root serialized proto.
      This is passed to keep string_views of the tensor valid for all children
      of the root expression

  Returns:
    An _AbstractProtoChildExpression.
  """
  if field_descriptor is None:
    return None
  field_message = field_descriptor.message_type
  if field_message is None:
    return _ProtoLeafExpression(parent, field_descriptor, field_name)
  return _ProtoChildExpression(
      parent, field_message,
      field_descriptor.label == descriptor.FieldDescriptor.LABEL_REPEATED,
      field_name, backing_str_tensor)


def _get_child(
    parent: Union[_ProtoChildExpression, _ProtoRootExpression],
    desc: descriptor.Descriptor,
    field_name: path.Step,
    backing_str_tensor: Optional[tf.Tensor],
) -> Optional[Union[_ProtoChildExpression, _ProtoLeafExpression]]:
  """Get a child expression.

  This will get one of the following:
    A regular field.
    An extension.
    An Any filtered by value.
    A map field.

  Args:
    parent: The parent expression.
    desc: The descriptor of the parent.
    field_name: The name of the field.
    backing_str_tensor: a string tensor representing the root serialized proto.
      This is passed to keep string_views of the tensor valid for all children
      of the root expression

  Returns:
    The child expression, either a submessage or a leaf.
  """
  if isinstance(field_name, path.AnonymousId):
    return None
  if parse_message_level_ex.is_any_descriptor(desc):
    return _get_any_child(parent, desc, field_name, backing_str_tensor)
  if path.is_map_indexing_step(field_name):
    return _get_map_child(parent, desc, field_name, backing_str_tensor)
  # Works for extensions and regular fields, but not any or map.
  return _get_child_helper(parent, _get_field_descriptor(desc, field_name),
                           field_name, backing_str_tensor)


def _get_needed_fields(
    destinations: Sequence[expression.Expression]) -> Set[StrStep]:
  field_names = set()  # type: Set[StrStep]
  for destination in destinations:
    if isinstance(destination, _AbstractProtoChildExpression):
      field_names.add(destination.name_as_field)
  return field_names
