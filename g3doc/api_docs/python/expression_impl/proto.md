description: Expressions to parse a proto.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="expression_impl.proto" />
<meta itemprop="path" content="Stable" />
</div>

# Module: expression_impl.proto

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/expression_impl/proto.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Expressions to parse a proto.


These expressions return values with more information than standard node values.
Specifically, each node calculates additional tensors that are used as inputs
for its children.

## Classes

[`class DescriptorPool`](../expression_impl/proto/DescriptorPool.md): A collection of protobufs dynamically constructed by descriptor protos.

[`class FileDescriptorSet`](../expression_impl/proto/FileDescriptorSet.md): A ProtocolMessage

## Functions

[`create_expression_from_file_descriptor_set(...)`](../expression_impl/proto/create_expression_from_file_descriptor_set.md): Create an expression from a 1D tensor of serialized protos.

[`create_expression_from_proto(...)`](../expression_impl/proto/create_expression_from_proto.md): Create an expression from a 1D tensor of serialized protos.

[`create_transformed_field(...)`](../expression_impl/proto/create_transformed_field.md): Create an expression that transforms serialized proto tensors.

[`is_proto_expression(...)`](../expression_impl/proto/is_proto_expression.md): Returns true if an expression is a ProtoExpression.

## Type Aliases

[`ProtoExpression`](../expression_impl/proto/ProtoExpression.md)

[`TransformFn`](../expression_impl/proto/TransformFn.md)

