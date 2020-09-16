description: Create an expression from a 1D tensor of serialized protos.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="expression_impl.proto.create_expression_from_file_descriptor_set" />
<meta itemprop="path" content="Stable" />
</div>

# expression_impl.proto.create_expression_from_file_descriptor_set

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/expression_impl/proto.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Create an expression from a 1D tensor of serialized protos.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>expression_impl.proto.create_expression_from_file_descriptor_set(
    tensor_of_protos: tf.Tensor,
    proto_name: ProtoFullName,
    file_descriptor_set: <a href="../../expression_impl/proto/FileDescriptorSet.md"><code>expression_impl.proto.FileDescriptorSet</code></a>,
    message_format: str = 'binary'
) -> expression.Expression
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`tensor_of_protos`
</td>
<td>
1D tensor of serialized protos.
</td>
</tr><tr>
<td>
`proto_name`
</td>
<td>
fully qualified name (e.g. "some.package.SomeProto") of the
proto in `tensor_of_protos`.
</td>
</tr><tr>
<td>
`file_descriptor_set`
</td>
<td>
The FileDescriptorSet proto containing `proto_name`'s
and all its dependencies' FileDescriptorProto. Note that if file1 imports
file2, then file2's FileDescriptorProto must precede file1's in
file_descriptor_set.file.
</td>
</tr><tr>
<td>
`message_format`
</td>
<td>
Indicates the format of the protocol buffer: is one of
'text' or 'binary'.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
An expression.
</td>
</tr>

</table>

