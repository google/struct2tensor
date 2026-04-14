description: Create an expression from a 1D tensor of serialized protos.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="expression_impl.proto.create_expression_from_proto" />
<meta itemprop="path" content="Stable" />
</div>

# expression_impl.proto.create_expression_from_proto

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/expression_impl/proto.py#L85-L100">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Create an expression from a 1D tensor of serialized protos.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>expression_impl.proto.create_expression_from_proto(
    tensor_of_protos: tf.Tensor,
    desc: descriptor.Descriptor,
    message_format: str = &#x27;binary&#x27;
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
`desc`
</td>
<td>
a descriptor of protos in tensor of protos.
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
