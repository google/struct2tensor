description: Map a ragged tensor.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="expression_impl.map_prensor.map_ragged_tensor" />
<meta itemprop="path" content="Stable" />
</div>

# expression_impl.map_prensor.map_ragged_tensor

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/expression_impl/map_prensor.py#L141-L164">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Map a ragged tensor.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>expression_impl.map_prensor.map_ragged_tensor(
    root: expression.Expression,
    root_path: path.Path,
    paths: Sequence[path.Path],
    operation: Callable[..., tf.RaggedTensor],
    is_repeated: bool,
    dtype: tf.DType,
    new_field_name: path.Step
) -> expression.Expression
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`root`
</td>
<td>
the root of the expression.
</td>
</tr><tr>
<td>
`root_path`
</td>
<td>
the path relative to which the ragged tensors are calculated.
</td>
</tr><tr>
<td>
`paths`
</td>
<td>
the input paths relative to the root_path
</td>
</tr><tr>
<td>
`operation`
</td>
<td>
a method that takes the list of ragged tensors as input and
returns a ragged tensor.
</td>
</tr><tr>
<td>
`is_repeated`
</td>
<td>
true if the result of operation is repeated.
</td>
</tr><tr>
<td>
`dtype`
</td>
<td>
dtype of the result of the operation.
</td>
</tr><tr>
<td>
`new_field_name`
</td>
<td>
root_path.get_child(new_field_name) is the path of the
result.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A new root expression containing the old root expression plus the new path,
root_path.get_child(new_field_name), with the result of the operation.
</td>
</tr>

</table>

