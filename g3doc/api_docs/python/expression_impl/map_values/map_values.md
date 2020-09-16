description: Map field into a new sibling.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="expression_impl.map_values.map_values" />
<meta itemprop="path" content="Stable" />
</div>

# expression_impl.map_values.map_values

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/expression_impl/map_values.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Map field into a new sibling.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>expression_impl.map_values.map_values(
    root: expression.Expression,
    source_path: path.Path,
    operation: Callable[[tf.Tensor], tf.Tensor],
    dtype: tf.DType,
    new_field_name: path.Step
) -> expression.Expression
</code></pre>



<!-- Placeholder for "Used in" -->

The shape of the output must be the same as the input.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`root`
</td>
<td>
original root.
</td>
</tr><tr>
<td>
`source_path`
</td>
<td>
source of the operation.
</td>
</tr><tr>
<td>
`operation`
</td>
<td>
operation from source_fields to new field.
</td>
</tr><tr>
<td>
`dtype`
</td>
<td>
type of new field.
</td>
</tr><tr>
<td>
`new_field_name`
</td>
<td>
name of the new field.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
The new expression.
</td>
</tr>

</table>

