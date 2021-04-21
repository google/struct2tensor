description: Gets the positional index.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="expression_impl.index.get_positional_index" />
<meta itemprop="path" content="Stable" />
</div>

# expression_impl.index.get_positional_index

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/expression_impl/index.py#L131-L153">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Gets the positional index.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>expression_impl.index.get_positional_index(
    expr: expression.Expression,
    source_path: path.Path,
    new_field_name: path.Step
) -> Tuple[expression.Expression, path.Path]
</code></pre>



<!-- Placeholder for "Used in" -->

Given a field with parent_index [0,1,1,2,3,4,4], this returns:
parent_index [0,1,1,2,3,4,4] and value [0,0,1,0,0,0,1]

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`expr`
</td>
<td>
original expression
</td>
</tr><tr>
<td>
`source_path`
</td>
<td>
path in expression to get index of.
</td>
</tr><tr>
<td>
`new_field_name`
</td>
<td>
the name of the new field.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
The new expression and the new path as a pair.
</td>
</tr>

</table>

