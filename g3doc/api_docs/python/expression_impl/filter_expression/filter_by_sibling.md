description: Filter an expression by its sibling.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="expression_impl.filter_expression.filter_by_sibling" />
<meta itemprop="path" content="Stable" />
</div>

# expression_impl.filter_expression.filter_by_sibling

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/expression_impl/filter_expression.py#L74-L99">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Filter an expression by its sibling.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>expression_impl.filter_expression.filter_by_sibling(
    expr: expression.Expression,
    p: path.Path,
    sibling_field_name: path.Step,
    new_field_name: path.Step
) -> expression.Expression
</code></pre>



<!-- Placeholder for "Used in" -->


This is similar to boolean_mask. The shape of the path being filtered and
the sibling must be identical (e.g., each parent object must have an
equal number of source and sibling children).

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`expr`
</td>
<td>
the root expression.
</td>
</tr><tr>
<td>
`p`
</td>
<td>
a path to the source to be filtered.
</td>
</tr><tr>
<td>
`sibling_field_name`
</td>
<td>
the sibling to use as a mask.
</td>
</tr><tr>
<td>
`new_field_name`
</td>
<td>
a new sibling to create.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
a new root.
</td>
</tr>

</table>

