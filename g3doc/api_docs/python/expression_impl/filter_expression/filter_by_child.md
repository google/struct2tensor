description: Filter an expression by an optional boolean child field.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="expression_impl.filter_expression.filter_by_child" />
<meta itemprop="path" content="Stable" />
</div>

# expression_impl.filter_expression.filter_by_child

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/expression_impl/filter_expression.py#L102-L124">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Filter an expression by an optional boolean child field.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>expression_impl.filter_expression.filter_by_child(
    expr: expression.Expression,
    p: path.Path,
    child_field_name: path.Step,
    new_field_name: path.Step
) -> expression.Expression
</code></pre>



<!-- Placeholder for "Used in" -->

If the child field is present and True, then keep that parent.
Otherwise, drop the parent.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`expr`
</td>
<td>
the original expression
</td>
</tr><tr>
<td>
`p`
</td>
<td>
the path to filter.
</td>
</tr><tr>
<td>
`child_field_name`
</td>
<td>
the boolean child field to use to filter.
</td>
</tr><tr>
<td>
`new_field_name`
</td>
<td>
the new, filtered version of path.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
The new root expression.
</td>
</tr>

</table>
