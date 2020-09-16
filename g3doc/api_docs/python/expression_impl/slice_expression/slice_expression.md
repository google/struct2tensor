description: Creates a new subtree with a sliced expression.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="expression_impl.slice_expression.slice_expression" />
<meta itemprop="path" content="Stable" />
</div>

# expression_impl.slice_expression.slice_expression

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/expression_impl/slice_expression.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Creates a new subtree with a sliced expression.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>expression_impl.slice_expression.slice_expression(
    expr: expression.Expression,
    p: path.Path,
    new_field_name: path.Step,
    begin: Optional[IndexValue],
    end: Optional[IndexValue]
) -> expression.Expression
</code></pre>



<!-- Placeholder for "Used in" -->

This follows the pattern of python slice() method.
See module-level comments for examples.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`expr`
</td>
<td>
the original root expression
</td>
</tr><tr>
<td>
`p`
</td>
<td>
the path to the source to be sliced.
</td>
</tr><tr>
<td>
`new_field_name`
</td>
<td>
the name of the new subtree.
</td>
</tr><tr>
<td>
`begin`
</td>
<td>
beginning index
</td>
</tr><tr>
<td>
`end`
</td>
<td>
end index.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A new root expression.
</td>
</tr>

</table>

