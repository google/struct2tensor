description: Gets the number of steps from the end of the array.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="expression_impl.index.get_index_from_end" />
<meta itemprop="path" content="Stable" />
</div>

# expression_impl.index.get_index_from_end

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/expression_impl/index.py#L156-L185">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Gets the number of steps from the end of the array.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>expression_impl.index.get_index_from_end(
    t: expression.Expression,
    source_path: path.Path,
    new_field_name: path.Step
) -> Tuple[expression.Expression, path.Path]
</code></pre>



<!-- Placeholder for "Used in" -->

Given an array ["a", "b", "c"], with indices [0, 1, 2], the result of this
is [-3,-2,-1].

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`t`
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
