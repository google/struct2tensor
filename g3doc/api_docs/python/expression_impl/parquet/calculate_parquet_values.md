description: Calculates expressions and returns a parquet dataset.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="expression_impl.parquet.calculate_parquet_values" />
<meta itemprop="path" content="Stable" />
</div>

# expression_impl.parquet.calculate_parquet_values

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/expression_impl/parquet.py#L71-L91">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Calculates expressions and returns a parquet dataset.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>expression_impl.parquet.calculate_parquet_values(
    expressions: List[expression.Expression],
    root_exp: placeholder._PlaceholderRootExpression,
    filenames: List[str],
    batch_size: int,
    options: Optional[calculate_options.Options] = None
)
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`expressions`
</td>
<td>
A list of expressions to calculate.
</td>
</tr><tr>
<td>
`root_exp`
</td>
<td>
The root placeholder expression to use as the feed dict.
</td>
</tr><tr>
<td>
`filenames`
</td>
<td>
A list of parquet files.
</td>
</tr><tr>
<td>
`batch_size`
</td>
<td>
The number of messages to batch.
</td>
</tr><tr>
<td>
`options`
</td>
<td>
calculate options.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A parquet dataset.
</td>
</tr>

</table>

