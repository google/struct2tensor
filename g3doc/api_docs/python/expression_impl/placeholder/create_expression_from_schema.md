description: Creates a placeholder expression from a parquet schema.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="expression_impl.placeholder.create_expression_from_schema" />
<meta itemprop="path" content="Stable" />
</div>

# expression_impl.placeholder.create_expression_from_schema

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/expression_impl/placeholder.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Creates a placeholder expression from a parquet schema.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>expression_impl.placeholder.create_expression_from_schema(
    schema: <a href="../../expression_impl/map_prensor_to_prensor/Schema.md"><code>expression_impl.map_prensor_to_prensor.Schema</code></a>
) -> "_PlaceholderRootExpression"
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`schema`
</td>
<td>
The schema that describes the prensor tree that this placeholder
represents.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A PlaceholderRootExpression that should be used as the root of an expression
graph.
</td>
</tr>

</table>

