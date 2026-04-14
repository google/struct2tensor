description: Gets the prensor value of the expressions and the graph used.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="s2t.calculate_prensors_with_graph" />
<meta itemprop="path" content="Stable" />
</div>

# s2t.calculate_prensors_with_graph

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/calculate.py#L106-L136">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Gets the prensor value of the expressions and the graph used.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>s2t.calculate_prensors_with_graph(
    expressions: Sequence[<a href="../s2t/Expression.md"><code>s2t.Expression</code></a>],
    options: Optional[calculate_options.Options] = None,
    feed_dict: Optional[Dict[expression.Expression, prensor.Prensor]] = None
) -> Tuple[Sequence[prensor.Prensor], 'ExpressionGraph']
</code></pre>



<!-- Placeholder for "Used in" -->

This method is useful for getting information like the protobuf fields parsed
to create an expression.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`expressions`
</td>
<td>
expressions to calculate prensors for.
</td>
</tr><tr>
<td>
`options`
</td>
<td>
options for calculate(...) methods.
</td>
</tr><tr>
<td>
`feed_dict`
</td>
<td>
a dictionary, mapping expression to prensor that will be used
as the initial expression in the expression graph.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
a list of prensors, and the graph used to calculate them.
</td>
</tr>

</table>
