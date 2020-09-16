description: Gets the prensor value of the expressions.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="s2t.calculate_prensors" />
<meta itemprop="path" content="Stable" />
</div>

# s2t.calculate_prensors

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/calculate.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Gets the prensor value of the expressions.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>s2t.calculate_prensors(
    expressions: Sequence[<a href="../s2t/Expression.md"><code>s2t.Expression</code></a>],
    options: Optional[calculate_options.Options] = None,
    feed_dict: Optional[Dict[<a href="../s2t/Expression.md"><code>s2t.Expression</code></a>, <a href="../s2t/Prensor.md"><code>s2t.Prensor</code></a>]] = None
) -> Sequence[<a href="../s2t/Prensor.md"><code>s2t.Prensor</code></a>]
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
expressions to calculate prensors for.
</td>
</tr><tr>
<td>
`options`
</td>
<td>
options for calculate(...).
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
a list of prensors.
</td>
</tr>

</table>

