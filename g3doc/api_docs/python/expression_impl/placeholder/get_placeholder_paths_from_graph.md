description: Gets all placeholder paths from an expression graph.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="expression_impl.placeholder.get_placeholder_paths_from_graph" />
<meta itemprop="path" content="Stable" />
</div>

# expression_impl.placeholder.get_placeholder_paths_from_graph

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/expression_impl/placeholder.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Gets all placeholder paths from an expression graph.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>expression_impl.placeholder.get_placeholder_paths_from_graph(
    graph: calculate.ExpressionGraph
) -> List[path.Path]
</code></pre>



<!-- Placeholder for "Used in" -->

This finds all leaf placeholder expressions in an expression graph, and gets
the path of these expressions.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`graph`
</td>
<td>
expression graph
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
a list of paths of placeholder expressions
</td>
</tr>

</table>

