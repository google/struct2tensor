description: Maps an expression to a prensor, and merges that prensor.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="expression_impl.map_prensor_to_prensor.map_prensor_to_prensor" />
<meta itemprop="path" content="Stable" />
</div>

# expression_impl.map_prensor_to_prensor.map_prensor_to_prensor

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/expression_impl/map_prensor_to_prensor.py#L205-L265">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Maps an expression to a prensor, and merges that prensor.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>expression_impl.map_prensor_to_prensor.map_prensor_to_prensor(
    root_expr: expression.Expression,
    source: path.Path,
    paths_needed: Sequence[path.Path],
    prensor_op: Callable[[prensor.Prensor], prensor.Prensor],
    output_schema: <a href="../../expression_impl/map_prensor_to_prensor/Schema.md"><code>expression_impl.map_prensor_to_prensor.Schema</code></a>
) -> expression.Expression
</code></pre>



<!-- Placeholder for "Used in" -->

For example, suppose you have an op my_op, that takes a prensor of the form:

  event
   / \
 foo   bar

and produces a prensor of the form my_result_schema:

   event
    / \
 foo2 bar2

If you give it an expression original with the schema:

 session
    |
  event
  /  \
foo   bar

result = map_prensor_to_prensor(
  original,
  path.Path(["session","event"]),
  my_op,
  my_output_schema)

Result will have the schema:

 session
    |
  event--------
  /  \    \    \
foo   bar foo2 bar2

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`root_expr`
</td>
<td>
the root expression
</td>
</tr><tr>
<td>
`source`
</td>
<td>
the path where the prensor op is applied.
</td>
</tr><tr>
<td>
`paths_needed`
</td>
<td>
the paths needed for the op.
</td>
</tr><tr>
<td>
`prensor_op`
</td>
<td>
the prensor op
</td>
</tr><tr>
<td>
`output_schema`
</td>
<td>
the output schema of the op.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A new expression where the prensor is merged.
</td>
</tr>

</table>
