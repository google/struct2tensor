description: select a subtree.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="expression_impl.project.project" />
<meta itemprop="path" content="Stable" />
</div>

# expression_impl.project.project

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/expression_impl/project.py#L40-L59">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



select a subtree.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>expression_impl.project.project(
    expr: expression.Expression,
    paths: Sequence[path.Path]
) -> expression.Expression
</code></pre>



<!-- Placeholder for "Used in" -->

Paths not selected are removed.
Paths that are selected are "known", such that if calculate_prensors is
called, they will be in the result.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`expr`
</td>
<td>
the original expression.
</td>
</tr><tr>
<td>
`paths`
</td>
<td>
the paths to include.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A projected expression.
</td>
</tr>

</table>
