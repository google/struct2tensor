description: Calculate the size of a field, and store it as an anonymous sibling.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="expression_impl.size.size_anonymous" />
<meta itemprop="path" content="Stable" />
</div>

# expression_impl.size.size_anonymous

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/expression_impl/size.py#L43-L54">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Calculate the size of a field, and store it as an anonymous sibling.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>expression_impl.size.size_anonymous(
    root: expression.Expression,
    source_path: path.Path
) -> Tuple[expression.Expression, path.Path]
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`root`
</td>
<td>
the original expression.
</td>
</tr><tr>
<td>
`source_path`
</td>
<td>
the source path to measure. Cannot be root.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
The new expression and the new field as a pair.
</td>
</tr>

</table>
