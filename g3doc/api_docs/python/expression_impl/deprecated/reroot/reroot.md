description: Reroot to a new path, maintaining a input proto index.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="expression_impl.reroot.reroot" />
<meta itemprop="path" content="Stable" />
</div>

# expression_impl.reroot.reroot

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/expression_impl/reroot.py#L31-L49">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Reroot to a new path, maintaining a input proto index.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>expression_impl.reroot.reroot(
    root: expression.Expression,
    source_path: path.Path
) -> expression.Expression
</code></pre>



<!-- Placeholder for "Used in" -->

Similar to root.get_descendant_or_error(source_path): however, this
method retains the ability to get a map to the original index.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`root`
</td>
<td>
the original root.
</td>
</tr><tr>
<td>
`source_path`
</td>
<td>
the path to the new root.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
the new root.
</td>
</tr>

</table>

