description: Promote and broadcast a set of paths to a particular location.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="expression_impl.promote_and_broadcast.promote_and_broadcast" />
<meta itemprop="path" content="Stable" />
</div>

# expression_impl.promote_and_broadcast.promote_and_broadcast

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/expression_impl/promote_and_broadcast.py#L150-L175">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Promote and broadcast a set of paths to a particular location.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>expression_impl.promote_and_broadcast.promote_and_broadcast(
    root: expression.Expression,
    path_dictionary: Mapping[path.Step, path.Path],
    dest_path_parent: path.Path
) -> expression.Expression
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
`path_dictionary`
</td>
<td>
a map from destination fields to origin paths.
</td>
</tr><tr>
<td>
`dest_path_parent`
</td>
<td>
a map from destination strings.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A new expression, where all the origin paths are promoted and broadcast
until they are children of dest_path_parent.
</td>
</tr>

</table>
