description: Create a prensor from a map of paths to NodeTensor.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="s2t.create_prensor_from_descendant_nodes" />
<meta itemprop="path" content="Stable" />
</div>

# s2t.create_prensor_from_descendant_nodes

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/prensor.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Create a prensor from a map of paths to NodeTensor.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>s2t.create_prensor_from_descendant_nodes(
    nodes: Mapping[<a href="../s2t/Path.md"><code>s2t.Path</code></a>, <a href="../s2t/NodeTensor.md"><code>s2t.NodeTensor</code></a>]
) -> "Prensor"
</code></pre>



<!-- Placeholder for "Used in" -->

If a path is a key in the map, all prefixes of that path must be present.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`nodes`
</td>
<td>
A map from paths to NodeTensors.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A Prensor.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
if there is a prefix of a path missing.
</td>
</tr>
</table>

