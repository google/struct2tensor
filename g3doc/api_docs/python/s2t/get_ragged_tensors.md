description: Gets ragged tensors for all the leaves of the prensor expression.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="s2t.get_ragged_tensors" />
<meta itemprop="path" content="Stable" />
</div>

# s2t.get_ragged_tensors

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/prensor_util.py#L301-L318">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Gets ragged tensors for all the leaves of the prensor expression.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>s2t.get_ragged_tensors(
    t: <a href="../s2t/Prensor.md"><code>s2t.Prensor</code></a>,
    options: calculate_options.Options = calculate_options.get_default_options()
) -> Mapping[<a href="../s2t/Path.md"><code>s2t.Path</code></a>, tf.RaggedTensor]
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`t`
</td>
<td>
The Prensor to extract tensors from.
</td>
</tr><tr>
<td>
`options`
</td>
<td>
used to pass options for calculating ragged tensors.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A map from paths to ragged tensors.
</td>
</tr>

</table>

