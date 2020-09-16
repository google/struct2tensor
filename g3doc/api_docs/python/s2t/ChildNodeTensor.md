description: The value of an intermediate node.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="s2t.ChildNodeTensor" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
</div>

# s2t.ChildNodeTensor

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/prensor.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



The value of an intermediate node.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>s2t.ChildNodeTensor(
    parent_index: tf.Tensor,
    is_repeated: bool
)
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`parent_index`
</td>
<td>
a 1-D int64 tensor where parent_index[i] represents the
parent index of the ith child.
</td>
</tr><tr>
<td>
`is_repeated`
</td>
<td>
a bool indicating if there can be more than one child per
parent.
</td>
</tr>
</table>





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`is_repeated`
</td>
<td>

</td>
</tr><tr>
<td>
`parent_index`
</td>
<td>

</td>
</tr><tr>
<td>
`size`
</td>
<td>
Returns the size, as if this was the root prensor.
</td>
</tr>
</table>



