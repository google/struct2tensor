description: The value of a leaf node.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="s2t.LeafNodeTensor" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="get_positional_index"/>
</div>

# s2t.LeafNodeTensor

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/prensor.py#L138-L196">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



The value of a leaf node.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>s2t.LeafNodeTensor(
    parent_index: tf.Tensor,
    values: tf.Tensor,
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
parent index of values[i]
</td>
</tr><tr>
<td>
`values`
</td>
<td>
a 1-D tensor of equal length to parent_index.
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
`values`
</td>
<td>

</td>
</tr>
</table>



## Methods

<h3 id="get_positional_index"><code>get_positional_index</code></h3>

<a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/prensor.py#L171-L191">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_positional_index() -> tf.Tensor
</code></pre>

Gets the positional index for this LeafNodeTensor.

The positional index tells us which index of the parent an element is.

For example, with the following parent indices: [0, 0, 2]
we would have positional index:
[
  0, # The 0th element of the 0th parent.
  1, # The 1st element of the 0th parent.
  0  # The 0th element of the 2nd parent.
].

For more information, view ops/run_length_before_op.cc

This is the same for Child NodeTensors.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A tensor of positional indices.
</td>
</tr>

</table>





