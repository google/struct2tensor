description: The value of the root.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="s2t.RootNodeTensor" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="get_positional_index"/>
</div>

# s2t.RootNodeTensor

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/prensor.py#L39-L72">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



The value of the root.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>s2t.RootNodeTensor(
    size: tf.Tensor
)
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`size`
</td>
<td>
A scalar int64 tensor saying how many root objects there are.
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
`size`
</td>
<td>

</td>
</tr>
</table>



## Methods

<h3 id="get_positional_index"><code>get_positional_index</code></h3>

<a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/prensor.py#L60-L69">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_positional_index() -> tf.Tensor
</code></pre>

Gets the positional index for this RootNodeTensor.

The positional index relative to the node's parent, and thus is always
monotonically increasing at step size 1 for a RootNodeTensor.

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





