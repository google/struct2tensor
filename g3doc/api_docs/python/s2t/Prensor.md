description: A expression of NodeTensor objects.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="s2t.Prensor" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="field_names"/>
<meta itemprop="property" content="get_child"/>
<meta itemprop="property" content="get_child_or_error"/>
<meta itemprop="property" content="get_children"/>
<meta itemprop="property" content="get_descendant"/>
<meta itemprop="property" content="get_descendant_or_error"/>
<meta itemprop="property" content="get_descendants"/>
<meta itemprop="property" content="get_ragged_tensor"/>
<meta itemprop="property" content="get_ragged_tensors"/>
<meta itemprop="property" content="get_sparse_tensor"/>
<meta itemprop="property" content="get_sparse_tensors"/>
</div>

# s2t.Prensor

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/prensor.py#L340-L524">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A expression of NodeTensor objects.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>s2t.Prensor(
    node: <a href="../s2t/NodeTensor.md"><code>s2t.NodeTensor</code></a>,
    children: "collections.OrderedDict[path.Step, Prensor]"
)
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`node`
</td>
<td>
the NodeTensor of the root.
</td>
</tr><tr>
<td>
`children`
</td>
<td>
a map from edge to subexpression.
</td>
</tr>
</table>





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`is_leaf`
</td>
<td>
True iff the node value is a LeafNodeTensor.
</td>
</tr><tr>
<td>
`node`
</td>
<td>
The node of the root of the subtree.
</td>
</tr>
</table>



## Methods

<h3 id="field_names"><code>field_names</code></h3>

<a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/prensor.py#L411-L413">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>field_names() -> FrozenSet[<a href="../s2t/Step.md"><code>s2t.Step</code></a>]
</code></pre>

Returns the field names of the children.


<h3 id="get_child"><code>get_child</code></h3>

<a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/prensor.py#L365-L367">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_child(
    field_name: <a href="../s2t/Step.md"><code>s2t.Step</code></a>
) -> Optional['Prensor']
</code></pre>

Gets the child at field_name.


<h3 id="get_child_or_error"><code>get_child_or_error</code></h3>

<a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/prensor.py#L374-L380">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_child_or_error(
    field_name: <a href="../s2t/Step.md"><code>s2t.Step</code></a>
) -> "Prensor"
</code></pre>

Gets the child at field_name.


<h3 id="get_children"><code>get_children</code></h3>

<a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/prensor.py#L398-L400">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_children() -> "collections.OrderedDict[path.Step, Prensor]"
</code></pre>

A map from field name to subexpression.


<h3 id="get_descendant"><code>get_descendant</code></h3>

<a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/prensor.py#L382-L389">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_descendant(
    p: <a href="../s2t/Path.md"><code>s2t.Path</code></a>
) -> Optional['Prensor']
</code></pre>

Finds the descendant at the path.


<h3 id="get_descendant_or_error"><code>get_descendant_or_error</code></h3>

<a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/prensor.py#L391-L396">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_descendant_or_error(
    p: <a href="../s2t/Path.md"><code>s2t.Path</code></a>
) -> "Prensor"
</code></pre>

Finds the descendant at the path.


<h3 id="get_descendants"><code>get_descendants</code></h3>

<a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/prensor.py#L402-L409">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_descendants() -> Mapping[path.Path, 'Prensor']
</code></pre>

A map from paths to all subexpressions.


<h3 id="get_ragged_tensor"><code>get_ragged_tensor</code></h3>

<a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/prensor.py#L430-L448">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_ragged_tensor(
    p: <a href="../s2t/Path.md"><code>s2t.Path</code></a>,
    options: calculate_options.Options = calculate_options.get_default_options()
) -> tf.RaggedTensor
</code></pre>

Get a ragged tensor for a path.

All steps are represented in the ragged tensor.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`p`
</td>
<td>
the path to a leaf node in `t`.
</td>
</tr><tr>
<td>
`options`
</td>
<td>
Options for calculating ragged tensors.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A ragged tensor containing values of the leaf node, preserving the
structure along the path. Raises an error if the path is not found.
</td>
</tr>

</table>



<h3 id="get_ragged_tensors"><code>get_ragged_tensors</code></h3>

<a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/prensor.py#L415-L428">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_ragged_tensors(
    options: calculate_options.Options = calculate_options.get_default_options()
) -> Mapping[<a href="../s2t/Path.md"><code>s2t.Path</code></a>, tf.RaggedTensor]
</code></pre>

Gets ragged tensors for all the leaves of the prensor expression.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`options`
</td>
<td>
Options for calculating ragged tensors.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A map from paths to ragged tensors.
</td>
</tr>

</table>



<h3 id="get_sparse_tensor"><code>get_sparse_tensor</code></h3>

<a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/prensor.py#L450-L469">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_sparse_tensor(
    p: <a href="../s2t/Path.md"><code>s2t.Path</code></a>,
    options: calculate_options.Options = calculate_options.get_default_options()
) -> tf.SparseTensor
</code></pre>

Gets a sparse tensor for path p.

Note that any optional fields are not registered as dimensions, as they
can't be represented in a sparse tensor.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`p`
</td>
<td>
The path to a leaf node in `t`.
</td>
</tr><tr>
<td>
`options`
</td>
<td>
Currently unused.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A sparse tensor containing values of the leaf node, preserving the
structure along the path. Raises an error if the path is not found.
</td>
</tr>

</table>



<h3 id="get_sparse_tensors"><code>get_sparse_tensors</code></h3>

<a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/prensor.py#L471-L484">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_sparse_tensors(
    options: calculate_options.Options = calculate_options.get_default_options()
) -> Mapping[<a href="../s2t/Path.md"><code>s2t.Path</code></a>, tf.SparseTensor]
</code></pre>

Gets sparse tensors for all the leaves of the prensor expression.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`options`
</td>
<td>
Currently unused.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A map from paths to sparse tensors.
</td>
</tr>

</table>





