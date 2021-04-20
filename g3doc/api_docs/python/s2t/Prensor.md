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
</div>

# s2t.Prensor

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/prensor.py#L339-L452">
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

<a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/prensor.py#L410-L412">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>field_names() -> FrozenSet[<a href="../s2t/Step.md"><code>s2t.Step</code></a>]
</code></pre>

Returns the field names of the children.


<h3 id="get_child"><code>get_child</code></h3>

<a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/prensor.py#L364-L366">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_child(
    field_name: <a href="../s2t/Step.md"><code>s2t.Step</code></a>
) -> Optional['Prensor']
</code></pre>

Gets the child at field_name.


<h3 id="get_child_or_error"><code>get_child_or_error</code></h3>

<a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/prensor.py#L373-L379">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_child_or_error(
    field_name: <a href="../s2t/Step.md"><code>s2t.Step</code></a>
) -> "Prensor"
</code></pre>

Gets the child at field_name.


<h3 id="get_children"><code>get_children</code></h3>

<a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/prensor.py#L397-L399">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_children() -> "collections.OrderedDict[path.Step, Prensor]"
</code></pre>

A map from field name to subexpression.


<h3 id="get_descendant"><code>get_descendant</code></h3>

<a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/prensor.py#L381-L388">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_descendant(
    p: <a href="../s2t/Path.md"><code>s2t.Path</code></a>
) -> Optional['Prensor']
</code></pre>

Finds the descendant at the path.


<h3 id="get_descendant_or_error"><code>get_descendant_or_error</code></h3>

<a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/prensor.py#L390-L395">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_descendant_or_error(
    p: <a href="../s2t/Path.md"><code>s2t.Path</code></a>
) -> "Prensor"
</code></pre>

Finds the descendant at the path.


<h3 id="get_descendants"><code>get_descendants</code></h3>

<a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/prensor.py#L401-L408">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_descendants() -> Mapping[path.Path, 'Prensor']
</code></pre>

A map from paths to all subexpressions.




