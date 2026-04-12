description: A finite schema for a prensor.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="expression_impl.map_prensor_to_prensor.Schema" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="get_child"/>
<meta itemprop="property" content="known_field_names"/>
</div>

# expression_impl.map_prensor_to_prensor.Schema

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/expression_impl/map_prensor_to_prensor.py#L84-L149">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A finite schema for a prensor.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>expression_impl.map_prensor_to_prensor.Schema(
    is_repeated: bool = True,
    dtype: Optional[tf.DType] = None,
    schema_feature: Optional[schema_pb2.Feature] = None,
    children: Optional[Dict[path.Step, 'Schema']] = None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Effectively, this stores everything for the prensor but the tensors
themselves.

Notice that this is slightly different than schema_pb2.Schema, although
similar in nature. At present, there is no clear way to extract is_repeated
and dtype from schema_pb2.Schema.

See create_schema below for constructing a schema.

Note that for LeafNodeTensor, dtype is not None.
Also, for ChildNodeTensor and RootNodeTensor, dtype is None. However,
a ChildNodeTensor or RootNodeTensor could be childless.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`is_repeated`
</td>
<td>
is the root repeated?
</td>
</tr><tr>
<td>
`dtype`
</td>
<td>
tf.dtype of the root if the root is a leaf, otherwise None.
</td>
</tr><tr>
<td>
`schema_feature`
</td>
<td>
schema_pb2.Feature of the root (no struct_domain
necessary)
</td>
</tr><tr>
<td>
`children`
</td>
<td>
child schemas.
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
`schema_feature`
</td>
<td>

</td>
</tr><tr>
<td>
`type`
</td>
<td>

</td>
</tr>
</table>



## Methods

<h3 id="get_child"><code>get_child</code></h3>

<a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/expression_impl/map_prensor_to_prensor.py#L134-L135">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_child(
    key: path.Step
)
</code></pre>




<h3 id="known_field_names"><code>known_field_names</code></h3>

<a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/expression_impl/map_prensor_to_prensor.py#L137-L138">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>known_field_names() -> FrozenSet[path.Step]
</code></pre>
