description: Create a schema recursively.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="expression_impl.map_prensor_to_prensor.create_schema" />
<meta itemprop="path" content="Stable" />
</div>

# expression_impl.map_prensor_to_prensor.create_schema

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/expression_impl/map_prensor_to_prensor.py#L152-L183">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Create a schema recursively.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>expression_impl.map_prensor_to_prensor.create_schema(
    is_repeated: bool = True,
    dtype: Optional[tf.DType] = None,
    schema_feature: Optional[schema_pb2.Feature] = None,
    children: Optional[Dict[path.Step, Any]] = None
) -> <a href="../../expression_impl/map_prensor_to_prensor/Schema.md"><code>expression_impl.map_prensor_to_prensor.Schema</code></a>
</code></pre>



<!-- Placeholder for "Used in" -->


#### Example:


my_result_schema = create_schema(
  is_repeated=True,
  children={"foo2":{is_repeated=True, dtype=tf.int64},
            "bar2":{is_repeated=False, dtype=tf.int64}})

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`is_repeated`
</td>
<td>
whether the root is repeated.
</td>
</tr><tr>
<td>
`dtype`
</td>
<td>
the dtype of a leaf (None for non-leaves).
</td>
</tr><tr>
<td>
`schema_feature`
</td>
<td>
the schema_pb2.Feature describing this expression. name and
struct_domain need not be specified.
</td>
</tr><tr>
<td>
`children`
</td>
<td>
the child schemas. Note that the value type of children is either
a Schema or a dictionary of arguments to create_schema.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
a new Schema represented by the inputs.
</td>
</tr>

</table>
