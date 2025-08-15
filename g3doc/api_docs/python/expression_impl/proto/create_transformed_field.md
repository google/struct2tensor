description: Create an expression that transforms serialized proto tensors.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="expression_impl.proto.create_transformed_field" />
<meta itemprop="path" content="Stable" />
</div>

# expression_impl.proto.create_transformed_field

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/expression_impl/proto.py#L116-L187">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Create an expression that transforms serialized proto tensors.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>expression_impl.proto.create_transformed_field(
    expr: expression.Expression,
    source_path: path.CoercableToPath,
    dest_field: StrStep,
    transform_fn: <a href="../../expression_impl/proto/TransformFn.md"><code>expression_impl.proto.TransformFn</code></a>
) -> expression.Expression
</code></pre>



<!-- Placeholder for "Used in" -->

The transform_fn argument should take the form:

def transform_fn(parent_indices, values):
  ...
  return (transformed_parent_indices, transformed_values)

#### Given:


- parent_indices: an int64 vector of non-decreasing parent message indices.
- values: a string vector of serialized protos having the same shape as
  `parent_indices`.
`transform_fn` must return new parent indices and serialized values encoding
the same proto message as the passed in `values`.  These two vectors must
have the same size, but it need not be the same as the input arguments.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`expr`
</td>
<td>
a source expression containing `source_path`.
</td>
</tr><tr>
<td>
`source_path`
</td>
<td>
the path to the field to reverse.
</td>
</tr><tr>
<td>
`dest_field`
</td>
<td>
the name of the newly created field. This field will be a
sibling of the field identified by `source_path`.
</td>
</tr><tr>
<td>
`transform_fn`
</td>
<td>
a callable that accepts parent_indices and serialized proto
values and returns a posibly modified parent_indices and values. Note that
when CalcuateOptions.use_string_view is set, transform_fn should not have
any stateful side effecting uses of serialized proto inputs. Doing so
could cause segfaults as the backing string tensor lifetime is not
guaranteed when the side effecting operations are run.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
An expression.
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
if the source path is not a proto message field.
</td>
</tr>
</table>
