description: The root of the promoted sub tree.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="expression_impl.promote.PromoteChildExpression" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="apply"/>
<meta itemprop="property" content="apply_schema"/>
<meta itemprop="property" content="broadcast"/>
<meta itemprop="property" content="calculate"/>
<meta itemprop="property" content="calculation_equal"/>
<meta itemprop="property" content="calculation_is_identity"/>
<meta itemprop="property" content="cogroup_by_index"/>
<meta itemprop="property" content="create_has_field"/>
<meta itemprop="property" content="create_proto_index"/>
<meta itemprop="property" content="create_size_field"/>
<meta itemprop="property" content="get_child"/>
<meta itemprop="property" content="get_child_or_error"/>
<meta itemprop="property" content="get_descendant"/>
<meta itemprop="property" content="get_descendant_or_error"/>
<meta itemprop="property" content="get_known_children"/>
<meta itemprop="property" content="get_known_descendants"/>
<meta itemprop="property" content="get_paths_with_schema"/>
<meta itemprop="property" content="get_schema"/>
<meta itemprop="property" content="get_source_expressions"/>
<meta itemprop="property" content="known_field_names"/>
<meta itemprop="property" content="map_field_values"/>
<meta itemprop="property" content="map_ragged_tensors"/>
<meta itemprop="property" content="map_sparse_tensors"/>
<meta itemprop="property" content="project"/>
<meta itemprop="property" content="promote"/>
<meta itemprop="property" content="promote_and_broadcast"/>
<meta itemprop="property" content="reroot"/>
<meta itemprop="property" content="schema_string"/>
<meta itemprop="property" content="slice"/>
<meta itemprop="property" content="truncate"/>
</div>

# expression_impl.promote.PromoteChildExpression

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/expression_impl/promote.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



The root of the promoted sub tree.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>expression_impl.promote.PromoteChildExpression(
    origin: expression.Expression,
    origin_parent: expression.Expression
)
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`is_repeated`
</td>
<td>
if the expression is repeated.
</td>
</tr><tr>
<td>
`my_type`
</td>
<td>
the DType of a field, or None for an internal node.
</td>
</tr><tr>
<td>
`schema_feature`
</td>
<td>
the local schema (StructDomain information should not be
present).
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
True iff the node tensor is a LeafNodeTensor.
</td>
</tr><tr>
<td>
`is_repeated`
</td>
<td>
True iff the same parent value can have multiple children values.
</td>
</tr><tr>
<td>
`schema_feature`
</td>
<td>
Return the schema of the field.
</td>
</tr><tr>
<td>
`type`
</td>
<td>
dtype of the expression, or None if not a leaf expression.
</td>
</tr>
</table>



## Methods

<h3 id="apply"><code>apply</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>apply(
    transform: Callable[['Expression'], 'Expression']
) -> "Expression"
</code></pre>




<h3 id="apply_schema"><code>apply_schema</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>apply_schema(
    schema: schema_pb2.Schema
) -> "Expression"
</code></pre>




<h3 id="broadcast"><code>broadcast</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>broadcast(
    source_path: CoercableToPath,
    sibling_field: path.Step,
    new_field_name: path.Step
) -> "Expression"
</code></pre>

Broadcasts the existing field at source_path to the sibling_field.


<h3 id="calculate"><code>calculate</code></h3>

<a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/expression_impl/promote.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>calculate(
    sources: Sequence[prensor.NodeTensor],
    destinations: Sequence[expression.Expression],
    options: calculate_options.Options,
    side_info: Optional[prensor.Prensor] = None
) -> prensor.NodeTensor
</code></pre>

Calculates the node tensor of the expression.

The node tensor must be a function of the properties of the expression
and the node tensors of the expressions from get_source_expressions().

If is_leaf, then calculate must return a LeafNodeTensor.
Otherwise, it must return a ChildNodeTensor or RootNodeTensor.

If calculate_is_identity is true, then this must return source_tensors[0].

Sometimes, for operations such as parsing the proto, calculate will return
additional information. For example, calculate() for the root of the
proto expression also parses out the tensors required to calculate the
tensors of the children. This is why destinations are required.

For a reference use, see calculate_value_slowly(...) below.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`source_tensors`
</td>
<td>
The node tensors of the expressions in
get_source_expressions().
</td>
</tr><tr>
<td>
`destinations`
</td>
<td>
The expressions that will use the output of this method.
</td>
</tr><tr>
<td>
`options`
</td>
<td>
Options for the calculation.
</td>
</tr><tr>
<td>
`side_info`
</td>
<td>
An optional prensor that is used to bind to a placeholder
expression.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A NodeTensor representing the output of this expression.
</td>
</tr>

</table>



<h3 id="calculation_equal"><code>calculation_equal</code></h3>

<a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/expression_impl/promote.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>calculation_equal(
    expr: expression.Expression
) -> bool
</code></pre>

self.calculate is equal to another expression.calculate.

Given the same source node tensors, self.calculate(...) and
expression.calculate(...) will have the same result.

Note that this does not check that the source expressions of the two
expressions are the same. Therefore, two operations can have the same
calculation, but not the same output, because their sources are different.
For example, if a.calculation_is_identity() is True and
b.calculation_is_identity() is True, then a.calculation_equal(b) is True.
However, unless a and b have the same source, the expressions themselves are
not equal.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`expression`
</td>
<td>
The expression to compare to.
</td>
</tr>
</table>



<h3 id="calculation_is_identity"><code>calculation_is_identity</code></h3>

<a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/expression_impl/promote.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>calculation_is_identity() -> bool
</code></pre>

True iff the self.calculate is the identity.

There is exactly one source, and the output of self.calculate(...) is the
node tensor of this source.

<h3 id="cogroup_by_index"><code>cogroup_by_index</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cogroup_by_index(
    source_path: CoercableToPath,
    left_name: path.Step,
    right_name: path.Step,
    new_field_name: path.Step
) -> "Expression"
</code></pre>

Creates a cogroup of left_name and right_name at new_field_name.


<h3 id="create_has_field"><code>create_has_field</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>create_has_field(
    source_path: CoercableToPath,
    new_field_name: path.Step
) -> "Expression"
</code></pre>

Creates a field that is the presence of the source path.


<h3 id="create_proto_index"><code>create_proto_index</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>create_proto_index(
    field_name: path.Step
) -> "Expression"
</code></pre>

Creates a proto index field as a direct child of the current root.

The proto index maps each root element to the original batch index.
For example: [0, 2] means the first element came from the first proto
in the original input tensor and the second element came from the third
proto. The created field is always "dense" -- it has the same valency as
the current root.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`field_name`
</td>
<td>
the name of the field to be created.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
An Expression object representing the result of the operation.
</td>
</tr>

</table>



<h3 id="create_size_field"><code>create_size_field</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>create_size_field(
    source_path: CoercableToPath,
    new_field_name: path.Step
) -> "Expression"
</code></pre>

Creates a field that is the size of the source path.


<h3 id="get_child"><code>get_child</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_child(
    field_name: path.Step
) -> Optional['Expression']
</code></pre>

Gets a named child.


<h3 id="get_child_or_error"><code>get_child_or_error</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_child_or_error(
    field_name: path.Step
) -> "Expression"
</code></pre>

Gets a named child.


<h3 id="get_descendant"><code>get_descendant</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_descendant(
    p: path.Path
) -> Optional['Expression']
</code></pre>

Finds the descendant at the path.


<h3 id="get_descendant_or_error"><code>get_descendant_or_error</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_descendant_or_error(
    p: path.Path
) -> "Expression"
</code></pre>

Finds the descendant at the path.


<h3 id="get_known_children"><code>get_known_children</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_known_children() -> Mapping[path.Step, 'Expression']
</code></pre>




<h3 id="get_known_descendants"><code>get_known_descendants</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_known_descendants() -> Mapping[path.Path, 'Expression']
</code></pre>

Gets a mapping from known paths to subexpressions.

The difference between this and get_descendants in Prensor is that
all paths in a Prensor are realized, thus all known. But an Expression's
descendants might not all be known at the point this method is called,
because an expression may have an infinite number of children.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A mapping from paths (relative to the root of the subexpression) to
expressions.
</td>
</tr>

</table>



<h3 id="get_paths_with_schema"><code>get_paths_with_schema</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_paths_with_schema() -> List[path.Path]
</code></pre>

Extract only paths that contain schema information.


<h3 id="get_schema"><code>get_schema</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_schema(
    create_schema_features=True
) -> schema_pb2.Schema
</code></pre>

Returns a schema for the entire tree.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`create_schema_features`
</td>
<td>
If True, schema features are added for all
children and a schema entry is created if not available on the child. If
False, features are left off of the returned schema if there is no
schema_feature on the child.
</td>
</tr>
</table>



<h3 id="get_source_expressions"><code>get_source_expressions</code></h3>

<a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/expression_impl/promote.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_source_expressions() -> Sequence[expression.Expression]
</code></pre>

Gets the sources of this expression.

The node tensors of the source expressions must be sufficient to
calculate the node tensor of this expression
(see calculate and calculate_value_slowly).

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The sources of this expression.
</td>
</tr>

</table>



<h3 id="known_field_names"><code>known_field_names</code></h3>

<a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/expression_impl/promote.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>known_field_names() -> FrozenSet[path.Step]
</code></pre>

Returns known field names of the expression.


Known field names of a parsed proto correspond to the fields declared in
the message. Examples of "unknown" fields are extensions and explicit casts
in an any field. The only way to know if an unknown field "(foo.bar)" is
present in an expression expr is to call (expr["(foo.bar)"] is not None).

Notice that simply accessing a field does not make it "known". However,
setting a field (or setting a descendant of a field) will make it known.

project(...) returns an expression where the known field names are the only
field names. In general, if you want to depend upon known_field_names
(e.g., if you want to compile a expression), then the best approach is to
project() the expression first.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
An immutable set of field names.
</td>
</tr>

</table>



<h3 id="map_field_values"><code>map_field_values</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>map_field_values(
    source_path: CoercableToPath,
    operator: Callable[[tf.Tensor], tf.Tensor],
    dtype: tf.DType,
    new_field_name: path.Step
) -> "Expression"
</code></pre>

Map a primitive field to create a new primitive field.

Note: the dtype argument is added since the v1 API.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`source_path`
</td>
<td>
the origin path.
</td>
</tr><tr>
<td>
`operator`
</td>
<td>
an element-wise operator that takes a 1-dimensional vector.
</td>
</tr><tr>
<td>
`dtype`
</td>
<td>
the type of the output.
</td>
</tr><tr>
<td>
`new_field_name`
</td>
<td>
the name of a new sibling of source_path.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
the resulting root expression.
</td>
</tr>

</table>



<h3 id="map_ragged_tensors"><code>map_ragged_tensors</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>map_ragged_tensors(
    parent_path: CoercableToPath,
    source_fields: Sequence[path.Step],
    operator: Callable[..., tf.SparseTensor],
    is_repeated: bool,
    dtype: tf.DType,
    new_field_name: path.Step
) -> "Expression"
</code></pre>

Maps a set of primitive fields of a message to a new field.

Unlike map_field_values, this operation allows you to some degree reshape
the field. For instance, you can take two optional fields and create a
repeated field, or perform a reduce_sum on the last dimension of a repeated
field and create an optional field. The key constraint is that the operator
must return a sparse tensor of the correct dimension: i.e., a
2D sparse tensor if is_repeated is true, or a 1D sparse tensor if
is_repeated is false. Moreover, the first dimension of the sparse tensor
must be equal to the first dimension of the input tensor.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`parent_path`
</td>
<td>
the parent of the input and output fields.
</td>
</tr><tr>
<td>
`source_fields`
</td>
<td>
the nonempty list of names of the source fields.
</td>
</tr><tr>
<td>
`operator`
</td>
<td>
an operator that takes len(source_fields) sparse tensors and
returns a sparse tensor of the appropriate shape.
</td>
</tr><tr>
<td>
`is_repeated`
</td>
<td>
whether the output is repeated.
</td>
</tr><tr>
<td>
`dtype`
</td>
<td>
the dtype of the result.
</td>
</tr><tr>
<td>
`new_field_name`
</td>
<td>
the name of the resulting field.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A new query.
</td>
</tr>

</table>



<h3 id="map_sparse_tensors"><code>map_sparse_tensors</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>map_sparse_tensors(
    parent_path: CoercableToPath,
    source_fields: Sequence[path.Step],
    operator: Callable[..., tf.SparseTensor],
    is_repeated: bool,
    dtype: tf.DType,
    new_field_name: path.Step
) -> "Expression"
</code></pre>

Maps a set of primitive fields of a message to a new field.

Unlike map_field_values, this operation allows you to some degree reshape
the field. For instance, you can take two optional fields and create a
repeated field, or perform a reduce_sum on the last dimension of a repeated
field and create an optional field. The key constraint is that the operator
must return a sparse tensor of the correct dimension: i.e., a
2D sparse tensor if is_repeated is true, or a 1D sparse tensor if
is_repeated is false. Moreover, the first dimension of the sparse tensor
must be equal to the first dimension of the input tensor.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`parent_path`
</td>
<td>
the parent of the input and output fields.
</td>
</tr><tr>
<td>
`source_fields`
</td>
<td>
the nonempty list of names of the source fields.
</td>
</tr><tr>
<td>
`operator`
</td>
<td>
an operator that takes len(source_fields) sparse tensors and
returns a sparse tensor of the appropriate shape.
</td>
</tr><tr>
<td>
`is_repeated`
</td>
<td>
whether the output is repeated.
</td>
</tr><tr>
<td>
`dtype`
</td>
<td>
the dtype of the result.
</td>
</tr><tr>
<td>
`new_field_name`
</td>
<td>
the name of the resulting field.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A new query.
</td>
</tr>

</table>



<h3 id="project"><code>project</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>project(
    path_list: Sequence[CoercableToPath]
) -> "Expression"
</code></pre>

Constrains the paths to those listed.


<h3 id="promote"><code>promote</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>promote(
    source_path: CoercableToPath,
    new_field_name: path.Step
)
</code></pre>

Promotes source_path to be a field new_field_name in its grandparent.


<h3 id="promote_and_broadcast"><code>promote_and_broadcast</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>promote_and_broadcast(
    path_dictionary: Mapping[path.Step, CoercableToPath],
    dest_path_parent: CoercableToPath
) -> "Expression"
</code></pre>




<h3 id="reroot"><code>reroot</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>reroot(
    new_root: CoercableToPath
) -> "Expression"
</code></pre>

Returns a new list of protocol buffers available at new_root.


<h3 id="schema_string"><code>schema_string</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>schema_string(
    limit: Optional[int] = None
) -> str
</code></pre>

Returns a schema for the expression.

E.g.

repeated root:
  optional int32 foo
  optional bar:
    optional string baz
  optional int64 bak

Note that unknown fields and subexpressions are not displayed.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`limit`
</td>
<td>
if present, limit the recursion.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A string, describing (a part of) the schema.
</td>
</tr>

</table>



<h3 id="slice"><code>slice</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>slice(
    source_path: CoercableToPath,
    new_field_name: path.Step,
    begin: Optional[IndexValue] = None,
    end: Optional[IndexValue] = None
) -> "Expression"
</code></pre>

Creates a slice copy of source_path at new_field_path.

Note that if begin or end is negative, it is considered relative to
the size of the array. e.g., slice(...,begin=-1) will get the last
element of every array.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`source_path`
</td>
<td>
the source of the slice.
</td>
</tr><tr>
<td>
`new_field_name`
</td>
<td>
the new field that is generated.
</td>
</tr><tr>
<td>
`begin`
</td>
<td>
the beginning of the slice (inclusive).
</td>
</tr><tr>
<td>
`end`
</td>
<td>
the end of the slice (exclusive).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
An Expression object representing the result of the operation.
</td>
</tr>

</table>



<h3 id="truncate"><code>truncate</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>truncate(
    source_path: CoercableToPath,
    limit: Union[int, tf.Tensor],
    new_field_name: path.Step
) -> "Expression"
</code></pre>

Creates a truncated copy of source_path at new_field_path.


<h3 id="__eq__"><code>__eq__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__eq__(
    expr: "Expression"
) -> bool
</code></pre>

if hash(expr1) == hash(expr2): then expr1 == expr2.

Do not override this method.
Args:
  expr: The expression to check equality against

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
Boolean of equality of two expressions
</td>
</tr>

</table>





