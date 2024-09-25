description: Import all modules in expression_impl.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="expression_impl" />
<meta itemprop="path" content="Stable" />
</div>

# Module: expression_impl

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/expression_impl/__init__.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Import all modules in expression_impl.


The modules in this file should be accessed like the following:

```
import struct2tensor as s2t
from struct2tensor import expression_impl

s2t.expression_impl.apply_schema
```

## Modules

[`apply_schema`][struct2tensor.expression_impl.apply_schema] module: Apply a schema to an expression.

[`broadcast`][struct2tensor.expression_impl.broadcast] module: Methods for broadcasting a path in a tree.

[`depth_limit`][struct2tensor.expression_impl.depth_limit] module: Caps the depth of an expression.

[`filter_expression`][struct2tensor.expression_impl.filter_expression] module: Create a new expression that is a filtered version of an original one.

[`index`][struct2tensor.expression_impl.index] module: get_positional_index and get_index_from_end methods.

[`map_prensor`][struct2tensor.expression_impl.map_prensor] module: Arbitrary operations from sparse and ragged tensors to a leaf field.

[`map_prensor_to_prensor`][struct2tensor.expression_impl.map_prensor_to_prensor] module: Arbitrary operations from prensors to prensors in an expression.

[`map_values`][struct2tensor.expression_impl.map_values] module: Maps the values of various leaves of the same child to a single result.

[`parquet`][struct2tensor.expression_impl.parquet] module: Apache Parquet Dataset.

[`placeholder`][struct2tensor.expression_impl.placeholder] module: Placeholder expression.

[`project`][struct2tensor.expression_impl.project] module: project selects a subtree of an expression.

[`promote`][struct2tensor.expression_impl.promote] module: Promote an expression to be a child of its grandparent.

[`promote_and_broadcast`][struct2tensor.expression_impl.promote_and_broadcast] module: promote_and_broadcast a set of nodes.

[`proto`][struct2tensor.expression_impl.proto] module: Expressions to parse a proto.

[`reroot`][struct2tensor.expression_impl.reroot] module: Reroot to a subtree, maintaining an input proto index.

[`size`][struct2tensor.expression_impl.size] module: Functions for creating new size or has expression.

[`slice_expression`][struct2tensor.expression_impl.slice_expression] module: Implementation of slice.

