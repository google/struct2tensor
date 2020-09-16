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

[`apply_schema`](./expression_impl/apply_schema.md) module: Apply a schema to an expression.

[`broadcast`](./expression_impl/broadcast.md) module: Methods for broadcasting a path in a tree.

[`depth_limit`](./expression_impl/depth_limit.md) module: Caps the depth of an expression.

[`filter_expression`](./expression_impl/filter_expression.md) module: Create a new expression that is a filtered version of an original one.

[`index`](./expression_impl/index.md) module: get_positional_index and get_index_from_end methods.

[`map_prensor`](./expression_impl/map_prensor.md) module: Arbitrary operations from sparse and ragged tensors to a leaf field.

[`map_prensor_to_prensor`](./expression_impl/map_prensor_to_prensor.md) module: Arbitrary operations from prensors to prensors in an expression.

[`map_values`](./expression_impl/map_values.md) module: Maps the values of various leaves of the same child to a single result.

[`parquet`](./expression_impl/parquet.md) module: Apache Parquet Dataset.

[`placeholder`](./expression_impl/placeholder.md) module: Placeholder expression.

[`project`](./expression_impl/project.md) module: project selects a subtree of an expression.

[`promote`](./expression_impl/promote.md) module: Promote an expression to be a child of its grandparent.

[`promote_and_broadcast`](./expression_impl/promote_and_broadcast.md) module: promote_and_broadcast a set of nodes.

[`proto`](./expression_impl/proto.md) module: Expressions to parse a proto.

[`reroot`](./expression_impl/reroot.md) module: Reroot to a subtree, maintaining an input proto index.

[`size`](./expression_impl/size.md) module: Functions for creating new size or has expression.

[`slice_expression`](./expression_impl/slice_expression.md) module: Implementation of slice.

