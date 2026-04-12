description: Import core names for struct2tensor.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="s2t" />
<meta itemprop="path" content="Stable" />
</div>

# Module: s2t

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/__init__.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Import core names for struct2tensor.



## Classes

[`class ChildNodeTensor`](./s2t/ChildNodeTensor.md): The value of an intermediate node.

[`class Expression`](./s2t/Expression.md): An expression represents the calculation of a prensor object.

[`class LeafNodeTensor`](./s2t/LeafNodeTensor.md): The value of a leaf node.

[`class Path`](./s2t/Path.md): A representation of a path in the expression.

[`class Prensor`](./s2t/Prensor.md): A expression of NodeTensor objects.

[`class RootNodeTensor`](./s2t/RootNodeTensor.md): The value of the root.

## Functions

[`calculate_prensors(...)`](./s2t/calculate_prensors.md): Gets the prensor value of the expressions.

[`calculate_prensors_with_graph(...)`](./s2t/calculate_prensors_with_graph.md): Gets the prensor value of the expressions and the graph used.

[`calculate_prensors_with_source_paths(...)`](./s2t/calculate_prensors_with_source_paths.md): Returns a list of prensor trees, and proto summaries.

[`create_expression_from_file_descriptor_set(...)`](./s2t/create_expression_from_file_descriptor_set.md): Create an expression from a 1D tensor of serialized protos.

[`create_expression_from_prensor(...)`](./s2t/create_expression_from_prensor.md): Gets an expression representing the prensor.

[`create_expression_from_proto(...)`](./s2t/create_expression_from_proto.md): Create an expression from a 1D tensor of serialized protos.

[`create_path(...)`](./s2t/create_path.md): Create a path from an object.

[`create_prensor_from_descendant_nodes(...)`](./s2t/create_prensor_from_descendant_nodes.md): Create a prensor from a map of paths to NodeTensor.

[`create_prensor_from_root_and_children(...)`](./s2t/create_prensor_from_root_and_children.md)

[`get_default_options(...)`](./s2t/get_default_options.md): Get the default options.

[`get_options_with_minimal_checks(...)`](./s2t/get_options_with_minimal_checks.md): Options for calculation with minimal runtime checks.

[`get_ragged_tensor(...)`](./s2t/get_ragged_tensor.md): Get a ragged tensor for a path. (deprecated)

[`get_ragged_tensors(...)`](./s2t/get_ragged_tensors.md): Gets ragged tensors for all the leaves of the prensor expression. (deprecated)

[`get_sparse_tensor(...)`](./s2t/get_sparse_tensor.md): Gets a sparse tensor for path p. (deprecated)

[`get_sparse_tensors(...)`](./s2t/get_sparse_tensors.md): Gets sparse tensors for all the leaves of the prensor expression. (deprecated)

## Type Aliases

[`NodeTensor`](./s2t/NodeTensor.md)

[`Step`](./s2t/Step.md)
