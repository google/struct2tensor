# Module: s2t

<div class="buttons-wrapper">
  <a class="md-button" target="_blank" href=
	  "https://github.com/google/struct2tensor/blob/master/struct2tensor/__init__.py">
    <div class="buttons-content">
      <img width="32px" src=
	   "https://www.tensorflow.org/images/GitHub-Mark-32px.png">
      View source on GitHub
    </div>
  </a>
</div>

Import core names for struct2tensor.

## Classes

[`class ChildNodeTensor`][struct2tensor.ChildNodeTensor]: The value of an intermediate node.

[`class Expression`][struct2tensor.Expression]: An expression represents the calculation of a prensor object.

[`class LeafNodeTensor`][struct2tensor.LeafNodeTensor]: The value of a leaf node.

[`class Path`][struct2tensor.Path]: A representation of a path in the expression.

[`class Prensor`][struct2tensor.Prensor]: A expression of NodeTensor objects.

[`class RootNodeTensor`][struct2tensor.RootNodeTensor]: The value of the root.

## Functions

[`calculate_prensors(...)`][struct2tensor.calculate_prensors]: Gets the prensor value of the expressions.

[`calculate_prensors_with_graph(...)`][struct2tensor.calculate_prensors_with_graph]: Gets the prensor value of the expressions and the graph used.

[`calculate_prensors_with_source_paths(...)`][struct2tensor.calculate_prensors_with_source_paths]: Returns a list of prensor trees, and proto summaries.

[`create_expression_from_file_descriptor_set(...)`][struct2tensor.create_expression_from_file_descriptor_set]: Create an expression from a 1D tensor of serialized protos.

[`create_expression_from_prensor(...)`][struct2tensor.create_expression_from_prensor]: Gets an expression representing the prensor.

[`create_expression_from_proto(...)`][struct2tensor.create_expression_from_proto]: Create an expression from a 1D tensor of serialized protos.

[`create_path(...)`][struct2tensor.create_path]: Create a path from an object.

[`create_prensor_from_descendant_nodes(...)`][struct2tensor.create_prensor_from_descendant_nodes]: Create a prensor from a map of paths to NodeTensor.

[`create_prensor_from_root_and_children(...)`][struct2tensor.create_prensor_from_root_and_children]

[`get_default_options(...)`][struct2tensor.get_default_options]: Get the default options.

[`get_options_with_minimal_checks(...)`][struct2tensor.get_options_with_minimal_checks]: Options for calculation with minimal runtime checks.

[`get_ragged_tensor(...)`][struct2tensor.get_ragged_tensor]: Get a ragged tensor for a path. (deprecated)

[`get_ragged_tensors(...)`][struct2tensor.get_ragged_tensors]: Gets ragged tensors for all the leaves of the prensor expression. (deprecated)

[`get_sparse_tensor(...)`][struct2tensor.get_sparse_tensor]: Gets a sparse tensor for path p. (deprecated)

[`get_sparse_tensors(...)`][struct2tensor.get_sparse_tensors]: Gets sparse tensors for all the leaves of the prensor expression. (deprecated)

## Type Aliases

[`NodeTensor`][struct2tensor.NodeTensor]

[`Step`][struct2tensor.Step]

