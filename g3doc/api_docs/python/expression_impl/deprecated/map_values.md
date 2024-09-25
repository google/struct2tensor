description: Maps the values of various leaves of the same child to a single result.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="expression_impl.map_values" />
<meta itemprop="path" content="Stable" />
</div>

# Module: expression_impl.map_values

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/expression_impl/map_values.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Maps the values of various leaves of the same child to a single result.


All inputs must have the same shape (parent_index must be equal).

The output is given the same shape (output of function must be of equal length).

Note that the operations are on 1-D tensors (as opposed to scalars).

## Functions

[`map_many_values(...)`](../expression_impl/map_values/map_many_values.md): Map multiple sibling fields into a new sibling.

[`map_values(...)`](../expression_impl/map_values/map_values.md): Map field into a new sibling.

[`map_values_anonymous(...)`](../expression_impl/map_values/map_values_anonymous.md): Map field into a new sibling.

