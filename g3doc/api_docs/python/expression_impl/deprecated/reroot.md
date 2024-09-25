description: Reroot to a subtree, maintaining an input proto index.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="expression_impl.reroot" />
<meta itemprop="path" content="Stable" />
</div>

# Module: expression_impl.reroot

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/expression_impl/reroot.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Reroot to a subtree, maintaining an input proto index.


reroot is similar to get_descendant_or_error. However, this method allows
you to call create_proto_index(...) later on, that gives you a reference to the
original proto.

## Functions

[`create_proto_index_field(...)`](../expression_impl/reroot/create_proto_index_field.md)

[`reroot(...)`](../expression_impl/reroot/reroot.md): Reroot to a new path, maintaining a input proto index.

