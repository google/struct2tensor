description: Functions for creating new size or has expression.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="expression_impl.size" />
<meta itemprop="path" content="Stable" />
</div>

# Module: expression_impl.size

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/expression_impl/size.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Functions for creating new size or has expression.


Given a field "foo.bar",

```
root = size(expr, path.Path(["foo","bar"]), "bar_size")
```

creates a new expression root that has an optional field "foo.bar_size", which
is always present, and contains the number of bar in a particular foo.

```
root_2 = has(expr, path.Path(["foo","bar"]), "bar_has")
```

creates a new expression root that has an optional field "foo.bar_has", which
is always present, and is true if there are one or more bar in foo.

## Classes

[`class SizeExpression`](../expression_impl/size/SizeExpression.md): Size of the given expression.

## Functions

[`has(...)`](../expression_impl/size/has.md): Get the has of a field as a new sibling field.

[`size(...)`](../expression_impl/size/size.md): Get the size of a field as a new sibling field.

[`size_anonymous(...)`](../expression_impl/size/size_anonymous.md): Calculate the size of a field, and store it as an anonymous sibling.
