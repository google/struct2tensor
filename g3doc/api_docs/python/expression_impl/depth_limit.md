description: Caps the depth of an expression.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="expression_impl.depth_limit" />
<meta itemprop="path" content="Stable" />
</div>

# Module: expression_impl.depth_limit

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/expression_impl/depth_limit.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Caps the depth of an expression.


Suppose you have an expression expr modeled as:

```
  *
   \
    A
   / \
  D   B
       \
        C
```

if expr_2 = depth_limit.limit_depth(expr, 2)
You get:

```
  *
   \
    A
   / \
  D   B
```

## Functions

[`limit_depth(...)`](../expression_impl/depth_limit/limit_depth.md): Limit the depth to nodes k steps from expr.
