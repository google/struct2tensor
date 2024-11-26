description: project selects a subtree of an expression.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="expression_impl.project" />
<meta itemprop="path" content="Stable" />
</div>

# Module: expression_impl.project

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/expression_impl/project.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



project selects a subtree of an expression.


project is often used right before calculating the value.

#### Example:



```
expr = ...
new_expr = project.project(expr, [path.Path(["foo","bar"]),
                                  path.Path(["x", "y"])])
[prensor_result] = calculate.calculate_prensors([new_expr])
```

prensor_result now has two paths, "foo.bar" and "x.y".

## Functions

[`project(...)`](../expression_impl/project/project.md): select a subtree.
