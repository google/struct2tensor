description: Promote an expression to be a child of its grandparent.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="expression_impl.promote" />
<meta itemprop="path" content="Stable" />
</div>

# Module: expression_impl.promote

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/expression_impl/promote.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Promote an expression to be a child of its grandparent.


Promote is part of the standard flattening of data, promote_and_broadcast,
which takes structured data and flattens it. By directly accessing promote,
one can perform simpler operations.

For example, suppose an expr represents:

```
+
|
+-session*   (stars indicate repeated)
     |
     +-event*
         |
         +-val*-int64

session: {
  event: {
    val: 111
  }
  event: {
    val: 121
    val: 122
  }
}

session: {
  event: {
    val: 10
    val: 7
  }
  event: {
    val: 1
  }
}

```

```
promote.promote(expr, path.Path(["session", "event", "val"]), nval)
```

produces:

```
+
|
+-session*   (stars indicate repeated)
     |
     +-event*
     |    |
     |    +-val*-int64
     |
     +-nval*-int64

session: {
  event: {
    val: 111
  }
  event: {
    val: 121
    val: 122
  }
  nval: 111
  nval: 121
  nval: 122
}

session: {
  event: {
    val: 10
    val: 7
  }
  event: {
    val: 1
  }
  nval: 10
  nval: 7
  nval: 1
}
```

## Classes

[`class PromoteChildExpression`](../expression_impl/promote/PromoteChildExpression.md): The root of the promoted sub tree.

[`class PromoteExpression`](../expression_impl/promote/PromoteExpression.md): A promoted leaf.

## Functions

[`promote(...)`](../expression_impl/promote/promote.md): Promote a path to be a child of its grandparent, and give it a name.

[`promote_anonymous(...)`](../expression_impl/promote/promote_anonymous.md): Promote a path to be a new anonymous child of its grandparent.

