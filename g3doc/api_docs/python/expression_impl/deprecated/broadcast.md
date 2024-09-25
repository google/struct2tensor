description: Methods for broadcasting a path in a tree.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="expression_impl.broadcast" />
<meta itemprop="path" content="Stable" />
</div>

# Module: expression_impl.broadcast

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/expression_impl/broadcast.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Methods for broadcasting a path in a tree.


This provides methods for broadcasting a field anonymously (that is used in
promote_and_broadcast), or with an explicitly given name.

Suppose you have an expr representing:

```
+
|
+-session*   (stars indicate repeated)
     |
     +-event*
     |
     +-val*-int64

session: {
  event: {}
  event: {}
  val: 10
  val: 11
}
session: {
  event: {}
  event: {}
  val: 20
}
```

#### Then:



```
broadcast.broadcast(expr, path.Path(["session","val"]), "event", "nv")
```

becomes:

```
+
|
+---session*   (stars indicate repeated)
       |
       +-event*
       |   |
       |   +---nv*-int64
       |
       +-val*-int64

session: {
  event: {
    nv: 10
    nv:11
  }
  event: {
    nv: 10
    nv:11
  }
  val: 10
  val: 11
}
session: {
  event: {nv: 20}
  event: {nv: 20}
  val: 20
}
```

## Functions

[`broadcast(...)`](../expression_impl/broadcast/broadcast.md)

[`broadcast_anonymous(...)`](../expression_impl/broadcast/broadcast_anonymous.md)

