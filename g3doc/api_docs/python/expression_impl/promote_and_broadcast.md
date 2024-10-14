description: promote_and_broadcast a set of nodes.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="expression_impl.promote_and_broadcast" />
<meta itemprop="path" content="Stable" />
</div>

# Module: expression_impl.promote_and_broadcast

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/expression_impl/promote_and_broadcast.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



promote_and_broadcast a set of nodes.


For example, suppose an expr represents:

```
+
|
+-session*   (stars indicate repeated)
     |
     +-event*
     |   |
     |   +-val*-int64
     |
     +-user_info? (question mark indicates optional)
           |
           +-age? int64

session: {
  event: {
    val: 1
  }
  event: {
    val: 4
    val: 5
  }
  user_info: {
    age: 25
  }
}

session: {
  event: {
    val: 7
  }
  event: {
    val: 8
    val: 9
  }
  user_info: {
    age: 20
  }
}
```

```
promote_and_broadcast.promote_and_broadcast(
    path.Path(["event"]),{"nage":path.Path(["user_info","age"])})
```

creates:

```
+
|
+-session*   (stars indicate repeated)
     |
     +-event*
     |   |
     |   +-val*-int64
     |   |
     |   +-nage*-int64
     |
     +-user_info? (question mark indicates optional)
           |
           +-age? int64

session: {
  event: {
    nage: 25
    val: 1
  }
  event: {
    nage: 25
    val: 4
    val: 5
  }
  user_info: {
    age: 25
  }
}

session: {
  event: {
    nage: 20
    val: 7
  }
  event: {
    nage: 20
    val: 8
    val: 9
  }
  user_info: {
    age: 20
  }
}
```

## Functions

[`promote_and_broadcast(...)`](../expression_impl/promote_and_broadcast/promote_and_broadcast.md): Promote and broadcast a set of paths to a particular location.

[`promote_and_broadcast_anonymous(...)`](../expression_impl/promote_and_broadcast/promote_and_broadcast_anonymous.md): Promotes then broadcasts the origin until its parent is new_parent.
