description: Arbitrary operations from sparse and ragged tensors to a leaf field.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="expression_impl.map_prensor" />
<meta itemprop="path" content="Stable" />
</div>

# Module: expression_impl.map_prensor

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/expression_impl/map_prensor.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Arbitrary operations from sparse and ragged tensors to a leaf field.


There are two public methods of note right now: map_sparse_tensor
and map_ragged_tensor.

#### Assume expr is:



```
session: {
  event: {
    val_a: 10
    val_b: 1
  }
  event: {
    val_a: 20
    val_b: 2
  }
  event: {
  }
  event: {
    val_a: 40
  }
  event: {
    val_b: 5
  }
}
```

Either of the following alternatives will add val_a and val_b
to create val_sum.

map_sparse_tensor converts val_a and val_b to sparse tensors,
and then add them to produce val_sum.

```
new_root = map_prensor.map_sparse_tensor(
    expr,
    path.Path(["event"]),
    [path.Path(["val_a"]), path.Path(["val_b"])],
    lambda x,y: x + y,
    False,
    tf.int32,
    "val_sum")
```

map_ragged_tensor converts val_a and val_b to ragged tensors,
and then add them to produce val_sum.

```
new_root = map_prensor.map_ragged_tensor(
    expr,
    path.Path(["event"]),
    [path.Path(["val_a"]), path.Path(["val_b"])],
    lambda x,y: x + y,
    False,
    tf.int32,
    "val_sum")
```

The result of either is:

```
session: {
  event: {
    val_a: 10
    val_b: 1
    val_sum: 11
  }
  event: {
    val_a: 20
    val_b: 2
    val_sum: 22
  }
  event: {
  }
  event: {
    val_a: 40
    val_sum: 40
  }
  event: {
    val_b: 5
    val_sum: 5
  }
}
```

## Functions

[`map_ragged_tensor(...)`](../expression_impl/map_prensor/map_ragged_tensor.md): Map a ragged tensor.

[`map_sparse_tensor(...)`](../expression_impl/map_prensor/map_sparse_tensor.md): Maps a sparse tensor.

