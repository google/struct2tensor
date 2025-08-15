description: get_positional_index and get_index_from_end methods.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="expression_impl.index" />
<meta itemprop="path" content="Stable" />
</div>

# Module: expression_impl.index

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/expression_impl/index.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



get_positional_index and get_index_from_end methods.


The parent_index identifies the index of the parent of each element. These
methods take the parent_index to determine the relationship with respect to
other elements.

#### Given:



```
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
get_positional_index(expr, path.Path(["event","val"]), "val_index")
```

yields:

```
session: {
  event: {
    val: 111
    val_index: 0
  }
  event: {
    val: 121
    val: 122
    val_index: 0
    val_index: 1
  }
}

session: {
  event: {
    val: 10
    val: 7
    val_index: 0
    val_index: 1
  }
  event: {
    val: 1
    val_index: 0
  }
}
```

```
get_index_from_end(expr, path.Path(["event","val"]), "neg_val_index")
```
yields:

```
session: {
  event: {
    val: 111
    neg_val_index: -1
  }
  event: {
    val: 121
    val: 122
    neg_val_index: -2
    neg_val_index: -1
  }
}

session: {
  event: {
    val: 10
    val: 7
    neg_val_index: 2
    neg_val_index: -1
  }
  event: {
    val: 1
    neg_val_index: -1
  }
}
```

These methods are useful when you want to depend upon the index of a field.
For example, if you want to filter examples based upon their index, or
cogroup two fields by index, then first creating the index is useful.

Note that while the parent indices of these fields seem like overhead, they
are just references to the parent indices of other fields, and are therefore
take little memory or CPU.

## Functions

[`get_index_from_end(...)`](../expression_impl/index/get_index_from_end.md): Gets the number of steps from the end of the array.

[`get_positional_index(...)`](../expression_impl/index/get_positional_index.md): Gets the positional index.
