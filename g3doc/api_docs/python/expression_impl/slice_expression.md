description: Implementation of slice.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="expression_impl.slice_expression" />
<meta itemprop="path" content="Stable" />
</div>

# Module: expression_impl.slice_expression

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/expression_impl/slice_expression.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Implementation of slice.



The slice operation is meant to replicate the slicing of a list in python.

Slicing a list in python is done by specifying a beginning and ending.
The resulting list consists of all elements in the range.

#### For example:



```
```
>>> x = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
>>> print(x[2:5]) # all elements between index 2 inclusive and index 5 exclusive
['c', 'd', 'e']
>>> print(x[2:]) # all elements between index 2 and the end.
['c', 'd', 'e', 'f', 'g']
>>> print(x[:4]) # all elements between the beginning and index 4 (exclusive).
['a', 'b', 'c', 'd']
>>> print(x[-3:-1]) # all elements starting three from the end.
>>>                 # until one from the end (exclusive).
['e', 'f']
>>> print(x[-3:6]) # all elements starting three from the end
                   # until index 6 exclusive.
['e', 'f', 'g']
```
```


over the elements (e.g. x[2:6:2]=['c', 'e'], giving you every other element.
This is not implemented.


A prensor can be considered to be interleaved lists and dictionaries.
E.g.:

```
my_expression = [{
  "foo":[
    {"bar":[
      {"baz":["a","b","c", "d"]},
      {"baz":["d","e","f"]}
      ]
    },
    {"bar":[
      {"baz":["g","h","i"]},
      {"baz":["j","k","l", ]}
      {"baz":["m"]}
    ]
    }]
}]
```

```
result_1 = slice_expression.slice_expression(
  my_expression, "foo.bar", "new_bar",begin=1, end=3)

result_1 = [{
  "foo":[
    {"bar":[
      {"baz":["a","b","c", "d"]},
      {"baz":["d","e","f"]}
      ],
     "new_bar":[
      {"baz":["d","e","f"]}
      ]
    },
    {"bar":[
      {"baz":["g","h","i"]},
      {"baz":["j","k","l", ]}
      {"baz":["m", ]}
     ],
    "new_bar":[
      {"baz":["j","k","l", ]}
      {"baz":["m", ]}
    ]
    }]
}]
```

```
result_2 = slice_expression.slice_expression(
  my_expression, "foo.bar.baz", "new_baz",begin=1, end=3)

result_2 = [{
  "foo":[
    {"bar":[
      {"baz":["a","b","c", "d"],
       "new_baz":["b","c"],
      },
      {"baz":["d","e","f"], "new_baz":["e","f"]}
      ]
    },
    {"bar":[
      {"baz":["g","h","i"], "new_baz":["h","i"]},
      {"baz":["j","k","l"], "new_baz":["k","l"]},
      {"baz":["m", ]}
      ]
    }]
}]
```

## Functions

[`slice_expression(...)`](../expression_impl/slice_expression/slice_expression.md): Creates a new subtree with a sliced expression.

## Type Aliases

[`IndexValue`](../expression_impl/slice_expression/IndexValue.md)
