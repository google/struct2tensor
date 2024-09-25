description: Arbitrary operations from prensors to prensors in an expression.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="expression_impl.map_prensor_to_prensor" />
<meta itemprop="path" content="Stable" />
</div>

# Module: expression_impl.map_prensor_to_prensor

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/expression_impl/map_prensor_to_prensor.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Arbitrary operations from prensors to prensors in an expression.


This is useful if a single op generates an entire structure. In general, it is
better to use the existing expressions framework or design a custom expression
than use this op. So long as any of the output is required, all of the input
is required.

For example, suppose you have an op my_op, that takes a prensor of the form:

```
  event
   / \
 foo   bar
```

and produces a prensor of the form my_result_schema:

```
   event
    / \
 foo2 bar2
```

```
my_result_schema = create_schema(
    is_repeated=True,
    children={"foo2":{is_repeated:True, dtype:tf.int64},
              "bar2":{is_repeated:False, dtype:tf.int64}})
```

If you give it an expression original with the schema:

```
 session
    |
  event
  /  \
foo   bar

result = map_prensor_to_prensor(
  original,
  path.Path(["session","event"]),
  my_op,
  my_result_schema)
```

Result will have the schema:

```
 session
    |
  event--------
  /  \    \    \
foo   bar foo2 bar2
```

## Classes

[`class Schema`](../expression_impl/map_prensor_to_prensor/Schema.md): A finite schema for a prensor.

## Functions

[`create_schema(...)`](../expression_impl/map_prensor_to_prensor/create_schema.md): Create a schema recursively.

[`map_prensor_to_prensor(...)`](../expression_impl/map_prensor_to_prensor/map_prensor_to_prensor.md): Maps an expression to a prensor, and merges that prensor.

