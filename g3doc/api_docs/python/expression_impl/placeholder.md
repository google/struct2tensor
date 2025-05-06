description: Placeholder expression.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="expression_impl.placeholder" />
<meta itemprop="path" content="Stable" />
</div>

# Module: expression_impl.placeholder

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/expression_impl/placeholder.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Placeholder expression.


A placeholder expression represents prensor nodes, however a prensor is not
needed until calculate is called. This allows the user to apply expression
queries to a placeholder expression before having an actual prensor object.
When calculate is called on a placeholder expression (or a descendant of a
placeholder expression), the feed_dict will need to be passed in. Then calculate
will bind the prensor with the appropriate placeholder expression.

#### Sample usage:



```
placeholder_exp = placeholder.create_expression_from_schema(schema)
new_exp = expression_queries(placeholder_exp, ..)
result = calculate.calculate_values([new_exp],
                                    feed_dict={placeholder_exp: pren})
# placeholder_exp requires a feed_dict to be passed in when calculating
```

## Functions

[`create_expression_from_schema(...)`](../expression_impl/placeholder/create_expression_from_schema.md): Creates a placeholder expression from a parquet schema.

[`get_placeholder_paths_from_graph(...)`](../expression_impl/placeholder/get_placeholder_paths_from_graph.md): Gets all placeholder paths from an expression graph.

