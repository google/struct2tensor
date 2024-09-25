description: Apply a schema to an expression.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="expression_impl.apply_schema" />
<meta itemprop="path" content="Stable" />
</div>

# Module: expression_impl.apply_schema

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/expression_impl/apply_schema.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Apply a schema to an expression.


A tensorflow metadata schema (
detailed information about the data: specifically, it presents domain
information (e.g., not just integers, but integers between 0 and 10), and more
detailed structural information (e.g., this field occurs in at least 70% of its
parents, and when it occurs, it shows up 5 to 7 times).

Applying a schema attaches a tensorflow metadata schema to an expression:
namely, it aligns the features in the schema with the expression's children by
name (possibly recursively).

After applying a schema to an expression, one can use promote, broadcast, et
cetera, and the schema for new expressions will be inferred. If you write a
custom expression, you can write code that determines the schema information of
the result.

To get the schema back, call get_schema().

This does not filter out fields not in the schema.


my_expr = ...
my_schema = ...schema here...
my_new_schema = my_expr.apply_schema(my_schema).get_schema()
my_new_schema has semantically identical information on the fields as my_schema.


1. Get the (non-deprecated) paths from a schema.
2. Check if any paths in the schema are not in the expression.
3. Check if any paths in the expression are not in the schema.
4. Project the expression to paths in the schema.

## Functions

[`apply_schema(...)`](../expression_impl/apply_schema/apply_schema.md)

