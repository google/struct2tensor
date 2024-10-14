description: Create a new expression that is a filtered version of an original one.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="expression_impl.filter_expression" />
<meta itemprop="path" content="Stable" />
</div>

# Module: expression_impl.filter_expression

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/expression_impl/filter_expression.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Create a new expression that is a filtered version of an original one.


There are two public methods in this module: filter_by_sibling and
filter_by_child. As with most other operations, these create a new tree which
has all the original paths of the original tree, but with a new subtree.

filter_by_sibling allows you to filter an expression by a boolean sibling field.

Beginning with the struct:

```
root =
         -----*----------------------------------------------------
        /                       \                                  \
     root0                    root1-----------------------      root2 (empty)
      /   \                   /    \               \      \
      |  keep_my_sib0:False  |  keep_my_sib1:True   | keep_my_sib2:False
    doc0-----               doc1---------------    doc2--------
     |       \                \           \    \               \
    bar:"a"  keep_me:False    bar:"b" bar:"c" keep_me:True      bar:"d"

# Note, keep_my_sib and doc must have the same shape (e.g., each root
has the same number of keep_my_sib children as doc children).
root_2 = filter_expression.filter_by_sibling(
    root, path.create_path("doc"), "keep_my_sib", "new_doc")

End with the struct (suppressing original doc):
         -----*----------------------------------------------------
        /                       \                                  \
    root0                    root1------------------        root2 (empty)
        \                   /    \                  \
        keep_my_sib0:False  |  keep_my_sib1:True   keep_my_sib2:False
                           new_doc0-----------
                             \           \    \
                             bar:"b" bar:"c" keep_me:True
```

filter_by_sibling allows you to filter an expression by a optional boolean
child field.

The following call will have the same effect as above:

```
root_2 = filter_expression.filter_by_child(
    root, path.create_path("doc"), "keep_me", "new_doc")
```

## Functions

[`filter_by_child(...)`](../expression_impl/filter_expression/filter_by_child.md): Filter an expression by an optional boolean child field.

[`filter_by_sibling(...)`](../expression_impl/filter_expression/filter_by_sibling.md): Filter an expression by its sibling.
