description: Promotes then broadcasts the origin until its parent is new_parent.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="expression_impl.promote_and_broadcast.promote_and_broadcast_anonymous" />
<meta itemprop="path" content="Stable" />
</div>

# expression_impl.promote_and_broadcast.promote_and_broadcast_anonymous

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/expression_impl/promote_and_broadcast.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Promotes then broadcasts the origin until its parent is new_parent.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>expression_impl.promote_and_broadcast.promote_and_broadcast_anonymous(
    root: expression.Expression,
    origin: path.Path,
    new_parent: path.Path
) -> Tuple[expression.Expression, path.Path]
</code></pre>



<!-- Placeholder for "Used in" -->
