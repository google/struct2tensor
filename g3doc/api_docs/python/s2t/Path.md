description: A representation of a path in the expression.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="s2t.Path" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__ge__"/>
<meta itemprop="property" content="__gt__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__le__"/>
<meta itemprop="property" content="__len__"/>
<meta itemprop="property" content="__lt__"/>
<meta itemprop="property" content="__ne__"/>
<meta itemprop="property" content="as_proto"/>
<meta itemprop="property" content="concat"/>
<meta itemprop="property" content="get_child"/>
<meta itemprop="property" content="get_least_common_ancestor"/>
<meta itemprop="property" content="get_parent"/>
<meta itemprop="property" content="is_ancestor"/>
<meta itemprop="property" content="prefix"/>
<meta itemprop="property" content="suffix"/>
</div>

# s2t.Path

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/path.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A representation of a path in the expression.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>s2t.Path(
    field_list: Sequence[<a href="../s2t/Step.md"><code>s2t.Step</code></a>]
)
</code></pre>



<!-- Placeholder for "Used in" -->

Do not implement __nonzero__, __eq__, __ne__, et cetera as these are
implicitly defined by __cmp__ and __len__.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`field_list`
</td>
<td>
a list or tuple of fields leading from one node to another.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
if any field is not a valid step (see is_valid_step).
</td>
</tr>
</table>



## Methods

<h3 id="as_proto"><code>as_proto</code></h3>

<a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/path.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>as_proto()
</code></pre>

Serialize a path as a proto.

This fails if there are any anonymous fields.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
a Path proto.
</td>
</tr>

</table>



<h3 id="concat"><code>concat</code></h3>

<a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/path.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>concat(
    other_path: "Path"
) -> "Path"
</code></pre>




<h3 id="get_child"><code>get_child</code></h3>

<a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/path.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_child(
    field_name: <a href="../s2t/Step.md"><code>s2t.Step</code></a>
) -> "Path"
</code></pre>

Get the child path.


<h3 id="get_least_common_ancestor"><code>get_least_common_ancestor</code></h3>

<a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/path.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_least_common_ancestor(
    other: "Path"
) -> "Path"
</code></pre>

Get the least common ancestor, the longest shared prefix.


<h3 id="get_parent"><code>get_parent</code></h3>

<a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/path.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_parent() -> "Path"
</code></pre>

Get the parent path.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The parent path.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
If this is the root path.
</td>
</tr>
</table>



<h3 id="is_ancestor"><code>is_ancestor</code></h3>

<a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/path.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>is_ancestor(
    other: "Path"
) -> bool
</code></pre>

True if self is ancestor of other (i.e. a prefix).


<h3 id="prefix"><code>prefix</code></h3>

<a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/path.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>prefix(
    ending_index: int
) -> "Path"
</code></pre>




<h3 id="suffix"><code>suffix</code></h3>

<a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/path.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>suffix(
    starting_index: int
) -> "Path"
</code></pre>




<h3 id="__eq__"><code>__eq__</code></h3>

<a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/path.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__eq__(
    other: "Path"
) -> bool
</code></pre>

Return self==value.


<h3 id="__ge__"><code>__ge__</code></h3>

<a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/path.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__ge__(
    other: "Path"
) -> bool
</code></pre>

Return self>=value.


<h3 id="__gt__"><code>__gt__</code></h3>

<a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/path.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__gt__(
    other: "Path"
) -> bool
</code></pre>

Return self>value.


<h3 id="__le__"><code>__le__</code></h3>

<a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/path.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__le__(
    other: "Path"
) -> bool
</code></pre>

Return self<=value.


<h3 id="__len__"><code>__len__</code></h3>

<a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/path.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__len__() -> int
</code></pre>




<h3 id="__lt__"><code>__lt__</code></h3>

<a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/path.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__lt__(
    other: "Path"
) -> bool
</code></pre>

Return self<value.


<h3 id="__ne__"><code>__ne__</code></h3>

<a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/path.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__ne__(
    other: "Path"
) -> bool
</code></pre>

Return self!=value.




