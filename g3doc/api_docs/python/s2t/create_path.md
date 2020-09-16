description: Create a path from an object.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="s2t.create_path" />
<meta itemprop="path" content="Stable" />
</div>

# s2t.create_path

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/path.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Create a path from an object.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>s2t.create_path(
    path_source: <a href="../s2t/Path.md"><code>s2t.Path</code></a>
) -> <a href="../s2t/Path.md"><code>s2t.Path</code></a>
</code></pre>



<!-- Placeholder for "Used in" -->


#### The BNF for a path is:


letter := [A-Za-z]
digit := [0-9]
<simple_step_char> := "_"|"-"| | letter | digit
<simple_step> := <simple_step_char>+
<extension> := "(" (<simple_step> ".")* <simple_step> ")"
<step> := <simple_step> | <extension>
<path> := ((<step> ".") * <step>)?



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`path_source`
</td>
<td>
a string or a Path object.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A Path.
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
if this is not a valid path.
</td>
</tr>
</table>

