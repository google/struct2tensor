description: A collection of protobufs dynamically constructed by descriptor protos.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="expression_impl.proto.DescriptorPool" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="Add"/>
<meta itemprop="property" content="AddDescriptor"/>
<meta itemprop="property" content="AddEnumDescriptor"/>
<meta itemprop="property" content="AddExtensionDescriptor"/>
<meta itemprop="property" content="AddFileDescriptor"/>
<meta itemprop="property" content="AddSerializedFile"/>
<meta itemprop="property" content="AddServiceDescriptor"/>
<meta itemprop="property" content="FindAllExtensions"/>
<meta itemprop="property" content="FindEnumTypeByName"/>
<meta itemprop="property" content="FindExtensionByName"/>
<meta itemprop="property" content="FindExtensionByNumber"/>
<meta itemprop="property" content="FindFieldByName"/>
<meta itemprop="property" content="FindFileByName"/>
<meta itemprop="property" content="FindFileContainingSymbol"/>
<meta itemprop="property" content="FindMessageTypeByName"/>
<meta itemprop="property" content="FindMethodByName"/>
<meta itemprop="property" content="FindOneofByName"/>
<meta itemprop="property" content="FindServiceByName"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__new__"/>
</div>

# expression_impl.proto.DescriptorPool

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



A collection of protobufs dynamically constructed by descriptor protos.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>expression_impl.proto.DescriptorPool(
    descriptor_db=None
)
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`descriptor_db`
</td>
<td>
A secondary source of file descriptors.
</td>
</tr>
</table>



## Methods

<h3 id="Add"><code>Add</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>Add(
    file_desc_proto
)
</code></pre>

Adds the FileDescriptorProto and its types to this pool.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>
<tr class="alt">
<td colspan="2">
file_desc_proto (FileDescriptorProto): The file descriptor to add.
</td>
</tr>

</table>



<h3 id="AddDescriptor"><code>AddDescriptor</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>AddDescriptor(
    *args, **kwargs
)
</code></pre>




<h3 id="AddEnumDescriptor"><code>AddEnumDescriptor</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>AddEnumDescriptor(
    *args, **kwargs
)
</code></pre>




<h3 id="AddExtensionDescriptor"><code>AddExtensionDescriptor</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>AddExtensionDescriptor(
    *args, **kwargs
)
</code></pre>




<h3 id="AddFileDescriptor"><code>AddFileDescriptor</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>AddFileDescriptor(
    *args, **kwargs
)
</code></pre>




<h3 id="AddSerializedFile"><code>AddSerializedFile</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>AddSerializedFile(
    serialized_file_desc_proto
)
</code></pre>

Adds the FileDescriptorProto and its types to this pool.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>
<tr class="alt">
<td colspan="2">
serialized_file_desc_proto (bytes): A bytes string, serialization of the
:class:`FileDescriptorProto` to add.
</td>
</tr>

</table>



<h3 id="AddServiceDescriptor"><code>AddServiceDescriptor</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>AddServiceDescriptor(
    *args, **kwargs
)
</code></pre>




<h3 id="FindAllExtensions"><code>FindAllExtensions</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>FindAllExtensions(
    message_descriptor
)
</code></pre>

Gets all the known extensions of a given message.

Extensions have to be registered to this pool by build related
:func:`Add` or :func:`AddExtensionDescriptor`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>
<tr class="alt">
<td colspan="2">
message_descriptor (Descriptor): Descriptor of the extended message.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
list[FieldDescriptor]: Field descriptors describing the extensions.
</td>
</tr>

</table>



<h3 id="FindEnumTypeByName"><code>FindEnumTypeByName</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>FindEnumTypeByName(
    full_name
)
</code></pre>

Loads the named enum descriptor from the pool.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>
<tr class="alt">
<td colspan="2">
full_name (str): The full name of the enum descriptor to load.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>

<tr>
<td>
`EnumDescriptor`
</td>
<td>
The enum descriptor for the named type.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`KeyError`
</td>
<td>
if the enum cannot be found in the pool.
</td>
</tr>
</table>



<h3 id="FindExtensionByName"><code>FindExtensionByName</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>FindExtensionByName(
    full_name
)
</code></pre>

Loads the named extension descriptor from the pool.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>
<tr class="alt">
<td colspan="2">
full_name (str): The full name of the extension descriptor to load.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>

<tr>
<td>
`FieldDescriptor`
</td>
<td>
The field descriptor for the named extension.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`KeyError`
</td>
<td>
if the extension cannot be found in the pool.
</td>
</tr>
</table>



<h3 id="FindExtensionByNumber"><code>FindExtensionByNumber</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>FindExtensionByNumber(
    message_descriptor, number
)
</code></pre>

Gets the extension of the specified message with the specified number.

Extensions have to be registered to this pool by calling :func:`Add` or
:func:`AddExtensionDescriptor`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>
<tr class="alt">
<td colspan="2">
message_descriptor (Descriptor): descriptor of the extended message.
number (int): Number of the extension field.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>

<tr>
<td>
`FieldDescriptor`
</td>
<td>
The descriptor for the extension.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`KeyError`
</td>
<td>
when no extension with the given number is known for the
specified message.
</td>
</tr>
</table>



<h3 id="FindFieldByName"><code>FindFieldByName</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>FindFieldByName(
    full_name
)
</code></pre>

Loads the named field descriptor from the pool.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>
<tr class="alt">
<td colspan="2">
full_name (str): The full name of the field descriptor to load.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>

<tr>
<td>
`FieldDescriptor`
</td>
<td>
The field descriptor for the named field.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`KeyError`
</td>
<td>
if the field cannot be found in the pool.
</td>
</tr>
</table>



<h3 id="FindFileByName"><code>FindFileByName</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>FindFileByName(
    file_name
)
</code></pre>

Gets a FileDescriptor by file name.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>
<tr class="alt">
<td colspan="2">
file_name (str): The path to the file to get a descriptor for.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>

<tr>
<td>
`FileDescriptor`
</td>
<td>
The descriptor for the named file.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`KeyError`
</td>
<td>
if the file cannot be found in the pool.
</td>
</tr>
</table>



<h3 id="FindFileContainingSymbol"><code>FindFileContainingSymbol</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>FindFileContainingSymbol(
    symbol
)
</code></pre>

Gets the FileDescriptor for the file containing the specified symbol.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>
<tr class="alt">
<td colspan="2">
symbol (str): The name of the symbol to search for.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>

<tr>
<td>
`FileDescriptor`
</td>
<td>
Descriptor for the file that contains the specified
symbol.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`KeyError`
</td>
<td>
if the file cannot be found in the pool.
</td>
</tr>
</table>



<h3 id="FindMessageTypeByName"><code>FindMessageTypeByName</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>FindMessageTypeByName(
    full_name
)
</code></pre>

Loads the named descriptor from the pool.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>
<tr class="alt">
<td colspan="2">
full_name (str): The full name of the descriptor to load.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>

<tr>
<td>
`Descriptor`
</td>
<td>
The descriptor for the named type.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`KeyError`
</td>
<td>
if the message cannot be found in the pool.
</td>
</tr>
</table>



<h3 id="FindMethodByName"><code>FindMethodByName</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>FindMethodByName(
    full_name
)
</code></pre>

Loads the named service method descriptor from the pool.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>
<tr class="alt">
<td colspan="2">
full_name (str): The full name of the method descriptor to load.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>

<tr>
<td>
`MethodDescriptor`
</td>
<td>
The method descriptor for the service method.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`KeyError`
</td>
<td>
if the method cannot be found in the pool.
</td>
</tr>
</table>



<h3 id="FindOneofByName"><code>FindOneofByName</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>FindOneofByName(
    full_name
)
</code></pre>

Loads the named oneof descriptor from the pool.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>
<tr class="alt">
<td colspan="2">
full_name (str): The full name of the oneof descriptor to load.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>

<tr>
<td>
`OneofDescriptor`
</td>
<td>
The oneof descriptor for the named oneof.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`KeyError`
</td>
<td>
if the oneof cannot be found in the pool.
</td>
</tr>
</table>



<h3 id="FindServiceByName"><code>FindServiceByName</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>FindServiceByName(
    full_name
)
</code></pre>

Loads the named service descriptor from the pool.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>
<tr class="alt">
<td colspan="2">
full_name (str): The full name of the service descriptor to load.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>

<tr>
<td>
`ServiceDescriptor`
</td>
<td>
The service descriptor for the named service.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`KeyError`
</td>
<td>
if the service cannot be found in the pool.
</td>
</tr>
</table>
