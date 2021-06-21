# `struct2tensor` release notes

# Current Version (not yet released; still in development)

## Major Features and Improvements

## Bug Fixes and Other Changes

## Breaking Changes

## Deprecations

## Version 0.32.0

## Major Features and Improvements

*   N/A

## Bug Fixes and Other Changes

*   Depends on `protobuf>=3.13,<4`.
*   Depends on `tensorflow-metadata>=1.1.0,<1.2.0`.

## Breaking Changes

*   N/A

## Deprecations

*   N/A

## Version 0.31.0

## Major Features and Improvements

*   N/A

## Bug Fixes and Other Changes

*   Introduced DecodeProtoSparseV4. It is same as V3 and will replace V3 soon.
*   DecodeProtoSparseV3 is now the default (instead of V2).
*   Bumped tf version for statically linked libraries to TF 2.5.0.
*   Depends on `tensorflow>=2.5.0,<2.6`.
*   Depends on `tensorflow-metadata>=1.0.0,<1.1.0`.

## Breaking Changes

*   N/A

## Deprecations

*   N/A

## Release 0.30.0

## Major Features and Improvements

*   N/A

## Bug Fixes and Other Changes

*   Deprecate `get_ragged_tensors()` and `get_sparse_tensors()` in prensor_util.
*   Expose `get_ragged_tensors()` and `get_sparse_tensors()` as `Prensor`
    methods.
*   Expose `get_positional_index` as a method of `NodeTensor`.
*   Depends on `tensorflow-metadata>=0.30,<0.31`

## Breaking Changes

*   N/A

## Deprecations

*   N/A

## Release 0.29.0

## Major Features and Improvements

*   N/A

## Bug Fixes and Other Changes

*   Allow path to concat with string.
*   Bumped the mininum bazel version required to build `struct2tensor` to 3.7.2.
*   Depends on `tensorflow-metadata>=0.29,<0.30`

## Breaking Changes

*   N/A

## Deprecations

*   N/A

## Release 0.28.0

## Major Features and Improvements

*   N/A

## Bug Fixes and Other Changes

*   Depends on `tensorflow-metadata>=0.28,<0.29`

## Breaking Changes

*   N/A

## Deprecations

*   N/A

## Release 0.27.0

## Major Features and Improvements

*   N/A

## Bug Fixes and Other Changes

*   Depends on `pyarrow>=1,<3`
*   Depends on `tensorflow>=2.4.0,<2.5`
*   Depends on `tensorflow-metadata>=0.27,<0.28`

## Breaking changes

*   N/A

## Deprecations

*   N/A

## Release 0.26.0

## Major Features and Improvements

*   Created a docker image that contains a TF model server with struct2tensor
    ops linked. This docker image is available at
    `gcr.io/tfx-oss-public/s2t_tf_serving` .
*   Add support for string_views for intermediate serialized protos. To use, set
    the option "use_string_view" in CalculateOptions to true. string_views are
    potentially more memory bandwidth efficient depending on the depth and
    complexity of the input proto.

## Bug Fixes and Other Changes

*   Depends on `tensorflow-metadata>=0.26,<0.27`.

## Breaking changes

*   N/A

## Deprecations

*   N/A

## Release 0.25.0

## Major Features and Improvements

*   From this release Struct2Tensor will also be hosting nightly packages on
    https://pypi-nightly.tensorflow.org. To install the nightly package use the
    following command:

    ```
    pip install -i https://pypi-nightly.tensorflow.org/simple struct2tensor
    ```

    Note: These nightly packages are unstable and breakages are likely to
    happen. The fix could often take a week or more depending on the complexity
    involved for the wheels to be available on the PyPI cloud service. You can
    always use the stable version of struct2tensor available on PyPI by running
    the command `pip install struct2tensor` .

## Bug Fixes and Other Changes

*   Update __init__.py to import the API, instead of just the modules.
*   Provide an __init__.py for struct2tensor.expression_impl directory. This is
    meant for power users.
*   Update python notebook to use import style.
*   Fix bug in prensor_to_structured_tensor.
*   Depends on `tensorflow-metadata>=0.25,<0.26`.
*   Depends on `pyarrow>=0.17,<1`.

## Breaking changes

*   N/A

## Deprecations

*   N/A

## Release 0.24.0

## Major Features and Improvements

*   Add support for converting prensor to `StructuredTensor`.

## Bug Fixes and Other Changes

*   Depends on `tensorflow>=2.3.0,<2.4`
*   Depends on `tensorflow-metadata>=0.24,<0.25`

## Breaking changes

*   N/A

## Deprecations

*   Deprecated py3.5 support

## Release 0.23.0

### Major Features and Improvements

*   Add promote for substructures.
*   Add support for converting `StructuredTensor` to prensor.

### Bug Fixes and Other Changes

*   Depends on `tensorflow>=2.3.0,<2.4`
*   Depends on `tensorflow-metadata>=0.23,<0.24`

### Breaking Changes

*   Drop snappy support for parquet dataset.

### Deprecations

*   Deprecating Py2 support.

## Release 0.22.0

### Major Features and Improvements

### Bug Fixes and Other Changes

*   Depends on `tensorflow>=2.2.0,<2.3

### Breaking Changes

### Deprecations

## Release 0.21.1

### Major Features and Improvements

*   Bumped Tensorflow version for statically linked libraries from 1.5 to 2.1.

### Bug Fixes and Other Changes

*   Added tests for statically linked libraries.
*   Statically linked libraries build now.

### Breaking Changes

### Deprecations

## Release 0.21.0

### Major Features and Improvements

*   Parquet dataset that can apply expressions to a parquet schema, allowing for
    reading data from IO to tensors directly.

### Bug Fixes and Other Changes

*   Now requires tensorflow>=2.1.0,<2.2.

### Breaking Changes

### Deprecations

## Release 0.0.1dev6

*   Initial release of struct2tensor.
