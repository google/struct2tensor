# `struct2tensor` release notes

# Current Version (not yet released; still in development)

## Major Features and Improvements

## Bug Fixes and Other Changes

*   Bumped the Ubuntu version on which `struct2tensor` is tested to 20.04
    (previously was 16.04).
*   Depends on `tensorflow>=2.15.0,<3`.
*   Bumped the minimum bazel version required to build `struct2tensor` to 6.1.0.
*   Depends on `protobuf>=4.25.2,<5` for Python 3.11 and on `protobuf>3.20.3,<5`
    for 3.9 and 3.10.

## Breaking Changes

## Deprecations

*   Deprecated python 3.8 support.

# Version 0.45.0

## Major Features and Improvements

*   N/A

## Bug Fixes and Other Changes

*   Depends on `pyarrow>=10,<11`.
*   Depends on `numpy>=1.22.0`.
*   Depends on `tensorflow>=2.13.0,<3`.

## Breaking Changes

*   N/A

## Deprecations

*   N/A

# Version 0.44.0

## Major Features and Improvements

*   N/A

## Bug Fixes and Other Changes

*   Introduced an argument to disable path step validation.
*   Depends on `tensorflow>=2.12.0,<2.13`.
*   Depends on `protobuf>=3.20.3,<5`.

## Breaking Changes

*   N/A

## Deprecations

*   Deprecated python 3.7 support.

# Version 0.43.0

## Major Features and Improvements

*   N/A

## Bug Fixes and Other Changes

*   Depends on `tensorflow>=2.11.0,<2.12`.

## Breaking Changes

*   N/A

## Deprecations

*   N/A

# Version 0.42.0

## Major Features and Improvements

*   N/A

## Bug Fixes and Other Changes

*   Updates bundled `arrow` version to 6.0.1.
*   Depends on `tensorflow>=2.10.0,<2.11`.

## Breaking Changes

*   N/A

## Deprecations

*   N/A

# Version 0.41.0

## Major Features and Improvements

*   N/A

## Bug Fixes and Other Changes

*   Depends on `pyarrow>=6,<7`.
*   Depends on `tensorflow-metadata>=1.10.0,<1.11.0`.

## Breaking Changes

*   N/A

## Deprecations

*   N/A

# Version 0.40.0

## Major Features and Improvements

*   Linux wheels now specifically conform to
    [manylinux2014](https://peps.python.org/pep-0599/), an upgrade from
    manylinux2010. This is aligned with TensorFlow 2.9 requirement.

## Bug Fixes and Other Changes

*   Depends on `tensorflow>=2.9.0,<2.10`.
*   Depends on `tensorflow-metadata>=1.9.0,<1.10.0`.

## Breaking Changes

*   N/A

## Deprecations

*   N/A

# Version 0.39.0

## Major Features and Improvements

*   From this version we will be releasing python 3.9 wheels.

## Bug Fixes and Other Changes

*   Depends on `tensorflow-metadata>=1.8.0,<1.9.0`.

## Breaking Changes

*   N/A

## Deprecations

*   N/A

# Version 0.38.0

## Major Features and Improvements

*   Added equi_join_any_indices_op.
*   Added broadcast for subtrees.

## Bug Fixes and Other Changes

*   Depends on `tensorflow>=2.8.0,<2.9`.
*   Depends on `tensorflow-metadata>=1.7.0,<1.8.0`.

## Breaking Changes

*   N/A

## Deprecations

*   N/A

# Version 0.37.0

## Major Features and Improvements

*   N/A

## Bug Fixes and Other Changes

*   Depends on `tensorflow-metadata>=1.6.0,<1.7.0`.

## Breaking Changes

*   N/A

## Deprecations

*   N/A

# Version 0.36.0

## Major Features and Improvements

*   N/A

## Bug Fixes and Other Changes

*   Depends on `tensorflow>=2.7.0,<2.8`.
*   Depends on `tensorflow-metadata>=1.5.0,<1.6.0`.

## Breaking Changes

*   N/A

## Deprecations

*   N/A

# Version 0.35.0

## Major Features and Improvements

*   N/A

## Bug Fixes and Other Changes

*   Fix bug in which expression.apply_schema mutated its input schema

## Breaking Changes

*   N/A

## Deprecations

*   Deprecated python 3.6 support.

# Version 0.34.0

## Major Features and Improvements

*   N/A

## Bug Fixes and Other Changes

*   Depends on `tensorflow>=2.6.0,<2.7`.
*   Depends on `pyarrow>=1,<6`.

## Breaking Changes

*   N/A

## Deprecations

*   N/A

## Version 0.33.0

## Major Features and Improvements

*   N/A

## Bug Fixes and Other Changes

*   Added doc with benchmark numbers. Also added the benchmark code and test
    data.
*   Depends on `tensorflow-metadata>=1.2.0,<1.3.0`.

## Breaking Changes

*   N/A

## Deprecations

*   N/A

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
*   Bumped the minimum bazel version required to build `struct2tensor` to 3.7.2.
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
    pip install --extra-index-url https://pypi-nightly.tensorflow.org/simple struct2tensor
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
