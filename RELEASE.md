# `struct2tensor` release notes

# Current version (not yet released; still in development)

## Major Features and Improvements

## Bug Fixes and Other Changes

## Breaking changes

## Deprecations

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
