# Version 0.52.0

## Major Features and Improvements

*   N/A

## Bug Fixes and Other Changes

*   Depends on `tensorflow 2.21.0`.
*   Depends on `tensorflow-metadata>=1.21.0,<1.22.0`.
*   Depends on `protobuf==6.31.1`.
*   Depends on `pyarrow==23.0.1`.
*   Enforced C++17 in `.bazelrc`.
*   Disabled Bzlmod in `.bazelrc` to resolve protobuf conflicts.
*   Added dummy repositories in `WORKSPACE` to bypass circular dependencies with TensorFlow.
*   Fixed missing `#include <cstdint>` in various files to support compilation with `gcc 15`.
*   Added support for Python 3.12 and 3.13.
*   Upgraded Bazel global pin to `7.7.0`.
*   Upgraded release build container configurations from legacy `manylinux2014` to modern `manylinux_2_28`.
*   Stamps dynamic release wheels to match `manylinux_2_35` dynamic system dependencies.

## Breaking Changes

*   N/A

## Deprecations

*   Dropped support for Python 3.9.
