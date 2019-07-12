# Struct2Tensor

<!--*
freshness: { owner: 'martinz' reviewed: '2019-06-14' }
*-->

NOTE: The libraries presented here can be used to build a tensorflow server.
However, there are some outstanding linking issues regarding using the
python APIs.


## Introduction
struct2tensor is a library for parsing structured data inside of tensorflow.
In particular, it makes it easy to manipulate structured data, e.g., slicing,
flattening, et cetera.

There are two main use cases of this package. One is to create a PIP package,
useful for creating models.
The other is to create static libraries for tensorflow-serving.
As these processes are independent, one can follow either set of directions
below.

## Creating a PIP package.

The struct2tensor PIP package is useful for creating models.

### Preliminaries

Before starting, you should install pip and virtualenv. You should create
a virtual environment, and build the library there. Finally, you should install
bazel.

### Building the PIP Package

Before building the pip package, you must configure bazel.
configure will make sure that the latest version of tensorflow is there,
and creates a .bazelrc file that links to that package.

```bash
./configure.sh
```

The bazel command will then download all the appropriate packages (such
as protobuf), build the .so files, and place all the appropriate files in
``./bazel-bin/build_pip_pkg.runfiles``.

```bash
bazel build -c opt build_pip_pkg
./bazel-bin/build_pip_pkg artifacts
```


### Installing the PIP Package

```bash
pip install artifacts/*.whl
```

## Creating a static library.

In order to construct a static library for tensorflow-serving, we run:

```bash
bazel build -c opt struct2tensor:prensor_kernels_and_ops
```
