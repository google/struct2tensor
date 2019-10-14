# Struct2Tensor

<!--*
freshness: { owner: 'martinz' reviewed: '2019-06-14' }
*-->

NOTE: The libraries presented here can be used to build a tensorflow server.
However, there are some outstanding linking issues regarding using the
python APIs. I am working on fixing these now.


## Introduction
struct2tensor is a library for parsing structured data inside of tensorflow.
In particular, it makes it easy to manipulate structured data, e.g., slicing,
flattening, et cetera.

There are two main use cases of this package. One is to create a PIP package,
useful for creating models.
The other is to create static libraries for tensorflow-serving.
As these processes are independent, one can follow either set of directions
below.

## Use a pre-built Linux PIP package.

Coming soon.


## Creating a PIP package.

The struct2tensor PIP package is useful for creating models.
It works with either tensorflow 2.0 or tensorflow 1.15.0rc3.

In order to unify the process, we recommend compiling struct2tensor inside
a docker container.



### Downloading the Code

In order to unify the process, we recommend compiling struct2tensor inside
a docker container.

Go to your home directory.

Download the source code.


git clone https://github.com/google/struct2tensor.git
cd ~/struct2tensor

### Start a Docker Container


Pull the docker image.

sudo docker pull tensorflow/tensorflow:custom-op-ubuntu16

Start the docker image, mounting the source code.

sudo docker run -it -v $(pwd):/struct2tensor tensorflow/tensorflow:custom-op-ubuntu16 /bin/bash

Now you are on the docker image. Start a virtual environment.

### Start a Virtual Environment

virtualenv venv
source venv/bin/activate


### Upgrade PIP


pip install --upgrade pip


### Install Tensorflow PIP Package.

Note: you have options here:

* Option 1 (TF 2.0): pip install tensorflow
* Option 2 (TF 1.15.0.rc3): pip install tensorflow==1.15.0rc3


### Go to the source code.


cd struct2tensor

### Configure the package.

./configure.sh

### Prepare to build the pip package.

bazel build -c opt build_pip_pkg

### Build the pip package.

./bazel-bin/build_pip_pkg artifacts

### Install the pip package.

pip install artifacts/*.whl 

### Test the pip package.

python struct2tensor/expression_impl/proto_test.py






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
