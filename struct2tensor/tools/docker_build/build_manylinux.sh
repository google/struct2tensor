#!/bin/bash
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This script is expected to run in the docker container defined in
# Dockerfile.manylinux2010
# Assumptions:
# - CentOS environment.
# - devtoolset-8 is installed.
# - $PWD is struct2tensor's project root.
# - Python of different versions are installed at /opt/python/.
# - patchelf, zip, bazel, rsync are installed and are in $PATH.

WORKING_DIR=$PWD


function setup_environment() {
  # Since someone may run this twice from the same directory,
  # it is important to delete the dist directory.
  rm -rf dist || exit 1;
  # TODO(martinz): move this directly to the docker image.
  # RUN yum -y install rsync
  # However, if we move this this, we get
  # ./bazel-bin/build_pip_pkg: line 55: rsync: command not found
  sudo yum -y install rsync || exit 1;
  if [[ -z "${PYTHON_VERSION}" ]]; then
    echo "Must set PYTHON_VERSION env to 35|36|37|27"; exit 1;
  fi
  # Bazel will use PYTHON_BIN_PATH to determine the right python library.
  if [[ "${PYTHON_VERSION}" == 27 ]]; then
    PYTHON_DIR=/opt/python/cp27-cp27mu
  elif [[ "${PYTHON_VERSION}" == 35 ]]; then
    PYTHON_DIR=/opt/python/cp35-cp35m
  elif [[ "${PYTHON_VERSION}" == 36 ]]; then
    PYTHON_DIR=/opt/python/cp36-cp36m
  elif [[ "${PYTHON_VERSION}" == 37 ]]; then
    PYTHON_DIR=/opt/python/cp37-cp37m
  else
    echo "Must set PYTHON_VERSION env to 35|36|37|27"; exit 1;
  fi
  if [[ -z "${TENSORFLOW_VERSION}" ]]; then
    echo "Must set TENSORFLOW_VERSION env to 1 or 2"; exit 1;
  fi
  # pip will use SPECIFIC_TENSORFLOW_VERSION to install.
  if [[ "${TENSORFLOW_VERSION}" == 1 ]]; then
    export SPECIFIC_TENSORFLOW_VERSION="1.15.0"
  elif [[ "${TENSORFLOW_VERSION}" == 2 ]]; then
    export SPECIFIC_TENSORFLOW_VERSION="2.0.0"
  else
    echo "Must set TENSORFLOW_VERSION env to 1|2"; exit 1;
  fi
  export PIP_BIN="${PYTHON_DIR}"/bin/pip || exit 1;
  export PYTHON_BIN_PATH="${PYTHON_DIR}"/bin/python || exit 1;
  echo "PYTHON_BIN_PATH=${PYTHON_BIN_PATH}" || exit 1;
  export WHEEL_BIN="${PYTHON_DIR}"/bin/wheel || exit 1;
  ${PIP_BIN} install --upgrade pip || exit 1;
  ${PIP_BIN} install wheel --upgrade || exit 1;
  # Auditwheel does not have a python2 version and auditwheel is just a binary.
  pip3 install auditwheel || exit 1;
}

function bazel_build() {
  virtualenv --python=${PYTHON_BIN_PATH} venv || exit 1;
  source venv/bin/activate || exit 1;
  pip install --upgrade pip || exit 1;
  pip install tensorflow==${SPECIFIC_TENSORFLOW_VERSION}  || exit 1;
  ./configure.sh || exit 1;
  bazel build -c opt build_pip_pkg || exit 1;
  ./bazel-bin/build_pip_pkg dist || exit 1;
  deactivate || exit 1;
}

libraries=(
"_decode_proto_map_op.so"
"_decode_proto_sparse_op.so"
"_run_length_before_op.so"
"_equi_join_indices_op.so"
)

LIBRARY_DIR="struct2tensor/ops/"

# This should have been simply an invocation of "auditwheel repair" but because
# of https://github.com/pypa/auditwheel/issues/76, tensorflow's shared libraries
# that struct2tensor depends on are treated incorrectly by auditwheel.
# We have to do this trick to make auditwheel happily stamp on our wheel.
# Note that even though auditwheel would reject the wheel produced in the end,
# it's still manylinux2010 compliant according to the standard, because it only
# depends on the specified shared libraries, assuming pyarrow is installed.
function stamp_wheel() {
  WHEEL_PATH="$(ls "$PWD"/dist/*.whl)"
  WHEEL_DIR=$(dirname "${WHEEL_PATH}")

  # Create a backup of the files in the wheel.
  # We will zip the original .so files back into the code when we are done.
  BACKUP_DIR="$(mktemp -d)"
  pushd "${BACKUP_DIR}"
  unzip "${WHEEL_PATH}" || exit 1;
  popd

  TMP_DIR="$(mktemp -d)"
  pushd "${TMP_DIR}"
  unzip "${WHEEL_PATH}" || exit 1;
  for SO_FILE in "${libraries[@]}"; do
    SO_FILE_PATH="${LIBRARY_DIR}${SO_FILE}"
    echo "file path: ${SO_FILE_PATH}"
    # LIBTENSORFLOW="libtensorflow_framework.so.1"
    LIBTENSORFLOW=$(patchelf --print-needed ${SO_FILE_PATH} | fgrep "libtensorflow_framework")
    echo "LIBTENSORFLOW: ${LIBTENSORFLOW}"
    patchelf --remove-needed "${LIBTENSORFLOW}" "${SO_FILE_PATH}"  || exit 1;
    zip "${WHEEL_PATH}" "${SO_FILE_PATH}" || exit 1;
  done
  popd

  auditwheel repair --plat manylinux2010_x86_64 -w "${WHEEL_DIR}" "${WHEEL_PATH}"  || exit 1;
  rm "${WHEEL_PATH}" || exit 1;
  MANY_LINUX_WHEEL_PATH=$(ls "${WHEEL_DIR}"/*manylinux*.whl)
  # Unzip the manylinux2010 wheel and pack it again with the original .so file.
  # We need to use "wheel pack" in order to compute the file hashes again.
  TMP_DIR="$(mktemp -d)"
  unzip "${MANY_LINUX_WHEEL_PATH}" -d "${TMP_DIR}" || exit 1;
  for SO_FILE in "${libraries[@]}"; do
    SO_FILE_PATH="${LIBRARY_DIR}${SO_FILE}"
    cp "${BACKUP_DIR}/${SO_FILE_PATH}" "${TMP_DIR}/${SO_FILE_PATH}"  || exit 1;
  done

  rm "${MANY_LINUX_WHEEL_PATH}" || exit 1;
  ${WHEEL_BIN} version || exit 1;
  ${WHEEL_BIN} pack "${TMP_DIR}" --dest-dir "${WHEEL_DIR}" || exit 1;
}

set -x
setup_environment && \
bazel_build && \
stamp_wheel
set +x

