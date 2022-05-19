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
# Dockerfile.manylinux2014
# Assumptions:
# - Based on TF SIG Build Dockerfile.
# - Will use manylinux2014 compatible devtoolset under /dt9
# - $PWD is struct2tensor's project root.
# - Python/pip/wheel/auditwheel are available in $PATH.
# - patchelf, zip, bazelisk, rsync are installed and are in $PATH.

WORKING_DIR=$PWD

function setup_environment() {
  set -x

  # Since someone may run this twice from the same directory,
  # it is important to delete the dist directory.
  rm -rf dist

  export PIP_BIN_PATH=$(which pip) || exit 1;
  export PYTHON_BIN_PATH=$(which python) || exit 1;
  ${PIP_BIN_PATH} install --upgrade pip || exit 1;
  ${PIP_BIN_PATH} install wheel --upgrade || exit 1;
}

function bazel_build() {
  set -x
  local build_args=()
  build_args+=("--python_bin_path=${PYTHON_BIN_PATH}")
  build_args+=("--pip_bin_path=${PIP_BIN_PATH}")
  build_args+=("--tf_version=${TF_VERSION}")
  if [[ -n ${SKIP_STATIC_LINK_TEST} ]]; then
    build_args+=("--skip_static_link_test")
  fi
  ./build_common.sh "${build_args[@]}"
}

libraries=(
"_decode_proto_map_op.so"
"_decode_proto_sparse_op.so"
"_run_length_before_op.so"
"_equi_join_any_indices_op.so"
"_equi_join_indices_op.so"
"_parquet_dataset_op.so"
)

LIBRARY_DIR="struct2tensor/ops/"

# This should have been simply an invocation of "auditwheel repair" but because
# of https://github.com/pypa/auditwheel/issues/76, tensorflow's shared libraries
# that struct2tensor depends on are treated incorrectly by auditwheel.
# We have to do this trick to make auditwheel happily stamp on our wheel.
# Note that even though auditwheel would reject the wheel produced in the end,
# it's still manylinux2014 compliant according to the standard, because it only
# depends on the specified shared libraries, assuming pyarrow is installed.
function stamp_wheel() {
  set -x
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
    LIBTENSORFLOW=$(patchelf --print-needed ${SO_FILE_PATH} | fgrep "libtensorflow_framework")
    patchelf --remove-needed "${LIBTENSORFLOW}" "${SO_FILE_PATH}"  || exit 1;
    zip "${WHEEL_PATH}" "${SO_FILE_PATH}" || exit 1;
  done
  popd
  auditwheel repair --plat manylinux2014_x86_64 -w "${WHEEL_DIR}" "${WHEEL_PATH}"  || exit 1;
  rm "${WHEEL_PATH}" || exit 1;
  MANY_LINUX_WHEEL_PATH=$(ls "${WHEEL_DIR}"/*manylinux*.whl)
  # Unzip the wheel and pack it again with the original .so file.
  # We need to use "wheel pack" in order to compute the file hashes again.
  TMP_DIR="$(mktemp -d)"
  unzip "${MANY_LINUX_WHEEL_PATH}" -d "${TMP_DIR}" || exit 1;
  for SO_FILE in "${libraries[@]}"; do
    SO_FILE_PATH="${LIBRARY_DIR}${SO_FILE}"
    cp "${BACKUP_DIR}/${SO_FILE_PATH}" "${TMP_DIR}/${SO_FILE_PATH}"  || exit 1;
  done

  rm "${MANY_LINUX_WHEEL_PATH}" || exit 1;
  wheel version || exit 1;
  wheel pack "${TMP_DIR}" --dest-dir "${WHEEL_DIR}" || exit 1;
}

set -x
setup_environment && \
bazel_build && \
stamp_wheel

