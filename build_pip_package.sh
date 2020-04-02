#!/usr/bin/env bash
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# Convenience binary to build struct2tensor from source.
#
# Usage: build_pip_package.sh [--python_bin_path PYTHON_BIN_PATH]

if [[ -z "$1" ]]; then
  PYTHON_BIN_PATH=python
else
  if [[ "$1" == --python_bin_path ]]; then
    shift
    PYTHON_BIN_PATH=$1
  else
    printf "Unrecognized argument $1"
    exit 1
  fi
fi

set -u -x

function main() {
  # We will build the wheel in a temp directory.
  TMPDIR=$(mktemp -d)

  # If run by "bazel run", $(pwd) is the .runfiles dir that contains all the
  # data dependencies.
  RUNFILES_DIR=$(pwd)

  cp ${BUILD_WORKSPACE_DIRECTORY}/setup.py "${TMPDIR}"
  cp ${BUILD_WORKSPACE_DIRECTORY}/MANIFEST.in "${TMPDIR}"
  cp ${BUILD_WORKSPACE_DIRECTORY}/LICENSE "${TMPDIR}"
  cp ${BUILD_WORKSPACE_DIRECTORY}/README.md "${TMPDIR}"

  rsync -avm -L --exclude='*_test.py' ${BUILD_WORKSPACE_DIRECTORY}/struct2tensor "${TMPDIR}"

  cp -f ${RUNFILES_DIR}/struct2tensor/proto/*_pb2.py "${TMPDIR}"/struct2tensor/proto/
  cp -f ${RUNFILES_DIR}/struct2tensor/test/*_pb2.py "${TMPDIR}"/struct2tensor/test/
  cp -f ${RUNFILES_DIR}/struct2tensor/ops/*.so "${TMPDIR}"/struct2tensor/ops/

  chmod +w "${TMPDIR}"/struct2tensor/ops/*


  cd ${TMPDIR}
  ${PYTHON_BIN_PATH} setup.py bdist_wheel

  mkdir -p ${BUILD_WORKSPACE_DIRECTORY}/dist/
  cp dist/*.whl ${BUILD_WORKSPACE_DIRECTORY}/dist/
  rm -rf ${TMPDIR}
}

main "$@"
