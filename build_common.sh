#!/bin/bash
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
# Usage: build_common.sh [--python_bin_path PYTHON_BIN_PATH] [--tf_version TF_VERSION]

function install_tensorflow() {
  # Install tensorflow from pip.
  #
  # Usage: install_tensorflow NIGHTLY_TF|NIGHTLY_TF_2|RELEASED_TF|PRERELEASED_TF|RELEASED_TF_2|PRERELEASED_TF_2 /PATH/TO/PIP /PATH/TO/PYTHON
  if [[ ("$1" == NIGHTLY_TF) || ("$1" == NIGHTLY_TF_2) ]]; then
    TF_PIP_PACKAGE="tf-nightly"
  elif [[ "$1" == RELEASED_TF ]]; then
    TF_PIP_PACKAGE="tensorflow<2"
  elif [[ "$1" == PRERELEASED_TF ]]; then
    TF_PIP_PACKAGE="tensorflow<2"
  elif [[ "$1" == RELEASED_TF_2 ]]; then
    TF_PIP_PACKAGE="tensorflow>=2"
  elif [[ "$1" == PRERELEASED_TF_2 ]]; then
    TF_PIP_PACKAGE="tensorflow>=2"
  else
    echo "Invalid tensorflow version string must be one of NIGHTLY_TF, NIGHTLY_TF_2, RELEASED_TF, PRERELEASED_TF, RELEASED_TF_2, PRERELEASED_TF_2."
    exit 1
  fi

  PIP_COMMAND=$2
  PYTHON_BIN_PATH=$3

  if [[ ("$1" == PRERELEASED_TF) || ("$1" == PRERELEASED_TF_2) ]]; then
    "${PIP_COMMAND}" install --pre "${TF_PIP_PACKAGE}" \
        || (echo "failed to install prereleased tensorflow version: ${TF_PIP_PACKAGE}" && exit 1)
  else
    "${PIP_COMMAND}" install "${TF_PIP_PACKAGE}" \
        || (echo "failed to install tensorflow version: ${TF_PIP_PACKAGE}" && exit 1)
  fi

  "${PYTHON_BIN_PATH}" -c 'import tensorflow as tf; print(tf.version.VERSION)'
}

set -x

for i in "$@"; do
  case "$i" in
    --python_bin_path)
      shift # past argument
      PYTHON_BIN_PATH=$1
      shift # past value
      ;;
    --tf_version)
      shift # past argument
      TF_VERSION=$1
      shift # past value
      ;;
    *)
      printf "Unrecognized argument $1"
      exit 1
      ;;
  esac
done

virtualenv --python=${PYTHON_BIN_PATH} venv \
    || (echo "failed to create a virtualenv" && exit 1)
source venv/bin/activate
VENV_PYTHON_BIN_PATH="$(which python)"
VENV_PIP_PATH="$(which pip)"
pip install --upgrade pip;
install_tensorflow ${TF_VERSION} ${VENV_PIP_PATH} ${VENV_PYTHON_BIN_PATH}
./configure.sh --python_bin_path "${VENV_PYTHON_BIN_PATH}"

deactivate

bazel run -c opt \
  :build_pip_package \
  -- \
  --python_bin_path "${PYTHON_BIN_PATH}"

set +x
