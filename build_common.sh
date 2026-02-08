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
# Usage: build_common.sh [--python_bin_path=PYTHON_BIN_PATH] [--pip_bin_path=PIP_BIN_PATH] [--tf_version=TF_VERSION] [--skip_static_link_test]

function install_tensorflow() {
  # Install tensorflow from pip.
  #
  # Usage: install_tensorflow NIGHTLY_TF|NIGHTLY_TF_2|RELEASED_TF|PRERELEASED_TF|RELEASED_TF_2|PRERELEASED_TF_2 /PATH/TO/PIP /PATH/TO/PYTHON

  #TODO(b/329181965): As TFX lags for TensorFlow version, let's pass
  #the ceiling TF Version to the installation script.
  TF_MAX_VERSION = $4

  if [[ ("$1" == NIGHTLY_TF) || ("$1" == NIGHTLY_TF_2) ]]; then
    TF_PIP_PACKAGE="tf-nightly"
  elif [[ "$1" == RELEASED_TF ]]; then
    TF_PIP_PACKAGE="tensorflow<2"
  elif [[ "$1" == PRERELEASED_TF ]]; then
    TF_PIP_PACKAGE="tensorflow<2"
  elif [[ "$1" == RELEASED_TF_2 ]]; then
    if [[ -z "$TF_MAX_VERSION" ]]; then
      TF_PIP_PACKAGE="tensorflow>=2,<3"
    else
      TF_PIP_PACKAGE="tensorflow>=2,<${TF_MAX_VERSION}"
    fi
  elif [[ "$1" == PRERELEASED_TF_2 ]]; then
    if [[ -z "$TF_MAX_VERSION" ]]; then
      TF_PIP_PACKAGE="tensorflow>=2,<3"
    else
      TF_PIP_PACKAGE="tensorflow>=2,<${TF_MAX_VERSION}"
    fi
  else
    echo "Invalid tensorflow version string must be one of NIGHTLY_TF, NIGHTLY_TF_2, RELEASED_TF, PRERELEASED_TF, RELEASED_TF_2, PRERELEASED_TF_2."
    exit 1
  fi

  PIP_COMMAND=$2
  PYTHON_BIN_PATH=$3

  if [[ ("$1" == PRERELEASED_TF) || ("$1" == PRERELEASED_TF_2) ]]; then
    "${PIP_COMMAND}" install --pre "${TF_PIP_PACKAGE}" \
        || { echo "failed to install prereleased tensorflow version: ${TF_PIP_PACKAGE}"; exit 1; }
  else
    "${PIP_COMMAND}" install "${TF_PIP_PACKAGE}" \
        || { echo "failed to install tensorflow version: ${TF_PIP_PACKAGE}"; exit 1; }
  fi

  "${PYTHON_BIN_PATH}" -c 'import tensorflow as tf; print(tf.version.VERSION)'
}

set -x

for i in "$@"; do
  case "$i" in
    --python_bin_path=*)
      shift # past argument=value
      PYTHON_BIN_PATH=${i#*=}
      ;;
    --pip_bin_path=*)
      shift # past argument=value
      PIP_BIN_PATH=${i#*=}
      ;;
    --tf_version=*)
      shift # past argument=value
      TF_VERSION=${i#*=}
      ;;
    --skip_static_link_test)
      shift # past argument=value
      SKIP_STATIC_LINK_TEST=1
      ;;
    --tf_max_version=*)
      shift # past argument=value
      TF_MAX_VERSION=${i#*=}
      ;;
    *)
      printf "Unrecognized argument $1"
      exit 1
      ;;
  esac
done

set -x

install_tensorflow ${TF_VERSION} ${PIP_BIN_PATH} ${PYTHON_BIN_PATH} ${TF_MAX_VERSION}
./configure.sh --python_bin_path "${PYTHON_BIN_PATH}"

if [[ ("${TF_VERSION}" == "NIGHTLY_TF") || ("${TF_VERSION}" == "NIGHTLY_TF_2") ]]; then
  # Get the github commit sha for tf-nightly
  GIT_VERSION=$(${PYTHON_BIN_PATH} -c "import tensorflow as tf; print(tf.__git_version__)") \
    || { echo "failed to get tf-nightly git version"; exit 1; }

  TF_NIGHTLY_COMMIT=$(curl -X GET "https://api.github.com/repos/tensorflow/tensorflow/commits?sha="${GIT_VERSION}"" | "${PYTHON_BIN_PATH}" -c "import sys, json; print(json.load(sys.stdin)[0]['sha'])") \
    || { echo "failed to get git commit sha"; exit 1; }

  # Replaces _TENSORFLOW_GIT_COMMIT with TF_NIGHTLY_COMMIT
  sed -i'' -e 's/^_TENSORFLOW_GIT_COMMIT = ".*/_TENSORFLOW_GIT_COMMIT = '\""${TF_NIGHTLY_COMMIT}"\"'/g' WORKSPACE \
    || { echo "failed to replace tf commit in tf_version.bzl"; exit 1; }

  # Replaces _TENSORFLOW_SHA256 with empty string
  sed -i'' -e 's/^_TENSORFLOW_ARCHIVE_SHA256 = ".*/_TENSORFLOW_ARCHIVE_SHA256 = '\"\"'/g' WORKSPACE \
    || { echo "failed to replace tf sha256 in tf_version.bzl"; exit 1; }

fi

# :build_pip_package builds and links struct2tensor ops against a TF
# installation and packages the result dynamic libraries.
bazel run -c opt \
  :build_pip_package \
  -- \
  --python_bin_path "${PYTHON_BIN_PATH}" \
  || { echo "failed to build the pip package"; exit 1; }

if [[ -z "${SKIP_STATIC_LINK_TEST}" ]]; then
  # struct2tensor/ops:op_kernel_registration_test builds and links struct2tensor
  # ops against TF source (pulled from @org_tensorflow).
  bazel test -c opt struct2tensor/ops:op_kernel_registration_test \
    || { echo "failed to build struct2tensor ops statically"; exit 1; }
fi
