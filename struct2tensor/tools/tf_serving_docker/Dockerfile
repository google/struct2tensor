# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Defines an image that contains a tensorflow-serving model server binary
# that has struct2tensor ops linked in.

FROM ubuntu:22.04 AS base_build
ARG TF_SERVING_VERSION_GIT_BRANCH=r2.19
ARG TF_SERVING_VERSION_GIT_COMMIT=head
ARG S2T_VERSION_GIT_BRANCH=master
ARG S2T_VERSION_GIT_COMMIT=head
ARG DEBIAN_FRONTEND=noninteractive

LABEL s2t_github_commit=${S2T_VERSION_GIT_COMMIT}
LABEL s2t_github_branchtag=${S2T_VERSION_GIT_BRANCH}
LABEL tensorflow_serving_github_branchtag=${TF_SERVING_VERSION_GIT_BRANCH}
LABEL tensorflow_serving_github_commit=${TF_SERVING_VERSION_GIT_COMMIT}

RUN apt-get update && apt-get install -y --no-install-recommends \
        automake \
        build-essential \
        ca-certificates \
        curl \
        git \
        libcurl3-dev \
        libfreetype6-dev \
        libpng-dev \
        libtool \
        libzmq3-dev \
        mlocate \
        openjdk-8-jdk\
        openjdk-8-jre-headless \
        pkg-config \
        gnupg \
        software-properties-common \
        swig \
        unzip \
        wget \
        zip \
        zlib1g-dev \
        python3-distutils \
        python3-dev \
        python3-pip \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install python 3.13.
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y \
    python3.13 python3.13-dev python3-pip python3.13-venv && \
    rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.13 0

# Make python3.13 and the default python version
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.13 0

RUN curl -fSsL -O https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py

RUN pip3 --no-cache-dir install --upgrade numpy

# Set up Bazel
ENV BAZEL_VERSION 7.7.0
WORKDIR /bazel
RUN curl -fSsL -O https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    chmod +x bazel-*.sh && \
    ./bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    cd / && \
    rm -f /bazel/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh

# Download buildozer
ENV BUILDOZER_VERSION 4.0.1
RUN curl -fSsL -O https://github.com/bazelbuild/buildtools/releases/download/$BUILDOZER_VERSION/buildozer-linux-amd64 && \
    chmod +x buildozer-linux-amd64 && \
    mv buildozer-linux-amd64 /bin/buildozer

# Download s2t sources.
WORKDIR /struct2tensor
RUN git clone --branch=${S2T_VERSION_GIT_BRANCH} https://github.com/google/struct2tensor . && \
    git remote add upstream https://github.com/google/struct2tensor.git && \
    if [ "${S2T_VERSION_GIT_COMMIT}" != "head" ]; then \
     git checkout ${S2T_VERSION_GIT_COMMIT}; fi && \
    echo "struct2tensor: $(git rev-parse HEAD)"


# Download TF Serving sources.
WORKDIR /tensorflow-serving
RUN git clone --branch=${TF_SERVING_VERSION_GIT_BRANCH} https://github.com/tensorflow/serving . && \
    git remote add upstream https://github.com/tensorflow/serving.git && \
    if [ "${TF_SERVING_VERSION_GIT_COMMIT}" != "head" ]; then \
     git checkout ${TF_SERVING_VERSION_GIT_COMMIT}; fi && \
     echo "tensorflow-serving: $(git rev-parse HEAD)"

RUN touch BUILD && \
    echo 'def _local_tsl_impl(repository_ctx):' > local_tsl.bzl && \
    echo '    repository_ctx.file("WORKSPACE", "")' >> local_tsl.bzl && \
    echo '    repository_ctx.file("BUILD", "cc_library(name = \"bazel_issue_21519\", visibility = [\"//visibility:public\"])")' >> local_tsl.bzl && \
    echo '    repository_ctx.symlink("/root/.cache/bazel/_bazel_root/e53bbb0b0da4e26d24b415310219b953/external/org_tensorflow/third_party/xla/third_party/tsl/tsl", "tsl")' >> local_tsl.bzl && \
    echo '    repository_ctx.symlink("/root/.cache/bazel/_bazel_root/e53bbb0b0da4e26d24b415310219b953/external/org_tensorflow/third_party/xla/third_party/tsl/third_party", "third_party")' >> local_tsl.bzl && \
    echo 'local_tsl_repo = repository_rule(implementation = _local_tsl_impl)' >> local_tsl.bzl

RUN echo 'load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")' > overrides.txt && \
    echo 'http_archive(name = "rules_cc", sha256 = "b8b918a85f9144c01f6cfe0f45e4f2838c7413961a8ff23bc0c6cdf8bb07a3b6", strip_prefix = "rules_cc-0.1.5", urls = ["https://github.com/bazelbuild/rules_cc/releases/download/0.1.5/rules_cc-0.1.5.tar.gz"])' >> overrides.txt && \
    echo 'http_archive(name = "build_bazel_rules_apple", sha256 = "b892911288715b354e05b9a6fe9009635f7155991f24f27e779fe80d435c92bc", url = "https://github.com/bazelbuild/rules_apple/releases/download/3.13.0/rules_apple.3.13.0.tar.gz")' >> overrides.txt && \
    echo 'http_archive(name = "build_bazel_rules_swift", sha256 = "bb01097c7c7a1407f8ad49a1a0b1960655cf823c26ad2782d0b7d15b323838e2", url = "https://github.com/bazelbuild/rules_swift/releases/download/1.18.0/rules_swift.1.18.0.tar.gz")' >> overrides.txt && \
    echo 'http_archive(name = "build_bazel_apple_support", sha256 = "c4bb2b7367c484382300aee75be598b92f847896fb31bbd22f3a2346adf66a80", url = "https://github.com/bazelbuild/apple_support/releases/download/1.15.1/apple_support.1.15.1.tar.gz")' >> overrides.txt && \
    echo 'load("//:local_tsl.bzl", "local_tsl_repo")\nlocal_tsl_repo(name = "local_tsl")' >> overrides.txt && \
    sed -i '1r overrides.txt' WORKSPACE && \
    rm overrides.txt

RUN echo "local_repository(name=\"com_google_struct2tensor\", path=\"/struct2tensor\")" >> WORKSPACE && \
    buildozer "add deps @com_google_struct2tensor//struct2tensor:struct2tensor_kernels_and_ops" \
        //tensorflow_serving/model_servers:tensorflow_model_server


FROM base_build AS binary_build
ARG TF_SERVING_BUILD_OPTIONS="--config=release"
RUN echo "Building with build options: ${TF_SERVING_BUILD_OPTIONS}"
ARG TF_SERVING_BAZEL_OPTIONS="--noenable_bzlmod"
RUN echo "Building with Bazel options: ${TF_SERVING_BAZEL_OPTIONS}"
RUN bazel build --color=yes --curses=yes \
    ${TF_SERVING_BAZEL_OPTIONS} \
    --verbose_failures \
    --output_filter=DONT_MATCH_ANYTHING \
    ${TF_SERVING_BUILD_OPTIONS} \
    tensorflow_serving/model_servers:tensorflow_model_server && \
    cp bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server \
    /usr/local/bin/


# The following defines the serving container.
FROM ubuntu:22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install TF Serving pkg
COPY --from=binary_build /usr/local/bin/tensorflow_model_server /usr/bin/tensorflow_model_server

# Expose ports
# gRPC
EXPOSE 8500

# REST
EXPOSE 8501

# Set where models should be stored in the container
ENV MODEL_BASE_PATH=/models
RUN mkdir -p ${MODEL_BASE_PATH}

# The only required piece is the model name in order to differentiate endpoints
ENV MODEL_NAME=s2t

# Create a script that runs the model server so we can use environment variables
# while also passing in arguments from the docker command line
RUN echo '#!/bin/bash \n\n\
tensorflow_model_server --port=8500 --rest_api_port=8501 \
--model_name=${MODEL_NAME} --model_base_path=${MODEL_BASE_PATH}/${MODEL_NAME} \
"$@"' > /usr/bin/tf_serving_entrypoint.sh \
&& chmod +x /usr/bin/tf_serving_entrypoint.sh

ENTRYPOINT ["/usr/bin/tf_serving_entrypoint.sh"]
