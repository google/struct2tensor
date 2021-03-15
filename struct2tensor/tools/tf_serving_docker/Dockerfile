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

FROM ubuntu:18.04 as base_build
ARG TF_SERVING_VERSION_GIT_BRANCH=master
ARG TF_SERVING_VERSION_GIT_COMMIT=head
ARG S2T_VERSION_GIT_BRANCH=master
ARG S2T_VERSION_GIT_COMMIT=head

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
        python-dev \
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

# Make python3.6 default.
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 0 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.6 0

RUN curl -fSsL -O https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py

RUN pip3 --no-cache-dir install --upgrade numpy

# Set up Bazel
ENV BAZEL_VERSION 3.7.2
WORKDIR /bazel
RUN curl -fSsL -O https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    chmod +x bazel-*.sh && \
    ./bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    cd / && \
    rm -f /bazel/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh

# Download buildozer
ENV BUILDOZER_VERSION 4.0.1
RUN curl -fSsL -O https://github.com/bazelbuild/buildtools/releases/download/$BUILDOZER_VERSION/buildozer-linux-amd64 && \
    chmod +x buildozer && \
    mv buildozer /bin

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
     echo "tensorflow-serving: $(git rev-parse HEAD)" && \
     echo "local_repository(name=\"com_google_struct2tensor\", path=\"/struct2tensor\")" \
         >> WORKSPACE && \
     buildozer "add deps @com_google_struct2tensor//struct2tensor:struct2tensor_kernels_and_ops" \
         //tensorflow_serving/model_servers:tensorflow_model_server


FROM base_build as binary_build
ARG TF_SERVING_BUILD_OPTIONS="--config=release"
RUN echo "Building with build options: ${TF_SERVING_BUILD_OPTIONS}"
ARG TF_SERVING_BAZEL_OPTIONS=""
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
FROM ubuntu:18.04

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
ENV MODEL_NAME=model

# Create a script that runs the model server so we can use environment variables
# while also passing in arguments from the docker command line
RUN echo '#!/bin/bash \n\n\
tensorflow_model_server --port=8500 --rest_api_port=8501 \
--model_name=${MODEL_NAME} --model_base_path=${MODEL_BASE_PATH}/${MODEL_NAME} \
"$@"' > /usr/bin/tf_serving_entrypoint.sh \
&& chmod +x /usr/bin/tf_serving_entrypoint.sh

ENTRYPOINT ["/usr/bin/tf_serving_entrypoint.sh"]
