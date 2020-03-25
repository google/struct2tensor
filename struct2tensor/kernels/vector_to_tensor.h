/* Copyright 2019 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef THIRD_PARTY_PY_STRUCT2TENSOR_KERNELS_VECTOR_TO_TENSOR_H_
#define THIRD_PARTY_PY_STRUCT2TENSOR_KERNELS_VECTOR_TO_TENSOR_H_

#include "absl/strings/string_view.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/tstring.h"

namespace struct2tensor {

// Populate `tensor` from a vector of `T`. This assumes `tensor`'s type is also
// `T`, with the exception of int64_t types.
template <typename T>
inline void VectorToTensor(const std::vector<T>& v,
                           tensorflow::Tensor* tensor) {
  std::copy_n(v.begin(), v.size(), tensor->flat<T>().data());
}

// Because tensorflow::int64 != int64_t, we are implicitly casting the
// elements here.
// TODO(martinz): if this is resolved, remove this implementation.
// See https://github.com/tensorflow/tensorflow/pull/21042
template <>
inline void VectorToTensor(const std::vector<int64_t>& v,
                           tensorflow::Tensor* tensor) {
  static_assert(sizeof(int64_t) == sizeof(tensorflow::int64),
                "int64s are not the same size.");
  std::copy_n(v.data(), v.size(), tensor->flat<tensorflow::int64>().data());
}

// Because tensorflow::uint64 != uint64_t, we are implicitly casting the
// elements here.
// TODO(martinz): if this is resolved, remove this implementation.
// See https://github.com/tensorflow/tensorflow/pull/21042
template <>
inline void VectorToTensor(const std::vector<uint64_t>& v,
                           tensorflow::Tensor* tensor) {
  static_assert(sizeof(uint64_t) == sizeof(tensorflow::uint64),
                "uint64s are not the same size.");
  std::copy_n(v.data(), v.size(), tensor->flat<tensorflow::uint64>().data());
}

// Specialization for vector<string_view> - copies the strings into a string
// tensor.
template <>
inline void VectorToTensor<absl::string_view>(
    const std::vector<absl::string_view>& v, tensorflow::Tensor* tensor) {
  tensorflow::tstring* output = tensor->flat<tensorflow::tstring>().data();
  for (auto sv : v) {
    (output++)->assign(sv.data(), sv.size());
  }
}

}  // namespace struct2tensor

#endif  // THIRD_PARTY_PY_STRUCT2TENSOR_KERNELS_VECTOR_TO_TENSOR_H_
