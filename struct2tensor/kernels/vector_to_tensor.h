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
// `T`, with the exception of int64 types.
template <typename T>
inline void VectorToTensor(const std::vector<T>& v, tensorflow::Tensor* tensor,
                           bool produce_string_view) {
  std::copy_n(v.begin(), v.size(), tensor->flat<T>().data());
}

// Specialization for vector<string_view> - copies the strings into a string
// tensor.
template <>
inline void VectorToTensor<absl::string_view>(
    const std::vector<absl::string_view>& v, tensorflow::Tensor* tensor,
    bool produce_string_view) {
  tensorflow::tstring* output = tensor->flat<tensorflow::tstring>().data();
  for (auto sv : v) {
    if (produce_string_view) {
      (output++)->assign_as_view(sv);
    } else {
      (output++)->assign(sv.data(), sv.size());
    }
  }
}
}  // namespace struct2tensor

#endif  // THIRD_PARTY_PY_STRUCT2TENSOR_KERNELS_VECTOR_TO_TENSOR_H_
