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
// An op on a 1-D tensor that, given [a_0,...,a_n], returns [b_0,...,b_n] where:
// b_n := \sum_{i=0}^{n-1} I(a_i=a_n)
// This assumes that for all a_i, a_j, if i <= j, then a_i <= a_j.

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"

namespace struct2tensor {

namespace {

using ::tensorflow::DEVICE_CPU;
using ::tensorflow::OpKernel;
using ::tensorflow::OpKernelConstruction;
using ::tensorflow::OpKernelContext;
using ::tensorflow::Status;
using ::tensorflow::Tensor;

class RunLengthBeforeOp : public OpKernel {
 public:
  explicit RunLengthBeforeOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<tensorflow::int64>();

    // Create an output tensor
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    const int64_t input_length = input.size();
    if (input_length > 0) {
      typename tensorflow::TTypes<tensorflow::int64>::Flat output_flat =
          output_tensor->flat<tensorflow::int64>();

      int64_t repeats_so_far = 0;
      output_flat(0) = 0;
      int64_t last_value = input(0);
      for (int64_t i = 1; i < input_length; i++) {
        const int64_t current_value = input(i);
        if (current_value == last_value) {
          ++repeats_so_far;
        } else {
          repeats_so_far = 0;
        }
        output_flat(i) = repeats_so_far;
        last_value = current_value;
      }
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("RunLengthBefore").Device(DEVICE_CPU),
                        RunLengthBeforeOp);

}  // namespace
}  // namespace struct2tensor
