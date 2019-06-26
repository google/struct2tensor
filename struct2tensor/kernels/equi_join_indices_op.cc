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
// Given two ascending int64_t vectors a,b this
// returns two vectors c (index_a),d (index_b) of equal length, where:
// 1. for all i,j, if a[i]=b[j], then there exists a unique k where c[k]=i and
//    d[k]=j.
// 2. for all k, a[c[k]]=b[d[k]].
// 3. Finally, (c,d) are lexicographically ordered. For all k<m, c[k]<=c[m], and
//    if d[k]>d[m], then c[k]<c[m].
// Notice that when combined with tf.gather(...), this can be used to do "some"
// sparse_gather operations.

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace struct2tensor {

namespace {
using tensorflow::DEVICE_CPU;
using tensorflow::errors::InvalidArgument;
using tensorflow::OpKernel;
using tensorflow::OpKernelConstruction;
using tensorflow::OpKernelContext;
using tensorflow::Status;
using tensorflow::Tensor;
using tensorflow::TensorShape;
using tensorflow::TensorShapeUtils;

// Creates an output vector tensor of dtype int64_t from a vector<int64_t>.
::tensorflow::Status ToOutputVector(OpKernelContext* context, int index,
                                    const std::vector<tensorflow::int64>& vec) {
  TensorShape output_shape;
  tensorflow::int64 tensor_size = vec.size();
  TF_RETURN_IF_ERROR(
      TensorShapeUtils::MakeShape(&tensor_size, 1, &output_shape));

  Tensor* result = nullptr;
  TF_RETURN_IF_ERROR(context->allocate_output(index, output_shape, &result));
  if (tensor_size > 0) {
    auto result_flat = result->flat<tensorflow::int64>();

    for (tensorflow::int64 i = 0; i < tensor_size; ++i) {
      result_flat(i) = vec[i];
    }
  }
  return tensorflow::Status::OK();
}

}  // namespace

class EquiJoinIndicesOp : public OpKernel {
 public:
  explicit EquiJoinIndicesOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  // Returns true iff shape is a real vector or a row vector (N x 1 matrix).
  static bool IsEquivToVector(const TensorShape& shape) {
    return TensorShapeUtils::IsVector(shape) ||
           (TensorShapeUtils::IsMatrix(shape) && (shape.dim_size(1) == 1));
  }

  // Computes indices for an equi-join of its inputs, as described at the top.
  // Inputs/outputs are set up as follows:
  // input(0): a
  // input(1): b
  // output(0): index_a_vec
  // output(1): index_b_vec
  void Compute(OpKernelContext* context) override {
    const Tensor& a = context->input(0);
    OP_REQUIRES(context, IsEquivToVector(a.shape()),
                InvalidArgument("First argument not a vector"));
    const Tensor& b = context->input(1);
    OP_REQUIRES(context, IsEquivToVector(b.shape()),
                InvalidArgument("Second argument not a vector"));
    std::vector<tensorflow::int64> index_a_vec;
    std::vector<tensorflow::int64> index_b_vec;
    auto a_flat = a.flat<tensorflow::int64>();
    auto b_flat = b.flat<tensorflow::int64>();
    const tensorflow::int64 a_rows = a_flat.size();
    const tensorflow::int64 b_rows = b_flat.size();

    tensorflow::int64 index_a = 0;
    tensorflow::int64 index_b = 0;
    // Lots of checks that don't need to be done. Optimize this later.
    while (index_a < a_rows && index_b < b_rows) {
      if (a_flat(index_a) == b_flat(index_b)) {
        tensorflow::int64 run_index_b = index_b;
        while (run_index_b < b_rows && b_flat(run_index_b) == a_flat(index_a)) {
          index_a_vec.push_back(index_a);
          index_b_vec.push_back(run_index_b);
          ++run_index_b;
        }
        ++index_a;
      } else if (a_flat(index_a) < b_flat(index_b)) {
        ++index_a;
      } else if (a_flat(index_a) > b_flat(index_b)) {
        ++index_b;
      }
    }
    OP_REQUIRES_OK(context, ToOutputVector(context, 0, index_a_vec));
    OP_REQUIRES_OK(context, ToOutputVector(context, 1, index_b_vec));
  }
};

// Given two ascending int64_t vectors a,b returns two vectors c,d of equal
// length, where:
// 1. for all i,j, if a[i]=b[j], then there exists a unique k where c[k]=i and
//    d[k]=j.
// 2. for all k, a[c[k]]=b[d[k]].
// 3. Finally, (c,d) are lexicographically ordered. For all k<m, c[k]<=c[m], and
//    if d[k]>d[m], then c[k]<c[m].
// Notice that when combined with tf.gather(...), this can be used to do "some"
// sparse_gather operations.
REGISTER_KERNEL_BUILDER(Name("EquiJoinIndices").Device(DEVICE_CPU),
                        EquiJoinIndicesOp);

}  // namespace struct2tensor
