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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

REGISTER_OP("RunLengthBefore")
    .Input("ordered_indices: int64")
    .Output("run_length_before: int64")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* context) {
      context->set_output(0, context->input(0));
      return absl::OkStatus();
    })
    .Doc(R"doc(
The `run_length_before` op, given [a_0,...,a_n], returns [b_0,...,b_n] where:
  b_n := \sum_{i=0}^{n-1} I(a_i=a_n)
  This assumes that for all a_i, a_j, if i <= j, then a_i <= a_j.

This is useful for creating the last index column of a ragged array, or from
converting from global orderings to local orderings or dewey orderings.

For example:
  input:  [0, 0, 7, 7, 8, 9, 9]
  output: [0, 1, 0, 1, 0, 0, 1]

ordered_indices: a int64 vector where for all i, a[i] <= a[i+1]
run_length_before: for all n:
   run_length_before[n] := \sum_{i=0}^{n-1} I(a_i=a_n)

)doc");
