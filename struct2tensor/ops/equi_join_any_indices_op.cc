/* Copyright 2021 Google LLC

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
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

using tensorflow::shape_inference::InferenceContext;

REGISTER_OP("EquiJoinAnyIndices")
    .Input("a: int64")
    .Input("b: int64")
    .Output("index_a: int64")
    .Output("index_b: int64")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->Vector(InferenceContext::kUnknownDim));
      c->set_output(1, c->Vector(InferenceContext::kUnknownDim));
      return tensorflow::OkStatus();
    })
    .Doc(R"doc(
This op is similiar to EquiJoinIndices. However this op does not assume that
`a` and `b` are monotonically increasing. Prefer to use EquiJoinIndices if
possible.

)doc");
