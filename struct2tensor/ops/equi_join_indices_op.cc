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
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

using tensorflow::shape_inference::InferenceContext;

REGISTER_OP("EquiJoinIndices")
    .Input("a: int64")
    .Input("b: int64")
    .Output("index_a: int64")
    .Output("index_b: int64")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->Vector(InferenceContext::kUnknownDim));
      c->set_output(1, c->Vector(InferenceContext::kUnknownDim));
      return tensorflow::Status::OK();
    })
    .Doc(R"doc(
An op on two 1-D int64_t tensors a,b that
returns two 1-D int64_t tensors [index_a, index_b] where:
1. For every k, a[index_a[k]] = b[index_b[k]]
2. For every i,j, iff a[i]==b[j], then there exists a k where
     index_a[k]=i and index_b[k]=j.
3. For any k, k' where k < k',
     index_a[k] <= index_a[k'], and if index_a[k] == index_a[k'], then
       index_b[k] <= index_b[k'].

Imagine if you had two tables, A with fields "a_key" and "a_value", and
B with fields "b_key" and "b_value", where a_key is monotonically increasing
int64_t, and b_key is monotonically increasing int64_t.

C = SELECT * FROM A, B WHERE A.a_key = B.b_key;

Imagine that A.a_key, B.b_key, A.a_value, and B.b_value are all 1-D tensors.

Then we can create the result C:
a_index, b_index = equi_join_indices(A.a_key,B.b_key)
C.a_key = tf.gather(A.a_key, a_index)
C.a_value = tf.gather(A.a_value, a_index)
C.b_key = tf.gather(B.b_key, b_index)
C.b_value = tf.gather(B.b_value, b_index)


a: a 1-D tensor where for all i, a[i] <= a[i+1]
b: a 1-D tensor where for all i, b[i] <= b[i+1]
index_a: a 1-D tensor of indices of a
index_b: a 1-D tensor of indices of b

)doc");
