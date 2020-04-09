/* Copyright 2020 Google LLC

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
#include <string>

#include <gtest/gtest.h>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace {

TEST(OpAndKernelRegistrationTest, Struct2TensorOpsAndKernelsAreRegistered) {
  static constexpr char const* kStruct2TensorOps[] = {
    "DecodeProtoMap",
    "EquiJoinIndices",
    "DecodeProtoSparseV2",
    "RunLengthBefore",
    "ParquetDataset",
  };

  const auto* global_op_registry = tensorflow::OpRegistry::Global();
  for (const char* op_name : kStruct2TensorOps) {
    const tensorflow::OpRegistrationData* ord;
    EXPECT_TRUE(global_op_registry->LookUp(op_name, &ord).ok()) << op_name;
    EXPECT_EQ(1, tensorflow::GetRegisteredKernelsForOp(op_name).kernel_size())
        << op_name;
  }
}

}  // namespace
