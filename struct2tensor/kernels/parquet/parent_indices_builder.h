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
#ifndef THIRD_PARTY_PY_STRUCT2TENSOR_GOOGLE_PARQUET_KERNEL_PARENT_INDICES_BUILDER_H_
#define THIRD_PARTY_PY_STRUCT2TENSOR_GOOGLE_PARQUET_KERNEL_PARENT_INDICES_BUILDER_H_

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"

namespace struct2tensor {
namespace parquet_dataset {

// A class that converts a column's repetition and definition levels to
// parent indices.
// This class is thread-compatible.
// Sample usage:
// std::unique_ptr<ParentIndicesBuilder> parent_indices_builder;
// std::vector<RepetitionType> repetition_pattern =
//     std::vector<RepetitionType>{kRepeated, kOptional, kRepeated};
// TF_RETURN_IF_ERROR(ParentIndicesBuilder::Create(repetition_pattern,
//                                        &parent_indices_builder));
// parent_indices_builder->AddParentIndices(d, r);
// const auto& parent_indices = parent_indices_builder->GetParentIndices();
// parent_indices_builder->ResetParentIndices();
class ParentIndicesBuilder {
 public:
  enum RepetitionType { kRequired = 0, kOptional = 1, kRepeated = 2 };

  // Factory method for creating ParentIndicesBuilder
  // This will return an error status if repetition_pattern is invalid:
  // i.e. it is empty, or does not start with a kRepeated.
  // Output parameter: ParentIndicesBuilder
  static tensorflow::Status Create(
      std::vector<RepetitionType> repetition_pattern,
      std::unique_ptr<ParentIndicesBuilder>* parent_indices_builder);

  ParentIndicesBuilder& operator=(const ParentIndicesBuilder&) = delete;

  ParentIndicesBuilder(const ParentIndicesBuilder&) = delete;

  // The `repetition_level` is the index at which repeated field in the fields
  // path the value has repeated.
  // The `definition_level` is the number of fields in the path that could be
  // undefined (i.e. are optional or repeated), which are actually present.
  // The `repetition_pattern` describes the repetition of each field in a path.
  // For example, given the following schema:
  // repeated Document
  //   repeated Name
  //     optional Url
  // Then the path 'Document.Name.Url' could have a repetition pattern of
  // {kRepeated, kRepeated, kOptional}.
  //
  // This function Inserts a column entry's parent indexes into the 2D
  // parent_indices vector. For example, consider the above schema. If we have a
  // parent indices vector, {{0}, {0}, {0}} that represents the following:
  // document
  //   name
  //     url: http://A
  // and we wanted to add another entry: (with repetition_level = 1,
  // definition_level = 2):
  // document
  //   name
  //     url: http://A
  //   name
  //     url: http://B
  // Then this function will modify the parent indices vector to become:
  // {{0}, {0, 0}, {0, 1}}
  void AddParentIndices(int16_t definition_level, int16_t repetition_level);

  const std::vector<std::vector<tensorflow::int64>>& GetParentIndices() const;

  const std::vector<RepetitionType>& GetRepetitionPattern() const;

  // Call this function once we are done building the parent indices.
  // This will reinititalize the parent_indices_ from this instance,
  // allowing this instance to be reused on the same column.
  void ResetParentIndices();

 private:
  ParentIndicesBuilder(const std::vector<RepetitionType>& repetition_pattern);

  // repetition_pattern_[0] should always be kRepeated.
  // Sample repetition_pattern_ is {kRepeated, kRepeated, kOptional, kRequired}
  const std::vector<RepetitionType> repetition_pattern_;

  // A vector holding vectors of parent indices of each field.
  // Sample parent_indices_ is {{0, 0}, {0, 1}}.
  std::vector<std::vector<tensorflow::int64>> parent_indices_;

  // The maximum possible definition level of a column is
  // repetition_pattern_.size() - num_required_fields - 1
  const int16_t max_definition_level_;

  // The maximum possible repetition level of a column is the number of
  // repeated fields, not including the root. max_repetition_level_ will be 0
  // if the only repeated field is the root.
  const int16_t max_repetition_level_;
};

}  // namespace parquet_dataset
}  // namespace struct2tensor

#endif  // THIRD_PARTY_PY_STRUCT2TENSOR_GOOGLE_PARQUET_KERNEL_PARENT_INDICES_BUILDER_H_
