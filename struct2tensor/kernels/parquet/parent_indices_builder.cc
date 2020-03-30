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
#include "struct2tensor/kernels/parquet/parent_indices_builder.h"

#include "absl/strings/str_cat.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"

namespace struct2tensor {
namespace parquet_dataset {

tensorflow::Status ParentIndicesBuilder::Create(
    std::vector<RepetitionType> repetition_pattern,
    std::unique_ptr<ParentIndicesBuilder>* parent_indices_builder) {
  if (repetition_pattern.empty()) {
    return tensorflow::errors::OutOfRange(
        "repetition_pattern cannot be empty.");
  }

  if (repetition_pattern[0] != RepetitionType::kRepeated) {
    return tensorflow::errors::InvalidArgument(
        absl::StrCat("repetition_pattern[0]: ", repetition_pattern[0],
                     " != ", RepetitionType::kRepeated,
                     ". The first repetition label must be kRepeatetd."));
  }

  *parent_indices_builder =
      absl::WrapUnique(new ParentIndicesBuilder(repetition_pattern));
  return tensorflow::Status::OK();
}

const std::vector<std::vector<tensorflow::int64>>&
ParentIndicesBuilder::GetParentIndices() const {
  return parent_indices_;
}

const std::vector<ParentIndicesBuilder::RepetitionType>&
ParentIndicesBuilder::GetRepetitionPattern() const {
  return repetition_pattern_;
}

void ParentIndicesBuilder::ResetParentIndices() {
  for (int i = 0; i < parent_indices_.size(); ++i) {
    parent_indices_[i].clear();
  }
}

void ParentIndicesBuilder::AddParentIndices(const int16_t definition_level,
                                            int16_t repetition_level) {
  int num_non_required = 0;
  // Loop invariant:
  // the following is true for all x, where 0 < x < i,
  // parent_indices_[x].back() == parent_indices[x-1].size()
  for (int i = 0; i < repetition_pattern_.size(); ++i) {
    if (repetition_pattern_[i] != RepetitionType::kRequired) {
      ++num_non_required;
    }
    if (max_definition_level_ > 0 && definition_level < max_definition_level_ &&
        num_non_required > definition_level + 1) {
      // this handles the case where the value is NONE
      // if there is a NONE, then its definition level would be less than the
      // max definition level.
      // We also need to make sure that max_definition_level > 0. Because if
      // max_definition_level_ == 0, only required fields exist, meaning
      // there cannot possibly be a none. In that case, the definition
      // level is not needed. So we can ignore it completely.
      break;
    }
    // We need to check max_repetition_level_ because repetition_level may be
    // arbitrary when it's not applicable.
    if (max_repetition_level_ > 0 && repetition_level > 0) {
      if (repetition_pattern_[i] == RepetitionType::kRepeated) {
        --repetition_level;
      }
      // if the field was REQUIRED or OPTIONAL, we don't need to do anything,
      // because repetition level is > 0, i.e. we still have not found
      // which level to start adding parent indices to
    } else {
      // Either all fields (except the root) in the pattern are not repeated or
      // repetition_level is 0 in this branch.
      if (i == 0) {  // we are at the root, so all parent indices are 0
        parent_indices_[i].push_back(0);
      } else {  // we are on a child or leaf
        const tensorflow::int64 num_parents = parent_indices_[i - 1].size() - 1;
        const bool parent_index_exists =
            !parent_indices_[i].empty() &&
            parent_indices_[i].back() == num_parents;
        if (repetition_pattern_[i] != RepetitionType::kOptional ||
            !parent_index_exists) {
          // if this field is repeated/required (i.e. we can always add),
          // OR the current parent index does not exist (i.e. it needs to be
          // added), then we need to add its parent index
          parent_indices_[i].push_back(num_parents);
        }
      }
    }
  }
}

ParentIndicesBuilder::ParentIndicesBuilder(
    const std::vector<RepetitionType>& repetition_pattern)
    : repetition_pattern_(repetition_pattern),
      parent_indices_(repetition_pattern_.size()),
      max_definition_level_([this]() -> int16_t {
        int16_t num_optional_or_repeated = 0;
        for (const auto repetition_type : repetition_pattern_) {
          if (repetition_type != RepetitionType::kRequired) {
            ++num_optional_or_repeated;
          }
        }
        return num_optional_or_repeated - 1;
      }()),
      max_repetition_level_([this]() -> int16_t {
        int16_t num_repeated = 0;
        for (const auto repetition_type : repetition_pattern_) {
          if (repetition_type == RepetitionType::kRepeated) {
            ++num_repeated;
          }
        }
        return num_repeated - 1;
      }()) {}

}  // namespace parquet_dataset
}  // namespace struct2tensor
