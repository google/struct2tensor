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
#include "struct2tensor/kernels/parquet/parquet_reader.h"

#include <memory>

#include "absl/strings/str_cat.h"
#include "struct2tensor/kernels/parquet/parquet_reader_util.h"
#include "struct2tensor/kernels/vector_to_tensor.h"

namespace struct2tensor {
namespace parquet_dataset {
namespace internal {
// A template class that wraps parquet's column reader.
// This adds a peek functionality to parquet's ReadBatch.
// This class also handles reading across row groups. That means that Peek will
// always return the next level in the parquet file, until we have reached the
// end of the file.
// This class is thread-compatible.
// Sample usage to read all levels in a column:
// auto pcr =
// absl::make_unique<internal::PeekableColumnReader<parquet::Int32Type>>(
//                                                   column_index, file_reader);
// int16_t def_level;
// int16_t rep_level;
// while (pcr->PeekLevels(def_level, rep_level)) {
//   TF_RETURN_IF_ERROR(pcr->Advance());
// }
template <typename ParquetDataType>
class PeekableColumnReader : public PeekableColumnReaderBase {
 public:
  // Factory method for creating PeekableColumnReader.
  // This will read the first level of the file, and initialize member variables
  // based on the level read.
  // Returns an Internal error if the wrong number of levels is read.
  // Returns an OutOfRange error if the file is empty.
  static tensorflow::Status Create(
      const int column_index, parquet::ParquetFileReader* file_reader,
      std::unique_ptr<PeekableColumnReader<ParquetDataType>>* pcr) {
    *pcr = absl::WrapUnique(
        new PeekableColumnReader<ParquetDataType>(column_index, file_reader));
    TF_RETURN_IF_ERROR(pcr->get()->Advance());
    return tensorflow::Status::OK();
  }

  PeekableColumnReader<ParquetDataType>& operator=(
      const PeekableColumnReader<ParquetDataType>&) = delete;

  PeekableColumnReader<ParquetDataType>(
      const PeekableColumnReader<ParquetDataType>&) = delete;

  // Sets the output parameter to the current buffer values.
  // Returns false if there are no levels to peek.
  // This will not touch disk, nor change the column reader's position.
  bool PeekLevels(int16_t* definition_level, int16_t* repetition_level) const {
    if (end_of_column_) {
      return false;
    }
    *definition_level = curr_def_level_;
    *repetition_level = curr_rep_level_;
    return true;
  }

  // Sets the output parameter to the value, if it exists (i.e. it is not None).
  // Returns true if value exists, false otherwise.
  bool PeekValue(typename ParquetDataType::c_type* value) const {
    if (value_exists_) {
      *value = curr_value_;
      return true;
    } else {
      return false;
    }
  }

  // Advances the column reader's position. Updates the buffer.
  // Returns an Internal error if the wrong number of levels is read.
  // Returns an OutOfRange error if we reach the end of the file.
  tensorflow::Status Advance() {
    parquet::TypedColumnReader<ParquetDataType>* typed_column_reader =
        static_cast<parquet::TypedColumnReader<ParquetDataType>*>(
            column_reader_.get());
    while (!typed_column_reader || !typed_column_reader->HasNext()) {
      ++row_group_counter_;
      if (row_group_counter_ < file_reader_->metadata()->num_row_groups()) {
        row_group_reader_ = file_reader_->RowGroup(row_group_counter_);
        column_reader_ = row_group_reader_->Column(column_index_);
        typed_column_reader =
            static_cast<parquet::TypedColumnReader<ParquetDataType>*>(
                column_reader_.get());
      } else {
        end_of_column_ = true;
        return tensorflow::errors::OutOfRange("Reached end of Column");
      }
    }

    int64_t values_read;
    int64_t levels_read = typed_column_reader->ReadBatch(
        1, &curr_def_level_, &curr_rep_level_, &curr_value_, &values_read);
    if (levels_read != 1) {
      return tensorflow::errors::Internal(
          "Expected to read 1 level. Actually read %d level", levels_read);
    }
    value_exists_ = (values_read == 1);
    return tensorflow::Status::OK();
  }

 private:
  // Constructor for PeekableColumnReader.
  // We can ignore the return status of Advance() because Advance() will set
  // the appropriate flags, if something went wrong. If the column was empty,
  // then end_of_column_ is set to true, and Peek() will always return false.
  PeekableColumnReader(const int column_index,
                       parquet::ParquetFileReader* file_reader)
      : column_index_(column_index),
        row_group_counter_(-1),
        end_of_column_(false),
        value_exists_(false),
        curr_def_level_(-1),
        curr_rep_level_(-1),
        file_reader_(file_reader) {}

  const int column_index_;
  int row_group_counter_;
  bool end_of_column_;
  bool value_exists_;
  int16_t curr_def_level_;
  int16_t curr_rep_level_;
  typename ParquetDataType::c_type curr_value_;
  parquet::ParquetFileReader* file_reader_;
  std::shared_ptr<parquet::RowGroupReader> row_group_reader_;
  std::shared_ptr<parquet::ColumnReader> column_reader_;
};
}  // namespace internal

namespace {

template <typename ParquetDataType, typename T>
inline T ParquetTypeBridge(const ParquetDataType value) {
  return value;
}

// template specialization for handling parquet's ByteArray.
template <>
inline tensorflow::tstring ParquetTypeBridge(const parquet::ByteArray value) {
  return parquet::ByteArrayToString(value);
}

// Gets the column index in the parquet file, based on the column name from
// the row group reader.
// Returns error status if the column does not exist.
tensorflow::Status GetColumnIndex(const std::string& column_name,
                                  parquet::ParquetFileReader* file_reader,
                                  int* column_index) {
  *column_index = file_reader->metadata()->schema()->ColumnIndex(column_name);
  if (*column_index == -1) {
    return tensorflow::errors::NotFound(
        absl::StrCat("Column not found: ", column_name));
  }

  return tensorflow::Status::OK();
}

// Creates the repetition pattern for a path. For example "Document.DocID"
// would have repetition pattern {REPEATED, REQUIRED}.
std::vector<ParentIndicesBuilder::RepetitionType> CreateRepetitionPattern(
    const int column_index,
    const std::unique_ptr<parquet::ParquetFileReader>& file_reader) {
  std::vector<ParentIndicesBuilder::RepetitionType> res =
      std::vector<ParentIndicesBuilder::RepetitionType>();
  const parquet::schema::Node* curr = file_reader->metadata()
                                          ->schema()
                                          ->Column(column_index)
                                          ->schema_node()
                                          .get();
  while (curr) {
    if (curr->is_optional()) {
      res.push_back(ParentIndicesBuilder::RepetitionType::kOptional);
    } else if (curr->is_required()) {
      res.push_back(ParentIndicesBuilder::RepetitionType::kRequired);
    } else if (curr->is_repeated()) {
      res.push_back(ParentIndicesBuilder::RepetitionType::kRepeated);
    }
    curr = curr->parent();
  }
  if (res.back() != ParentIndicesBuilder::RepetitionType::kRepeated) {
    LOG(ERROR) << absl::StrCat(
        "The repetition type of the root node was ", res.back(),
        ", but should be ", ParentIndicesBuilder::RepetitionType::kRepeated,
        ". There may be something wrong with your supplied parquet schema. "
        "We will treat it as a repeated field.");
    res[res.size() - 1] = ParentIndicesBuilder::RepetitionType::kRepeated;
  }
  std::reverse(res.begin(), res.end());
  return res;
}
}  // namespace

// Creates a peekable column reader, and appends it to peekable_column_readers.
template <typename ParquetDataType>
tensorflow::Status PopulatePeekableColumnReadersVector(
    int column_index, parquet::ParquetFileReader* file_reader,
    std::vector<std::unique_ptr<internal::PeekableColumnReaderBase>>*
        peekable_column_readers) {
  std::unique_ptr<internal::PeekableColumnReader<ParquetDataType>> pcr;
  TF_RETURN_IF_ERROR(internal::PeekableColumnReader<ParquetDataType>::Create(
      column_index, file_reader, &pcr));
  peekable_column_readers->push_back(std::move(pcr));
  return tensorflow::Status::OK();
}

tensorflow::Status ParquetReader::Create(
    const std::string& filename, const std::vector<std::string>& value_paths,
    const tensorflow::DataTypeVector& value_dtypes,
    const tensorflow::int64 batch_size,
    std::unique_ptr<ParquetReader>* parquet_reader) {
  std::unique_ptr<parquet::ParquetFileReader> file_reader;
  TF_RETURN_IF_ERROR(OpenFileWithStatus(filename, &file_reader));
  // TODO(andylou) add handling of a metadata file, if it is provided.

  std::vector<tensorflow::int64> column_indices;
  std::vector<std::unique_ptr<ParentIndicesBuilder>> parent_indices_builders;
  std::vector<int16_t> max_repetition_level;

  for (int i = 0; i < value_paths.size(); ++i) {
    int index;
    TF_RETURN_IF_ERROR(
        GetColumnIndex(value_paths[i], file_reader.get(), &index));
    column_indices.push_back(index);
  }

  for (int i = 0; i < value_paths.size(); ++i) {
    std::unique_ptr<ParentIndicesBuilder> parent_indices_builder;
    std::vector<ParentIndicesBuilder::RepetitionType> repetition_pattern =
        CreateRepetitionPattern(column_indices[i], file_reader);
    TF_RETURN_IF_ERROR(ParentIndicesBuilder::Create(
        std::move(repetition_pattern), &parent_indices_builder));
    parent_indices_builders.push_back(std::move(parent_indices_builder));
  }

  std::vector<std::unique_ptr<internal::PeekableColumnReaderBase>>
      peekable_column_readers;

  for (int i = 0; i < value_paths.size(); ++i) {
    switch (value_dtypes[i]) {
      case tensorflow::DT_INT32:
        TF_RETURN_IF_ERROR(
            PopulatePeekableColumnReadersVector<parquet::Int32Type>(
                column_indices[i], file_reader.get(),
                &peekable_column_readers));
        break;
      case tensorflow::DT_INT64:
        TF_RETURN_IF_ERROR(
            PopulatePeekableColumnReadersVector<parquet::Int64Type>(
                column_indices[i], file_reader.get(),
                &peekable_column_readers));
        break;
      case tensorflow::DT_FLOAT:
        TF_RETURN_IF_ERROR(
            PopulatePeekableColumnReadersVector<parquet::FloatType>(
                column_indices[i], file_reader.get(),
                &peekable_column_readers));
        break;
      case tensorflow::DT_DOUBLE:
        TF_RETURN_IF_ERROR(
            PopulatePeekableColumnReadersVector<parquet::DoubleType>(
                column_indices[i], file_reader.get(),
                &peekable_column_readers));
        break;
      case tensorflow::DT_BOOL:
        TF_RETURN_IF_ERROR(
            PopulatePeekableColumnReadersVector<parquet::BooleanType>(
                column_indices[i], file_reader.get(),
                &peekable_column_readers));
        break;
      case tensorflow::DT_STRING:
        TF_RETURN_IF_ERROR(
            PopulatePeekableColumnReadersVector<parquet::ByteArrayType>(
                column_indices[i], file_reader.get(),
                &peekable_column_readers));
        break;
      default:
        return tensorflow::errors::Unimplemented(
            DataTypeString(value_dtypes[i]),
            " is currently not supported in ParquetDataset");
    }
  }

  *parquet_reader = absl::WrapUnique(new ParquetReader(
      value_paths, value_dtypes, batch_size, column_indices,
      std::move(file_reader), std::move(peekable_column_readers),
      std::move(parent_indices_builders)));
  return tensorflow::Status::OK();
}

ParquetReader::ParquetReader(
    const std::vector<std::string>& value_paths,
    const tensorflow::DataTypeVector& value_dtypes,
    const tensorflow::int64 batch_size,
    const std::vector<tensorflow::int64>& column_indices,
    std::unique_ptr<parquet::ParquetFileReader> file_reader,
    std::vector<std::unique_ptr<internal::PeekableColumnReaderBase>>
        peekable_column_readers,
    std::vector<std::unique_ptr<ParentIndicesBuilder>> parent_indices_builders)
    : value_paths_(value_paths),
      value_dtypes_(value_dtypes),
      batch_size_(batch_size),
      column_indices_(column_indices),
      file_reader_(std::move(file_reader)),
      peekable_column_readers_(std::move(peekable_column_readers)),
      parent_indices_builders_(std::move(parent_indices_builders)),
      max_repetition_level_([this]() {
        std::vector<int16_t> res = std::vector<int16_t>(value_paths_.size());
        for (int i = 0; i < value_paths_.size(); ++i) {
          res[i] = parent_indices_builders_[i]->GetRepetitionPattern().size();
        }
        return res;
      }()) {}

tensorflow::Status ParquetReader::ReadMessages(
    tensorflow::data::IteratorContext* ctx,
    std::vector<ParentIndicesAndValues>* parent_indices_and_values,
    bool* end_of_file) {
  int prev_column_messages_read = 0;
  for (int column_index = 0; column_index < column_indices_.size();
       ++column_index) {
    int messages_read;
    TF_RETURN_IF_ERROR(ReadOneColumn(
        ctx, column_index, parent_indices_and_values, &messages_read));

    // check if we read the same number of messages at this column as all the
    // previous columns
    if (column_index == 0) {
      prev_column_messages_read = messages_read;
    } else {
      if (messages_read != prev_column_messages_read) {
        return tensorflow::errors::Internal(
            absl::StrCat("Read ", messages_read, " messages, but expected ",
                         prev_column_messages_read));
      }
    }
  }
  total_rows_read_ += prev_column_messages_read;
  if (total_rows_read_ >= file_reader_->metadata()->num_rows()) {
    *end_of_file = true;
    return tensorflow::Status::OK();
  }
  return tensorflow::Status::OK();
}

tensorflow::Status ParquetReader::ReadOneColumn(
    tensorflow::data::IteratorContext* ctx, const int column_index,
    std::vector<ParentIndicesAndValues>* parent_indices_and_values,
    int* messages_read) {
  parent_indices_builders_[column_index]->ResetParentIndices();
  const tensorflow::DataType data_type = value_dtypes_[column_index];
  std::vector<int16_t> def_levels;
  std::vector<int16_t> rep_levels;

  std::vector<tensorflow::Tensor> value_tensor;

  switch (data_type) {
    case tensorflow::DT_INT32:
      TF_RETURN_IF_ERROR(ReadOneColumnTemplated<parquet::Int32Type, int32_t>(
          ctx, column_index, &def_levels, &rep_levels, &value_tensor,
          messages_read));
      break;
    case tensorflow::DT_INT64:
      TF_RETURN_IF_ERROR(
          ReadOneColumnTemplated<parquet::Int64Type, tensorflow::int64>(
              ctx, column_index, &def_levels, &rep_levels, &value_tensor,
              messages_read));
      break;
    case tensorflow::DT_FLOAT:
      TF_RETURN_IF_ERROR(ReadOneColumnTemplated<parquet::FloatType, float>(
          ctx, column_index, &def_levels, &rep_levels, &value_tensor,
          messages_read));
      break;
    case tensorflow::DT_DOUBLE:
      TF_RETURN_IF_ERROR(ReadOneColumnTemplated<parquet::DoubleType, double>(
          ctx, column_index, &def_levels, &rep_levels, &value_tensor,
          messages_read));
      break;
    case tensorflow::DT_BOOL:
      TF_RETURN_IF_ERROR(ReadOneColumnTemplated<parquet::BooleanType, bool>(
          ctx, column_index, &def_levels, &rep_levels, &value_tensor,
          messages_read));
      break;
    case tensorflow::DT_STRING:
      TF_RETURN_IF_ERROR(
          ReadOneColumnTemplated<parquet::ByteArrayType, tensorflow::tstring>(
              ctx, column_index, &def_levels, &rep_levels, &value_tensor,
              messages_read));
      break;
    default:
      return tensorflow::errors::Unimplemented(
          DataTypeString(data_type),
          " is currently not supported in ParquetDataset");
  }

  if (def_levels.size() != rep_levels.size()) {
    return tensorflow::errors::Internal(
        "def level size was not the same as rep level size.. "
        "something is wrong");
  }
  for (int j = 0; j < def_levels.size(); ++j) {
    parent_indices_builders_[column_index]->AddParentIndices(def_levels[j],
                                                             rep_levels[j]);
  }

  std::vector<std::vector<tensorflow::int64>> parent_indices =
      parent_indices_builders_[column_index]->GetParentIndices();

  parent_indices_and_values->emplace_back(
      ParentIndicesAndValues{parent_indices, value_tensor[0]});

  return tensorflow::Status::OK();
}

template <typename ParquetDataType, typename T>
tensorflow::Status ParquetReader::ReadOneColumnTemplated(
    tensorflow::data::IteratorContext* ctx, int column_index,
    std::vector<int16_t>* def_levels, std::vector<int16_t>* rep_levels,
    std::vector<tensorflow::Tensor>* value_tensor, int* messages_read) {
  std::vector<T> cumulative_values;
  *messages_read = 0;
  for (int i = 0; i < batch_size_; ++i) {
    tensorflow::Status s = ReadOneMessageFromOneColumn<ParquetDataType, T>(
        column_index, def_levels, rep_levels, &cumulative_values);

    ++(*messages_read);
    if (tensorflow::errors::IsOutOfRange(s)) {
      break;
    }
    TF_RETURN_IF_ERROR(s);
  }
  tensorflow::Tensor res(ctx->allocator({}), value_dtypes_[column_index],
                         {static_cast<long long>(cumulative_values.size())});
  struct2tensor::VectorToTensor(cumulative_values, &res);
  value_tensor->push_back(res);
  return tensorflow::Status::OK();
}

template <typename ParquetDataType, typename T>
tensorflow::Status ParquetReader::ReadOneMessageFromOneColumn(
    const int column_index, std::vector<int16_t>* def_levels,
    std::vector<int16_t>* rep_levels, std::vector<T>* values) {
  internal::PeekableColumnReader<ParquetDataType>* pcr =
      static_cast<internal::PeekableColumnReader<ParquetDataType>*>(
          peekable_column_readers_[column_index].get());

  int16_t definition_level;
  int16_t repetition_level;
  typename ParquetDataType::c_type value;
  pcr->PeekLevels(&definition_level, &repetition_level);
  do {
    def_levels->push_back(definition_level);
    rep_levels->push_back(repetition_level);
    bool has_value = pcr->PeekValue(&value);
    if (has_value) {
      T bridged_value =
          ParquetTypeBridge<typename ParquetDataType::c_type, T>(value);
      values->push_back(bridged_value);
    }
    TF_RETURN_IF_ERROR(pcr->Advance());
    pcr->PeekLevels(&definition_level, &repetition_level);
  } while (repetition_level > 0 &&
           repetition_level < max_repetition_level_[column_index]);

  return tensorflow::Status::OK();
}

}  // namespace parquet_dataset
}  // namespace struct2tensor
