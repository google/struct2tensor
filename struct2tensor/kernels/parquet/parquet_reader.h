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
#ifndef THIRD_PARTY_PY_STRUCT2TENSOR_GOOGLE_PARQUET_KERNEL_PARQUET_READER_H_
#define THIRD_PARTY_PY_STRUCT2TENSOR_GOOGLE_PARQUET_KERNEL_PARQUET_READER_H_

#include <string>

#include "absl/strings/str_cat.h"
#include "parquet/api/reader.h"
#include "struct2tensor/kernels/parquet/parent_indices_builder.h"
#include "tensorflow/core/framework/dataset.h"

namespace struct2tensor {
namespace parquet_dataset {
namespace internal {

class PeekableColumnReaderBase {
 public:
  PeekableColumnReaderBase() = default;
  virtual ~PeekableColumnReaderBase() = default;
};
}  // namespace internal

// A class that reads requested columns from parquet, and converts the
// definition and repetition levels to parent_indices and values.
// This class is thread-compatible.
// Sample usage:
// std::unique_ptr<ParquetReader> parquet_reader;
// ParquetReader::Create(
//        filename, value_paths, value_dtypes, batch_size, &parquet_reader_);
// bool end_of_sequence = false;
// std::vector<ParquetReader::ParentIndicesAndValues> p_i_and_values;
// // ctx is a kernel context used for allocating Tensors.
// parquet_reader_->ReadMessages(ctx, &p_i_and_values, &end_of_file);
// For example usage, see parquet_dataset_kernel.cc.
class ParquetReader {
 public:
  // Factory method for creating ParquetReaders.
  // Initializes class member variables needed from the metadata.
  // Returns error status if filename is not a valid file.
  // Returns error status if value_paths contains invalid columns
  // (i.e. doesn't exist in the parquet file).
  // Returns error status if parent_indices_builders_ is not successfully
  // created.
  static tensorflow::Status Create(
      const std::string& filename, const std::vector<std::string>& value_paths,
      const tensorflow::DataTypeVector& value_dtypes,
      const tensorflow::int64 batch_size,
      std::unique_ptr<ParquetReader>* parquet_reader);

  ParquetReader& operator=(const ParquetReader&) = delete;

  ParquetReader(const ParquetReader&) = delete;

  // Bundles parent indices with its respective values tensor.
  struct ParentIndicesAndValues {
    std::vector<std::vector<tensorflow::int64>> parent_indices;
    tensorflow::Tensor values;

    ParentIndicesAndValues(
        std::vector<std::vector<tensorflow::int64>> parent_indices,
        tensorflow::Tensor values)
        : parent_indices(std::move(parent_indices)),
          values(std::move(values)) {}
  };

  // Reads messages up to the batch size, or until the end of the file.
  // Constructs parent indices based on the repetition and definition levels
  // read.
  // parent_indices_and_values[i] would contain the parent indices
  // and values read from value_paths[i], where value_paths was specified in the
  // factory function.
  // end_of_file is set to true if the call to ReadMessages occurs when the
  // file reaches the end of the file. This will result in GetNextInteral
  // (in parquet_dataset_kernel.cc) opening the next file to read, if there is a
  // next file.
  // Returns an Unimplemented error if the tensorflow data type is not handled.
  // Returns an Internal error if there was a problem reading the levels.
  tensorflow::Status ReadMessages(
      tensorflow::data::IteratorContext* ctx,
      std::vector<ParentIndicesAndValues>* parent_indices_and_values,
      bool* end_of_file);

 private:
  ParquetReader(const std::vector<std::string>& value_paths,
                const tensorflow::DataTypeVector& value_dtypes,
                const tensorflow::int64 batch_size,
                const std::vector<tensorflow::int64>& column_indices,
                std::unique_ptr<parquet::ParquetFileReader> file_reader,
                std::vector<std::unique_ptr<internal::PeekableColumnReaderBase>>
                    peekable_column_readers,
                std::vector<std::unique_ptr<ParentIndicesBuilder>>
                    parent_indices_builders);

  // Initializes peekable_column_readers_ by creating a PeekableColumnReader
  // for each column, and reading the first level of each column.
  // Returns an Unimplemented error status if any of the column dtypes are not
  // supported. Or returns an Internal error status if the column is empty or
  // if there was an error reading the first level.
  tensorflow::Status CreatePeekableColumnReaders();

  // Reads values from one column, until the batch size is reached.
  // After reading the column, it will push the parent indices and then values
  // to out_tensors.
  // Returns an Unimplemented error if the tensorflow data type is not handled.
  // Returns an Internal error if there was a problem reading the levels.
  tensorflow::Status ReadOneColumn(
      tensorflow::data::IteratorContext* ctx, const int column_index,
      std::vector<ParentIndicesAndValues>* parent_indices_and_values,
      int* messages_read);

  // Reads values up to batch size from one column.
  // Writes the levels and values to def_levels, rep_levels, and value_tensor.
  template <typename ParquetDataType, typename T>
  tensorflow::Status ReadOneColumnTemplated(
      tensorflow::data::IteratorContext* ctx, int column_index,
      std::vector<int16_t>* def_levels, std::vector<int16_t>* rep_levels,
      std::vector<tensorflow::Tensor>* value_tensor, int* messages_read);

  // Reads one entire message from the i-th column.
  // We know that we have finished reading one entire message when the
  // repetition level becomes 0 again. This will peek levels until
  // one entire message is read.
  template <typename ParquetDataType, typename T>
  tensorflow::Status ReadOneMessageFromOneColumn(
      const int column_index, std::vector<int16_t>* def_levels,
      std::vector<int16_t>* rep_levels, std::vector<T>* values);

  const std::vector<std::string> value_paths_;
  const tensorflow::DataTypeVector value_dtypes_;

  // Number of messages to read per call of get_next().
  const tensorflow::int64 batch_size_;

  // The index of the column in the parquet file. i.e. [0, 4] means we want the
  // 0th and 4th column in the parquet file.
  const std::vector<tensorflow::int64> column_indices_;

  std::unique_ptr<parquet::ParquetFileReader> file_reader_;

  // Vector of PeekableColumnReaderBase. We will static cast each
  // PeekableColumnReaderBase to create the PeekableColumnReader of the correct
  // data type.
  std::vector<std::unique_ptr<internal::PeekableColumnReaderBase>>
      peekable_column_readers_;

  // Each column needs its own ParentIndicesBuilder,
  // since each column will have a different repretition pattern.
  // This ParentIndicesBuilder will be reused across multiple calls of
  // GetNextInternal().
  const std::vector<std::unique_ptr<ParentIndicesBuilder>>
      parent_indices_builders_;

  // max repetition level for each column.
  const std::vector<int16_t> max_repetition_level_;

  // Counts the number of rows read. This is only incremented when all columns
  // are done reading a row.
  tensorflow::int64 total_rows_read_ = 0;
};

}  // namespace parquet_dataset
}  // namespace struct2tensor

#endif  // THIRD_PARTY_PY_STRUCT2TENSOR_GOOGLE_PARQUET_KERNEL_PARQUET_READER_H_
