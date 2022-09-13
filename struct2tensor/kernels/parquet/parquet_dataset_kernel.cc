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
#include "absl/container/flat_hash_map.h"
#include "struct2tensor/kernels/parquet/parquet_reader.h"
#include "struct2tensor/kernels/parquet/parquet_reader_util.h"
#include "struct2tensor/kernels/vector_to_tensor.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"

namespace struct2tensor {
namespace parquet_dataset {

class Dataset : public tensorflow::data::DatasetBase {
 public:
  explicit Dataset(tensorflow::OpKernelContext* ctx,
                   const std::vector<std::string>& filenames,
                   const std::vector<std::string>& value_paths,
                   const tensorflow::DataTypeVector& value_dtypes,
                   const std::vector<std::vector<int>>& segregated_path_indices,
                   const tensorflow::int64 batch_size,
                   const tensorflow::DataTypeVector& output_dtypes)
      : DatasetBase(tensorflow::data::DatasetContext(ctx)),
        filenames_(filenames),
        value_paths_(value_paths),
        value_dtypes_(value_dtypes),
        segregated_path_indices_(segregated_path_indices),
        batch_size_(batch_size),
        output_dtypes_(output_dtypes),
        output_shapes_([this]() {
          // The first output tensor is always the root size (number of messages
          // read) which is a scalar. Other output tensors are parent indices
          // so they are 1-D.
          std::vector<tensorflow::PartialTensorShape> shapes(
              output_dtypes_.size(), tensorflow::PartialTensorShape({-1}));

          shapes[0] = tensorflow::PartialTensorShape({});
          return shapes;
        }()) {}

  std::unique_ptr<tensorflow::data::IteratorBase> MakeIteratorInternal(
      const std::string& prefix) const override {
    return absl::WrapUnique(new Iterator(
        {this, tensorflow::strings::StrCat(prefix, "::Parquet")}, filenames_,
        value_paths_, value_dtypes_, segregated_path_indices_, batch_size_));
  }

  const tensorflow::DataTypeVector& output_dtypes() const override {
    return output_dtypes_;
  }
  const std::vector<tensorflow::PartialTensorShape>& output_shapes()
      const override {
    return output_shapes_;
  }

  std::string DebugString() const override {
    return "ParquetDatasetOp::Dataset";
  }

  tensorflow::Status CheckExternalState() const
  {
    return tensorflow::OkStatus();
  }

 protected:
  // TODO(andylou): Implement saving dataset state.
  tensorflow::Status AsGraphDefInternal(
      tensorflow::data::SerializationContext* ctx, DatasetGraphDefBuilder* b,
      tensorflow::Node** output) const override {
    return tensorflow::errors::Unimplemented(
        DebugString(), " does not support serialization.");
  }

 private:
  class Iterator : public tensorflow::data::DatasetIterator<Dataset> {
   public:
    explicit Iterator(
        const Params& params, const std::vector<std::string>& filenames,
        const std::vector<std::string>& value_paths,
        const tensorflow::DataTypeVector& value_dtypes,
        const std::vector<std::vector<int>>& segregated_path_indices,
        const tensorflow::int64 batch_size)
        : DatasetIterator<Dataset>(params),
          filenames_(filenames),
          value_paths_(value_paths),
          value_dtypes_(value_dtypes),
          segregated_path_indices_(segregated_path_indices),
          batch_size_(batch_size),
          current_file_index_(0) {}

    // For a deeper understanding of what tensors are returned in out_tensors,
    // see parquet_dataset_op.cc.
    tensorflow::Status GetNextInternal(
        tensorflow::data::IteratorContext* ctx,
        std::vector<tensorflow::Tensor>* out_tensors,
        bool* end_of_sequence) override {
      tensorflow::mutex_lock l(mu_);
      if (current_file_index_ >= filenames_.size()) {
        *end_of_sequence = true;
        return tensorflow::OkStatus();
      }

      if (!parquet_reader_) {
        // Once a file is finished reading, this will create a ParquetReader
        // for the next file in file_names_.
        TF_RETURN_IF_ERROR(
            ValidateFileAndSchema(filenames_[current_file_index_]));
        TF_RETURN_IF_ERROR(ParquetReader::Create(
            filenames_[current_file_index_], value_paths_, value_dtypes_,
            batch_size_, &parquet_reader_));
      }

      bool end_of_file = false;
      std::vector<ParquetReader::ParentIndicesAndValues>
          parent_indices_and_values;
      TF_RETURN_IF_ERROR(parquet_reader_->ReadMessages(
          ctx, &parent_indices_and_values, &end_of_file));
      if (end_of_file) {
        ++current_file_index_;
        parquet_reader_.reset();
      }

      // pushes the number of messages read as the first output tensor.
      tensorflow::Tensor root_tensor(ctx->allocator({}), tensorflow::DT_INT64,
                                     {});

      if (parent_indices_and_values.size() != value_paths_.size()) {
        return tensorflow::errors::Internal(absl::StrCat(
            parent_indices_and_values.size(),
            " messages read, expected to read ", value_paths_.size()));
      }
      if (parent_indices_and_values[0].parent_indices.empty()) {
        return tensorflow::errors::Internal(
            absl::StrCat("0 messages read, expected to read ", batch_size_));
      }

      root_tensor.flat<tensorflow::int64>()(0) =
          parent_indices_and_values[0].parent_indices[0].size();
      out_tensors->push_back(std::move(root_tensor));

      for (int column_index = 0; column_index < value_paths_.size();
           ++column_index) {
        for (int path_index : segregated_path_indices_[column_index]) {
          tensorflow::Tensor res(
              ctx->allocator({}), tensorflow::DT_INT64,
              {static_cast<long long>(parent_indices_and_values[column_index]
                                          .parent_indices[path_index]
                                          .size())});
          struct2tensor::VectorToTensor(parent_indices_and_values[column_index]
                                            .parent_indices[path_index],
                                        &res,
                                        /*produce_string_view=*/false);
          out_tensors->push_back(std::move(res));
        }
        out_tensors->push_back(
            std::move(parent_indices_and_values[column_index].values));
      }

      return tensorflow::OkStatus();
    }

   protected:
    // TODO(b/139440495): Implement saving and restoring iterator state.
    tensorflow::Status SaveInternal(
        tensorflow::data::SerializationContext* ctx,
        tensorflow::data::IteratorStateWriter* writer)
    {
      return tensorflow::errors::Unimplemented(
          "Parquet Dataset Iterator does not support checkpointing.");
    }

    tensorflow::Status RestoreInternal(
        tensorflow::data::IteratorContext* ctx,
        tensorflow::data::IteratorStateReader* reader)
    {
      return tensorflow::errors::Unimplemented(
          "Parquet Dataset Iterator does not support checkpointing.");
    }

   private:
    // validates that the file exists and can be opened as a parquet file.
    // validates that the schema is the expected schema.
    tensorflow::Status ValidateFileAndSchema(const std::string& filename) {
      std::unique_ptr<parquet::ParquetFileReader> file_reader;
      tensorflow::Status s = OpenFileWithStatus(filename, &file_reader);

      absl::flat_hash_map<std::string, tensorflow::DataType> paths;
      std::shared_ptr<parquet::FileMetaData> file_metadata =
          file_reader->metadata();
      for (int i = 0; i < file_metadata->num_columns(); ++i) {
        std::string path =
            file_metadata->schema()->Column(i)->path()->ToDotString();
        switch (file_metadata->schema()->Column(i)->physical_type()) {
          case parquet::Type::INT32:
            paths[path] = tensorflow::DT_INT32;
            break;
          case parquet::Type::INT64:
            paths[path] = tensorflow::DT_INT64;
            break;
          case parquet::Type::FLOAT:
            paths[path] = tensorflow::DT_FLOAT;
            break;
          case parquet::Type::DOUBLE:
            paths[path] = tensorflow::DT_DOUBLE;
            break;
          case parquet::Type::BOOLEAN:
            paths[path] = tensorflow::DT_BOOL;
            break;
          case parquet::Type::BYTE_ARRAY:
            paths[path] = tensorflow::DT_STRING;
            break;
          default:
            return tensorflow::errors::Unimplemented(absl::StrCat(
                "This Parquet Data Type is unimplemented ",
                file_metadata->schema()->Column(i)->physical_type()));
        }
      }
      for (int i = 0; i < value_dtypes_.size(); ++i) {
        auto paths_iter = paths.find(value_paths_[i]);
        if (paths_iter == paths.end()) {
          return tensorflow::errors::InvalidArgument(
              absl::StrCat("path not found ", value_paths_[i]));
        } else if (paths_iter->second != value_dtypes_[i]) {
          return tensorflow::errors::InvalidArgument(
              absl::StrCat("This dtype is incorrect: ", value_dtypes_[i],
                           ". dtype should be: ", paths_iter->second));
        }
      }
      return s;
    }

    const std::vector<std::string>& filenames_;
    const std::vector<std::string>& value_paths_;
    const tensorflow::DataTypeVector& value_dtypes_;
    const std::vector<std::vector<int>>& segregated_path_indices_;
    const tensorflow::int64 batch_size_;
    int current_file_index_ ABSL_GUARDED_BY(mu_);
    std::unique_ptr<ParquetReader> parquet_reader_ ABSL_GUARDED_BY(mu_);
    tensorflow::mutex mu_;
  };

  const std::vector<std::string> filenames_;
  const std::vector<std::string> value_paths_;
  const tensorflow::DataTypeVector value_dtypes_;

  // 2D vectore to tell us which parent_indices from the path we want. i.e.
  // [[0,1],[0]] means we want the 0th field and 1st field of the 0th path, and
  // the 0th field of the 1st path.
  const std::vector<std::vector<int>> segregated_path_indices_;
  const tensorflow::int64 batch_size_;
  const tensorflow::DataTypeVector output_dtypes_;
  const std::vector<tensorflow::PartialTensorShape> output_shapes_;
};

class ParquetDatasetOp : public tensorflow::data::DatasetOpKernel {
 public:
  ParquetDatasetOp(tensorflow::OpKernelConstruction* ctx)
      : DatasetOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("value_paths", &value_paths_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("value_dtypes", &value_dtypes_));
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("parent_index_paths", &parent_index_paths_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("path_index", &path_index_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("batch_size", &batch_size_));
  }

  void MakeDataset(tensorflow::OpKernelContext* ctx,
                   tensorflow::data::DatasetBase** output) override {
    const tensorflow::Tensor* filenames_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("filenames", &filenames_tensor));

    std::vector<std::string> filenames;
    filenames.reserve(filenames_tensor->NumElements());
    for (int i = 0; i < filenames_tensor->NumElements(); ++i) {
      filenames.push_back(filenames_tensor->flat<tensorflow::tstring>()(i));
    }

    tensorflow::DataTypeVector output_dtypes = tensorflow::DataTypeVector();

    int column_counter = 0;
    std::string prev = parent_index_paths_[0];
    output_dtypes.push_back(tensorflow::DT_INT64);
    for (int i = 1; i < parent_index_paths_.size(); ++i) {
      std::string curr = parent_index_paths_[i];
      output_dtypes.push_back(tensorflow::DT_INT64);
      if (curr != prev) {
        output_dtypes.push_back(value_dtypes_[column_counter]);
        ++column_counter;
        prev = curr;
      }
    }
    output_dtypes.push_back(tensorflow::DT_INT64);
    output_dtypes.push_back(value_dtypes_[column_counter]);

    // This validates that parent_index_paths is aligned with value_paths,
    // so segregated_path_indices can correctly be constructed.
    for (int i = 0, j = 0; i < parent_index_paths_.size(); ++i) {
      while (parent_index_paths_[i] != value_paths_[j]) {
        ++j;
        if (j >= value_paths_.size()) {
          ctx->CtxFailure(tensorflow::errors::InvalidArgument(
              "parent_index_paths is not aligned with value_paths"));
          return;
        }
      }
    }

    std::vector<std::vector<int>> segregated_path_indices(value_paths_.size());

    // This is used to transform path_index to a 2d vector, splitting it up
    // by clustering the same paths. for example: [0, 1, 2, 0, 1, 0, 1, 2, 3]
    // becomes: [[0, 1, 2], [0, 1], [0, 1, 2, 3]]
    for (int i = 0, j = 0; i < parent_index_paths_.size(); ++i) {
      if (parent_index_paths_[i] == value_paths_[j]) {
        segregated_path_indices[j].push_back(path_index_[i] + 1);
      }
      if (i < parent_index_paths_.size() - 1 &&
          parent_index_paths_[i + 1] != parent_index_paths_[i]) {
        ++j;
      }
    }

    *output = new Dataset(ctx, filenames, value_paths_, value_dtypes_,
                          segregated_path_indices, batch_size_, output_dtypes);
  }

 private:
  std::vector<std::string> value_paths_;
  tensorflow::DataTypeVector value_dtypes_;

  // Paths of parent indices that we want. For example:
  // ["DocId", "Name.Language.Code", "Name.Language.Code", "Name.Language.Code"]
  std::vector<std::string> parent_index_paths_;
  std::vector<int> path_index_;
  int batch_size_;
};

// Register the kernel implementation for ParquetDataset.
REGISTER_KERNEL_BUILDER(Name("ParquetDataset").Device(tensorflow::DEVICE_CPU),
                        ParquetDatasetOp);
}  // namespace parquet_dataset
}  // namespace struct2tensor
