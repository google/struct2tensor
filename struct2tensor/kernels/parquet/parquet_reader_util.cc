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
#include "parquet/api/reader.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace struct2tensor {
namespace parquet_dataset {

tensorflow::Status OpenFileWithStatus(
    const std::string& filename,
    std::unique_ptr<parquet::ParquetFileReader>* file_reader) {
  try {
    *file_reader = parquet::ParquetFileReader::OpenFile(filename, false);
    return absl::OkStatus();
  } catch (const parquet::ParquetException& e) {
    return tensorflow::errors::Internal(
        absl::StrCat("Invalid File: ", filename));
  }
}
}  // namespace parquet_dataset
}  // namespace struct2tensor
