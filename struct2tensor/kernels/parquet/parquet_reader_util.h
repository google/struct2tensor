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
#ifndef THIRD_PARTY_PY_STRUCT2TENSOR_GOOGLE_PARQUET_KERNEL_PARQUET_READER_UTIL_H_
#define THIRD_PARTY_PY_STRUCT2TENSOR_GOOGLE_PARQUET_KERNEL_PARQUET_READER_UTIL_H_

#include "parquet/api/reader.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace struct2tensor {
namespace parquet_dataset {

// This wraps parquet's open to handle exceptions. It's a standalone library
// to be compiled with exception enabled so that the rest of the kernel can be
// exception free.
// Sets file_reader as the file handle and returns status::ok() if successful.
// Returns an internal error if the file is unable to be opened.
tensorflow::Status OpenFileWithStatus(
    const std::string& filename,
    std::unique_ptr<parquet::ParquetFileReader>* file_reader);
}  // namespace parquet_dataset
}  // namespace struct2tensor

#endif  // THIRD_PARTY_PY_STRUCT2TENSOR_GOOGLE_PARQUET_KERNEL_PARQUET_READER_UTIL_H_
