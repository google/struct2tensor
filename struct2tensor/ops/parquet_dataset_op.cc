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
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/framework/shape_inference.h"

// Register the op definition for ParquetDataset.
REGISTER_OP("ParquetDataset")
    .Input("filenames: string")
    .Attr("value_paths: list(string) >= 1")
    .Attr("value_dtypes: list(type) >= 1")
    .Attr("parent_index_paths: list(string) >= 1")
    .Attr("path_index: list(int) >= 1")
    .Attr("batch_size: int = 1")  // TODO(andylou) add a metadata_filename Attr.
    .Output("handle: variant")
    .SetIsStateful()  // TODO(b/123753214): Source dataset ops must be marked
                      // stateful to inhibit constant folding.
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext* c) {
      tensorflow::shape_inference::ShapeHandle input = c->input(0);
      tensorflow::shape_inference::ShapeHandle unused;
      // `filenames` must be a vector.
      TF_RETURN_IF_ERROR(c->WithRank(input, 1, &unused));

      std::vector<std::string> value_paths;
      TF_RETURN_IF_ERROR(c->GetAttr("value_paths", &value_paths));

      std::vector<tensorflow::DataType> value_dtypes;
      TF_RETURN_IF_ERROR(c->GetAttr("value_dtypes", &value_dtypes));

      if (value_paths.size() != value_dtypes.size()) {
        return tensorflow::errors::InvalidArgument(
            absl::StrCat("value_paths.size()=", value_paths.size(),
                         " != value_dtypes.size()=", value_dtypes.size()));
      }

      std::vector<std::string> parent_index_paths;
      TF_RETURN_IF_ERROR(c->GetAttr("parent_index_paths", &parent_index_paths));

      std::vector<int> path_index;
      TF_RETURN_IF_ERROR(c->GetAttr("path_index", &path_index));

      if (parent_index_paths.size() != path_index.size()) {
        return tensorflow::errors::InvalidArgument(absl::StrCat(
            "parent_index_paths.size()=", parent_index_paths.size(),
            " != path_index.size()=", path_index.size()));
      }

      return tensorflow::shape_inference::ScalarShape(c);
    })
    .Doc(R"doc(
Creates a dataset that emits the column data from one or more Parquet files.

filenames: A list containing the name(s) of the file(s) to be read.
value_paths: A list of strings of the dotstring path(s) of each leaf path(s).
value_dtypes: value_dtypes[i] is the Tensorflow data type value_paths[i] would
be of.
parent_index_paths: A list of strings of the dotstring path(s) of the path(s)
to be read. If requesting multiple parent indices of one path, make sure the
same path is consecuative in this list.
i.e. ["DocId", "Name.Language", "Name.Language"] is valid, but
["Name.Language", "DocId", "Name.Language"] is not valid.
The parent_index_paths must also be aligned with value_paths, meaning whatever
order the paths appear in value_paths, the same order must occur
parent_index_paths.
path_index: A list containing the index of each field to get the parent index
of. This must be aligned with parent_index_paths, meaning the i-th element of
path_index, signifies we want the parent index of the path_index[i] step of the
i-th element of parent_index_paths.
batch_size: An optional int that determines how many messages are parsed into
one prensor tree in an iteration. If there are fewer than batch_size
remaining messages, then all remaining messages will be returned.

For example: If we have a group of sharded parquet files, and a metadata file,
we would pass them in as
filenames = ["parquet_0001.parquet", "parquet_0002.parquet", ...].

And if the metadata file contained the following parquet schema:
message Document
  optional group Links
    repeated string Backward
    repeated string Forward
  repeated group Name
    repeated group Language
      required int64 Code
      optional string Country
If we want the parent indices of "Links", "Backward", "Name", "Language",
and "Code", then value_paths would be:
["Links.Backward", "Name.Language.Code"],
and parent_index_paths would be:
["Links.Backward", "Links.Backward",
"Name.Language.Code", "Name.Language.Code", "Name.Language.Code"],
and path_index would be [0, 1, 0, 1, 2].
and value_dtypes would be [int64, string], which would be transformed into:
[int64, int64, int64, int64, int64, int64, int64, string],
for the dtypes of the output vector.

The iterator would then read values of the columns and yield a vector of tensors
that contains the parent indices of each field, and the values.
So following the same example above, the iterator would yield:
[[Number_of_Documents],
 [Links_parent_indices], [Backward_parent_indices], [Backward_values],
 [Name_parent_indices], [Language_parent_indices], [Code_parent_indices],
 [Code_values]]

If batch_size = 5, then Number_of_Documents would be <= 5.

)doc");
