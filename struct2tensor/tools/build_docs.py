# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Script to generate api_docs.

blaze run //struct2tensor/opensource_only/tools:build_docs -- \
  --output_dir=$(pwd)/struct2tensor/opensource_only/g3doc/api_docs/python
"""


import inspect


from absl import app
from absl import flags

import struct2tensor as s2t

from tensorflow_docs.api_generator import generate_lib
from tensorflow_docs.api_generator import public_api

flags.DEFINE_string("output_dir", "/tmp/s2t_api", "Where to output the docs")

flags.DEFINE_string(
    "code_url_prefix",
    "https://github.com/google/struct2tensor/blob/master/struct2tensor",
    "The url prefix for links to code.")


flags.DEFINE_bool("search_hints", True,
                  "Include metadata search hints in the generated files")


FLAGS = flags.FLAGS


def _filter_module_attributes(path, parent, children):
  """Filter out module attirubtes.

  This removes attributes that are google_type_annotation or a gen rule
  attribute. The custom ops that need gen rules will have their docs exposed
  from the python source file. No need to also get api docs from the generated
  files.

  Args:
    path: path to parent attribute.
    parent: the attribute (module or class).
    children: the children attributes of parent.

  Returns:
    The new, filtered, children of the current attribute.
  """
  del path
  skip_module_attributes = {
      "google_type_annotations",
      "gen_decode_proto_sparse",
      "gen_decode_proto_map_op",
      "gen_equi_join_indices",
      "gen_parquet_dataset",
      "gen_run_length_before"
  }
  if inspect.ismodule(parent):
    children = [(name, child)
                for (name, child) in children
                if name not in skip_module_attributes]
  return children


def main(unused_argv):

  doc_generator = generate_lib.DocGenerator(
      root_title="Struct2Tensor",
      py_modules=[("s2t", s2t)],
      code_url_prefix=FLAGS.code_url_prefix,
      search_hints=FLAGS.search_hints,
      # Use private_map to exclude doc locations by name if excluding by object
      # is insufficient.
      private_map={},
      # local_definitions_filter ensures that shared modules are only
      # documented in the location that defines them, instead of every location
      # that imports them.
      callbacks=[
          public_api.local_definitions_filter, _filter_module_attributes
      ])

  return doc_generator.build(output_dir=FLAGS.output_dir)


if __name__ == "__main__":
  app.run(main)
