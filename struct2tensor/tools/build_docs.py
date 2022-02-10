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
import pathlib
import shutil
import tempfile

from absl import app
from absl import flags
import struct2tensor as s2t
from struct2tensor import expression_impl
from tensorflow_docs.api_generator import generate_lib
from tensorflow_docs.api_generator import public_api

import yaml

flags.DEFINE_string("output_dir", "/tmp/s2t_api", "Where to output the docs")

flags.DEFINE_string(
    "code_url_prefix",
    "https://github.com/google/struct2tensor/blob/master/struct2tensor",
    "The url prefix for links to code.")

flags.DEFINE_bool("search_hints", True,
                  "Include metadata search hints in the generated files")


FLAGS = flags.FLAGS


def build_docs(output_dir: pathlib.Path) -> None:
  """Generates the api docs for struct2tensor and expression_impl.

  We need to splice the generated docs together, and create a new table of
  contents.

  Args:
    output_dir: A pathlib Path that is the top level dir where docs are
      generated.

  Returns:
    None
  """
  s2t_out = pathlib.Path(tempfile.mkdtemp())
  doc_generator = generate_lib.DocGenerator(
      root_title="Struct2Tensor",
      py_modules=[("s2t", s2t)],
      code_url_prefix=FLAGS.code_url_prefix,
      search_hints=FLAGS.search_hints,
      # explicit_package_contents_filter ensures that only modules imported
      # directly from s2t/__init__.py are documented in the location that
      # defines them, instead of every location that imports them.
      callbacks=[
          public_api.explicit_package_contents_filter, _filter_module_attributes
      ])

  doc_generator.build(s2t_out)

  expr_impl_out = pathlib.Path(tempfile.mkdtemp())
  doc_generator = generate_lib.DocGenerator(
      root_title="Struct2Tensor-expression_impl",
      py_modules=[("expression_impl", expression_impl)],
      code_url_prefix=FLAGS.code_url_prefix + "/expression_impl",
      search_hints=FLAGS.search_hints,
      # explicit_package_contents_filter ensures that only modules imported
      # directly from s2t/expression_impl/__init__.py are documented in the
      # location that defines them, instead of every location that imports them.
      callbacks=[
          public_api.explicit_package_contents_filter, _filter_module_attributes
      ])
  doc_generator.build(expr_impl_out)

  def splice(name, tmp_dir):
    shutil.rmtree(output_dir / name, ignore_errors=True)
    shutil.copytree(tmp_dir / name, output_dir / name)
    shutil.copy(tmp_dir / f"{name}.md", output_dir / f"{name}.md")
    try:
      shutil.copy(tmp_dir / name / "_redirects.yaml",
                  output_dir / name / "_redirects.yaml")
    except FileNotFoundError:
      pass
    shutil.copy(tmp_dir / name / "_toc.yaml", output_dir / name / "_toc.yaml")

  splice("s2t", s2t_out)
  splice("expression_impl", expr_impl_out)

  toc_path = output_dir / "_toc.yaml"
  toc_text = yaml.dump({
      "toc": [{
          "include": "/api_docs/python/s2t/_toc.yaml"
      }, {
          "break": True
      }, {
          "include": "/api_docs/python/expression_impl/_toc.yaml"
      }]
  })
  toc_path.write_text(toc_text)


def _filter_module_attributes(path, parent, children):
  """Filter out module attirubtes.

  This removes attributes that are a gen rule attribute.
  The custom ops that need gen rules will have their docs exposed
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
  del unused_argv
  build_docs(output_dir=pathlib.Path(FLAGS.output_dir))


if __name__ == "__main__":
  app.run(main)
