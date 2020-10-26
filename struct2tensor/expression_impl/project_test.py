# Copyright 2019 Google LLC
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
"""Tests for struct2tensor.project."""

from absl.testing import absltest
from struct2tensor import create_expression
from struct2tensor import path
from struct2tensor.expression_impl import project
from struct2tensor.test import prensor_test_util


class ProjectTest(absltest.TestCase):

  def test_project(self):
    expr = create_expression.create_expression_from_prensor(
        prensor_test_util.create_nested_prensor())
    projected = project.project(
        expr, [path.Path(["user", "friends"]),
               path.Path(["doc", "keep_me"])])
    self.assertIsNotNone(
        projected.get_descendant(path.Path(["user", "friends"])))
    self.assertIsNotNone(
        projected.get_descendant(path.Path(["doc", "keep_me"])))
    self.assertIsNone(projected.get_descendant(path.Path(["doc", "bar"])))


if __name__ == "__main__":
  absltest.main()
