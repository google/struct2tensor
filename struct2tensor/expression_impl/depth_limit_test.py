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

"""Tests for struct2tensor.depth_limit."""

from absl.testing import absltest

from struct2tensor import create_expression
from struct2tensor import path
from struct2tensor.expression_impl import depth_limit
from struct2tensor.test import prensor_test_util


class DepthLimitTest(absltest.TestCase):

  def test_depth_limit_1(self):
    """Tests depth_limit with a limit of 1.

    Starting with a prensor expression representing:
    {foo:9, foorepeated:[9], user:[{friends:["a"]}],
     event:{doc:[{bar:["a"], keep_me:False}]}}
    {foo:8, foorepeated:[8,7],
     event:{doc:[{bar:["b","c"], keep_me:True},{bar:["d"]}]},
     user:[{friends:["b", "c"]}, {friends:["d"]}]}
    {foo:7, foorepeated:[6], user:[friends:["e"]], event:{}}

    After depth_limit.limit_depth(expr, 1), you lose event.doc, user.friends,
    event.doc.bar, and event.doc.keep_me.

    {foo:9, foorepeated:[9], user:[{}], event:{}}
    {foo:8, foorepeated:[8,7], event:{},user:[{}, {}]}
    {foo:7, foorepeated:[6], user:[{}], event:{}}

    """
    expr = create_expression.create_expression_from_prensor(
        prensor_test_util.create_deep_prensor())
    new_root = depth_limit.limit_depth(expr, 1)

    self.assertIsNone(
        new_root.get_descendant(path.Path(["event", "doc", "bar"])))
    self.assertIsNone(
        new_root.get_descendant(path.Path(["event", "doc", "keep_me"])))

    self.assertIsNone(new_root.get_descendant(path.Path(["user", "friends"])))
    self.assertIsNone(new_root.get_descendant(path.Path(["event", "doc"])))

    self.assertIsNotNone(new_root.get_descendant(path.Path(["foo"])))
    self.assertIsNotNone(new_root.get_descendant(path.Path(["foorepeated"])))
    self.assertIsNotNone(new_root.get_descendant(path.Path(["user"])))
    self.assertIsNotNone(new_root.get_descendant(path.Path(["event"])))

  def test_depth_limit_2(self):
    """Tests depth_limit with a limit of 2.

    Starting with a prensor expression representing:
    {foo:9, foorepeated:[9], user:[{friends:["a"]}],
     event:{doc:[{bar:["a"], keep_me:False}]}}
    {foo:8, foorepeated:[8,7],
     event:{doc:[{bar:["b","c"], keep_me:True},{bar:["d"]}]},
     user:[{friends:["b", "c"]}, {friends:["d"]}]}
    {foo:7, foorepeated:[6], user:[friends:["e"]], event:{}}

    After depth_limit.limit_depth(expr, 2), you lose event.doc.bar
    and event.doc.keep_me:

    {foo:9, foorepeated:[9], user:[{friends:["a"]}], event:{doc:[{}]}}
    {foo:8, foorepeated:[8,7], event:{doc:[{},{}]},
     user:[{friends:["b", "c"]}, {friends:["d"]}]}
    {foo:7, foorepeated:[6], user:[friends:["e"]], event:{}}

    """
    expr = create_expression.create_expression_from_prensor(
        prensor_test_util.create_deep_prensor(), validate_step_format=False)
    new_root = depth_limit.limit_depth(expr, 2)
    self.assertFalse(new_root.validate_step_format)
    self.assertIsNone(
        new_root.get_descendant(path.Path(["event", "doc", "bar"])))
    self.assertIsNone(
        new_root.get_descendant(path.Path(["event", "doc", "keep_me"])))

    self.assertIsNotNone(new_root.get_descendant(path.Path(["foo"])))
    self.assertIsNotNone(new_root.get_descendant(path.Path(["foorepeated"])))
    self.assertIsNotNone(new_root.get_descendant(path.Path(["user"])))
    self.assertIsNotNone(
        new_root.get_descendant(path.Path(["user", "friends"])))
    self.assertIsNotNone(new_root.get_descendant(path.Path(["event"])))
    self.assertIsNotNone(new_root.get_descendant(path.Path(["event", "doc"])))


if __name__ == "__main__":
  absltest.main()
