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
"""Tests for prensor.path."""

# pylint: disable=protected-access
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from absl.testing import absltest
from struct2tensor.path import create_path
from struct2tensor.path import from_proto
from struct2tensor.path import parse_map_indexing_step
from struct2tensor.path import Path



class PathTest(absltest.TestCase):

  def test_get_child(self):
    original_path = create_path("foo.bar")
    child_path = original_path.get_child("baz")
    self.assertEqual(str(child_path), "foo.bar.baz")

  def test_get_child_root(self):
    original_path = create_path("")
    child_path = original_path.get_child("baz")
    self.assertEqual(str(child_path), "baz")

  def test_root_path(self):
    original_path = create_path("")
    self.assertEmpty(original_path.field_list)

  def test_eq(self):
    self.assertEqual(create_path("foo.bar"), create_path("foo.bar"))
    self.assertNotEqual(create_path("foo.bar"), create_path("foo.baz"))
    self.assertNotEqual(create_path("foo"), create_path("foo.baz"))

  def test_cmp(self):
    self.assertGreater(Path([1]), Path("foo"))
    self.assertLess(Path("foo"), Path([1]))
    self.assertGreater(Path([1]), Path([0]))
    self.assertGreater(create_path("foo.baz"), create_path("foo"))
    self.assertGreater(create_path("foo.baz"), create_path("foo.bar"))
    self.assertLess(create_path("foo"), create_path("foo.bar"))
    self.assertLess(create_path("foo.bar"), create_path("foo.baz"))
    self.assertEqual(create_path("foo.baz"), create_path("foo.baz"))

  def test_extension_begin(self):
    original_path = create_path("(foo.bar.baz.ext).bee")
    self.assertEqual("(foo.bar.baz.ext)", str(original_path.get_parent()))

  def test_extension_end(self):
    original_path = create_path("bee.(foo.bar.baz.ext)")
    self.assertEqual("(foo.bar.baz.ext)", original_path.field_list[1])

  def test_extension_middle(self):
    original_path = create_path("bee.(foo.bar.baz.ext).boo")
    self.assertEqual("(foo.bar.baz.ext)", original_path.field_list[1])
    self.assertEqual("boo", original_path.field_list[2])

  def test_multiple_extensions(self):
    original_path = create_path("bee.(foo.bar.ext).boo.(foo.bar.ext2).buu")
    self.assertEqual("(foo.bar.ext)", original_path.field_list[1])
    self.assertEqual("(foo.bar.ext2)", original_path.field_list[3])

  def test_hash(self):
    self.assertEqual(hash(create_path("foo.bar")), hash(create_path("foo.bar")))

  def test_nonzero(self):
    self.assertTrue(bool(create_path("foo")))
    self.assertFalse(bool(create_path("")))

  def test_get_parent(self):
    self.assertEqual(create_path("foo.bar").get_parent(), create_path("foo"))

  def test_concat(self):
    self.assertEqual(
        create_path("foo.bar").concat(create_path("baz.bax")),
        create_path("foo.bar.baz.bax"))

  def test_get_least_common_ancestor(self):
    self.assertEqual(
        create_path("foo.bar.bax").get_least_common_ancestor(
            create_path("foo.bar.baz")), create_path("foo.bar"))

  def test_get_least_common_ancestor_root(self):
    self.assertEqual(
        create_path("foo.bar.bax").get_least_common_ancestor(
            create_path("different.bar.baz")), create_path(""))

  def test_is_ancestor(self):
    ancestor = create_path("foo.bar")
    descendant = create_path("foo.bar.baz")
    not_ancestor = create_path("fuzz")
    self.assertEqual(ancestor.is_ancestor(descendant), True)
    self.assertEqual(ancestor.is_ancestor(ancestor), True)
    self.assertEqual(not_ancestor.is_ancestor(descendant), False)
    self.assertEqual(descendant.is_ancestor(ancestor), False)

  def test_dash_in_name(self):
    simple_step = create_path("foo-bar.baz")
    self.assertEqual(str(simple_step), "foo-bar.baz")
    first_extension = create_path("(foo-bar.bak).baz")
    self.assertEqual(str(first_extension), "(foo-bar.bak).baz")
    last_extension = create_path("(foo.bar-bak).baz")
    self.assertEqual(str(last_extension), "(foo.bar-bak).baz")

  def test_suffix(self):
    original = create_path("foo.bar.baz")
    self.assertEqual(str(original.suffix(0)), "foo.bar.baz")
    self.assertEqual(str(original.suffix(1)), "bar.baz")
    self.assertEqual(str(original.suffix(2)), "baz")
    self.assertEqual(str(original.suffix(3)), "")
    self.assertEqual(str(original.suffix(-1)), "baz")
    self.assertEqual(str(original.suffix(-2)), "bar.baz")

  def test_prefix(self):
    original = create_path("foo.bar.baz")
    self.assertEqual(str(original.prefix(0)), "")
    self.assertEqual(str(original.prefix(1)), "foo")
    self.assertEqual(str(original.prefix(2)), "foo.bar")
    self.assertEqual(str(original.prefix(3)), "foo.bar.baz")
    self.assertEqual(str(original.prefix(-1)), "foo.bar")
    self.assertEqual(str(original.prefix(-2)), "foo")

  def test_len(self):
    self.assertLen(create_path(""), 0)  # pylint: disable=g-generic-assert
    self.assertLen(create_path("foo"), 1)
    self.assertLen(create_path("foo.bar"), 2)
    self.assertLen(create_path("foo.bar.baz"), 3)

  def test_str(self):
    self.assertEqual(str(create_path("foo.bar.bax")), "foo.bar.bax")

  def test_empty_step(self):
    with self.assertRaises(ValueError):
      create_path("foo..bax")

  def test_non_alpha(self):
    with self.assertRaises(ValueError):
      create_path("foo.+.bax")

  def test_empty_extension_step(self):
    with self.assertRaises(ValueError):
      create_path("foo.(.baz).bax")

  def test_get_child_bad_field(self):
    with self.assertRaises(ValueError):
      create_path("foo.bax").get_child("foo.bar")

  def test_empty_extension(self):
    with self.assertRaises(ValueError):
      create_path("foo.().bax")

  def test_mismatched_open_paren(self):
    with self.assertRaises(ValueError):
      create_path("foo.(bar.bax")

  def test_mismatched_closed_paren(self):
    with self.assertRaises(ValueError):
      create_path("foo.bar).bax")

  def test_matched_middle_paren(self):
    with self.assertRaises(ValueError):
      create_path("foo.b(ar).bax")

  def test_valid_map_indexing_step(self):
    self.assertSequenceEqual(
        ["my_map[some_key]", "some_value"],
        create_path("my_map[some_key].some_value").field_list)
    self.assertSequenceEqual(
        ["my_map[]", "some_value"],
        create_path("my_map[].some_value").field_list)
    self.assertSequenceEqual(
        ["my_map[key.1]", "some_value"],
        create_path("my_map[key.1].some_value").field_list)
    self.assertSequenceEqual(
        ["my_map[(key)]", "some_value"],
        create_path("my_map[(key)].some_value").field_list)
    self.assertSequenceEqual(
        ["my_map[[]", "some_value"],
        create_path("my_map[[].some_value").field_list)

  def test_invalid_map_indexing_step(self):
    with self.assertRaises(ValueError):
      create_path("[mymap[s].some_value")
    with self.assertRaises(ValueError):
      create_path("mymap[]].some_value")

  def test_parse_map_indexing_step(self):
    map_field_name, map_key = parse_map_indexing_step("my_map[some_key]")
    self.assertEqual("my_map", map_field_name)
    self.assertEqual("some_key", map_key)

    map_field_name, map_key = parse_map_indexing_step("my_map[]")
    self.assertEqual("my_map", map_field_name)
    self.assertEqual("", map_key)

    map_field_name, map_key = parse_map_indexing_step("my_map[[.]")
    self.assertEqual("my_map", map_field_name)
    self.assertEqual("[.", map_key)

  def test_as_proto(self):
    p = create_path("foo.(bar.e).baz")
    path_proto = p.as_proto()
    self.assertSequenceEqual(path_proto.step, ["foo", "(bar.e)", "baz"])

  def test_proto_roundtrip(self):
    p = create_path("foo.(bar.e).baz")
    path_proto = p.as_proto()
    p_from_proto = from_proto(path_proto)
    self.assertEqual(p, p_from_proto)
    self.assertEqual(path_proto, p_from_proto.as_proto())


if __name__ == "__main__":
  absltest.main()
