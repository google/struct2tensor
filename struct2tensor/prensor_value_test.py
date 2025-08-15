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
"""Tests for struct2tensor.create_expression."""

import tensorflow as tf

from struct2tensor import (
    path,
    prensor,  # pylint: disable=unused-import
)
from struct2tensor.test import prensor_test_util


# This is a V1 test. The prensor value is not meaningful for V2.
class PrensorValueTest(tf.test.TestCase):
    def test_session_run(self):
        """Tests get_sparse_tensors on a deep expression."""
        if tf.executing_eagerly():
            return

        with self.cached_session(use_gpu=False) as sess:
            pren = prensor_test_util.create_nested_prensor()
            mat = sess.run(pren)
            self.assertAllEqual(
                mat.get_descendant_or_error(path.Path(["doc", "bar"])).node.values,
                [b"a", b"b", b"c", b"d"],
            )
            self.assertAllEqual(
                mat.get_descendant_or_error(
                    path.Path(["doc", "bar"])
                ).node.parent_index,
                [0, 1, 1, 2],
            )
            self.assertAllEqual(
                mat.get_descendant_or_error(path.Path(["doc"])).node.parent_index,
                [0, 1, 1],
            )
            self.assertAllEqual(mat.get_descendant_or_error(path.Path([])).node.size, 3)
            self.assertAllEqual(
                mat.get_descendant_or_error(
                    path.Path(["doc", "keep_me"])
                ).node.parent_index,
                [0, 1],
            )
            self.assertAllEqual(
                mat.get_descendant_or_error(path.Path(["doc", "keep_me"])).node.values,
                [False, True],
            )

    def test_children_order(self):
        # Different evaluations of the same prensor object should result in the same
        # prensor value objects.
        if tf.executing_eagerly():
            return

        def _check_children(pv):
            self.assertEqual(
                sorted(pv.get_children().keys()), list(pv.get_children().keys())
            )
            for child in pv.get_children().values():
                _check_children(child)

        with self.cached_session(use_gpu=False) as sess:
            p = prensor_test_util.create_nested_prensor()
            _check_children(sess.run(p))

        with self.cached_session(use_gpu=False) as sess:
            p = prensor.create_prensor_from_descendant_nodes(
                {
                    path.Path([]): prensor_test_util.create_root_node(1),
                    path.Path(["d"]): prensor_test_util.create_optional_leaf_node(
                        [0], [True]
                    ),
                    path.Path(["c"]): prensor_test_util.create_optional_leaf_node(
                        [0], [True]
                    ),
                    path.Path(["b"]): prensor_test_util.create_optional_leaf_node(
                        [0], [True]
                    ),
                    path.Path(["a"]): prensor_test_util.create_optional_leaf_node(
                        [0], [True]
                    ),
                }
            )
            pv = sess.run(p)
            self.assertEqual(["a", "b", "c", "d"], list(pv.get_children().keys()))


if __name__ == "__main__":
    tf.test.main()
