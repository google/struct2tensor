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
"""Example prensor expressions.

See create_simple_prensor() and create_nested_prensor().

"""

from typing import Any, Sequence

from struct2tensor import path
from struct2tensor import prensor
import tensorflow as tf

from google.protobuf import text_format
from tensorflow_metadata.proto.v0 import schema_pb2


def create_root_node(size: int) -> prensor.RootNodeTensor:
  return prensor.RootNodeTensor(tf.constant(size, dtype=tf.int64))


def create_child_node(parent_index: Sequence[int],
                      is_repeated: bool) -> prensor.ChildNodeTensor:
  return prensor.ChildNodeTensor(
      tf.constant(parent_index, dtype=tf.int64), is_repeated)


def create_optional_leaf_node(parent_index: Sequence[int],
                              values: Sequence[Any]) -> prensor.LeafNodeTensor:
  """Creates an optional leaf node.

  Args:
    parent_index: a list of integers that is converted to a 1-D int64 tensor.
    values: a list of whatever type that the field represents.

  Returns:
    A PrensorField with the parent_index and values set appropriately.
  """
  return prensor.LeafNodeTensor(
      tf.constant(parent_index, dtype=tf.int64), tf.constant(values), False)


def create_repeated_leaf_node(parent_index: Sequence[int],
                              values: Sequence[Any]):
  """Creates a repeated PrensorField.

  Args:
    parent_index: a list of integers that is converted to a 1-D int64 tensor.
    values: a list of whatever type that the field represents.

  Returns:
    A PrensorField with the parent_index and values set appropriately.
  """
  return prensor.LeafNodeTensor(
      tf.constant(parent_index, dtype=tf.int64), tf.constant(values), True)


def create_simple_prensor() -> prensor.Prensor:
  """Creates a prensor expression representing a list of flat protocol buffers.

  Returns:
    a RootPrensor representing:
    {foo:9, foorepeated:[9]}
    {foo:8, foorepeated:[8,7]}
    {foo:7, foorepeated:[6]}
  """
  return prensor.create_prensor_from_descendant_nodes({
      path.Path([]):
          create_root_node(3),
      path.Path(["foo"]):
          create_optional_leaf_node([0, 1, 2], [9, 8, 7]),
      path.Path(["foorepeated"]):
          create_repeated_leaf_node([0, 1, 1, 2], [9, 8, 7, 6])
  })


def create_broken_prensor() -> prensor.Prensor:
  """Creates a broken prensor where the parent indices are out of order.

  Returns:
    a RootPrensor where the parent indices are out of order.
  """
  return prensor.create_prensor_from_descendant_nodes({
      path.Path([]):
          create_root_node(3),
      path.Path(["foo"]):
          create_optional_leaf_node([1, 0, 2], [9, 8, 7]),
      path.Path(["foorepeated"]):
          create_repeated_leaf_node([1, 0, 1, 2], [9, 8, 7, 6])
  })


def create_nested_prensor() -> prensor.Prensor:
  """Creates a prensor representing a list of nested protocol buffers.

  Returns:
    a prensor expression representing:
    {doc:[{bar:["a"], keep_me:False}], user:[{friends:["a"]}]}
    {doc:[{bar:["b","c"], keep_me:True}, {bar:["d"]}],
     user:[{friends:["b", "c"]}, {friends:["d"]}]}
    {user:[friends:["e"]]}
  """
  return prensor.create_prensor_from_descendant_nodes({
      path.Path([]):
          create_root_node(3),
      path.Path(["doc"]):
          create_child_node([0, 1, 1], True),
      path.Path(["doc", "bar"]):
          create_repeated_leaf_node([0, 1, 1, 2], ["a", "b", "c", "d"]),
      path.Path(["doc", "keep_me"]):
          create_optional_leaf_node([0, 1], [False, True]),
      path.Path(["user"]):
          create_child_node([0, 1, 1, 2], True),
      path.Path(["user", "friends"]):
          create_repeated_leaf_node([0, 1, 1, 2, 3], ["a", "b", "c", "d", "e"])
  })


def create_nested_prensor_with_lenient_field_names() -> prensor.Prensor:
  """Creates a prensor representing a data structure with proto-invalid names.

  Returns:
    a prensor expression representing:
    {doc:[{bar.baz:[1], keep_me/x:False}], user:[{friends!:):[1]}]}
    {doc:[{bar:[2, 3], keep_me/x:True}, {bar:[4]}],
     user:[{friends!:):[2, 3]}, {friends!:):[4]}]}
    {user:[friends!:):[5]]}
  """
  return prensor.create_prensor_from_descendant_nodes({
      path.Path([], validate_step_format=False): create_root_node(3),
      path.Path(["doc"], validate_step_format=False): create_child_node(
          [0, 1, 1], True
      ),
      path.Path(
          ["doc", "bar.baz"], validate_step_format=False
      ): create_repeated_leaf_node([0, 1, 1, 2], [1, 2, 3, 4]),
      path.Path(
          ["doc", "keep_me/x"], validate_step_format=False
      ): create_optional_leaf_node(
          [0, 1],
          [False, True],
      ),
      path.Path(["user"], validate_step_format=False): create_child_node(
          [0, 1, 1, 2], True
      ),
      path.Path(
          ["user", "friends!:)"], validate_step_format=False
      ): create_repeated_leaf_node([0, 1, 1, 2, 3], [1, 2, 3, 4, 5]),
  })


def create_big_prensor_schema():
  """Create a schema that is aligned with create_big_prensor."""
  return text_format.Parse(
      """
      int_domain: {
        name: 'zero_to_ten'
        min: 0
        max: 10
      }
      string_domain: {
        name: 'abcde'
        value: ['a', 'b', 'c', 'd', 'e']
      }
      feature: {
        name: 'foo'
        value_count: {max: 1}
        presence: {min_count: 1}
        int_domain: {min: 0 max: 10}
      }
      feature: {
        name: 'foorepeated'
        presence: {min_count: 1}
        domain: 'zero_to_ten'
      }
      feature: {
        name: 'doc'
        struct_domain: {
          feature: {
            name: 'bar'
            domain: 'abcde'
          }
          feature: {
            name: 'keep_me'
            presence: {min_count: 1}
          }
        }
      }
      feature: {
        name: 'user'
        struct_domain: {
          feature: {
            name: 'friends'
            presence: {min_count: 1}
            domain: 'abcde'
          }
        }
      }
      """, schema_pb2.Schema())


def create_big_prensor():
  """Creates a prensor representing a list of nested protocol buffers.

  Returns:
    a prensor expression representing:
    {
      foo: 9,
      foorepeated: [9],
      doc: [{bar:["a"], keep_me:False}],
      user: [{friends:["a"]}]
    },
    {
      foo: 8,
      foorepeated: [8, 7],
      doc: [{bar:["b","c"],keep_me:True},{bar:["d"]}],
      user: [{friends:["b", "c"]},{friends:["d"]}],
    },
    {
      foo: 7,
      foorepeated: [6],
      user: [{friends:["e"]}]
    }
  """
  return prensor.create_prensor_from_descendant_nodes({
      path.Path([]):
          create_root_node(3),
      path.Path(["foo"]):
          create_optional_leaf_node([0, 1, 2], [9, 8, 7]),
      path.Path(["doc"]):
          create_child_node([0, 1, 1], True),
      path.Path(["doc", "keep_me"]):
          create_optional_leaf_node([0, 1], [False, True]),
      path.Path(["doc", "bar"]):
          create_repeated_leaf_node([0, 1, 1, 2], ["a", "b", "c", "d"]),
      path.Path(["user"]):
          create_child_node([0, 1, 1, 2], True),
      path.Path(["user", "friends"]):
          create_repeated_leaf_node([0, 1, 1, 2, 3], ["a", "b", "c", "d", "e"]),
      path.Path(["foorepeated"]):
          create_repeated_leaf_node([0, 1, 1, 2], [9, 8, 7, 6]),
  })


def create_deep_prensor_schema():
  """Create a schema that is aligned with create_deep_prensor."""
  return text_format.Parse(
      """
      int_domain: {
        name: 'zero_to_ten'
        min: 0
        max: 10
      }
      string_domain: {
        name: 'abcde'
        value: ['a', 'b', 'c', 'd', 'e']
      }
      feature: {
        name: 'foo'
        value_count: {max: 1}
        presence: {min_count: 1}
        int_domain: {min: 0 max: 10}
      }
      feature: {
        name: 'foorepeated'
        presence: {min_count: 1}
        domain: 'zero_to_ten'
      }
      feature: {
        name: 'event'
        struct_domain: {
          feature: {
            name: 'doc'
            struct_domain: {
              feature: {
                name: 'bar'
                domain: 'abcde'
              }
              feature: {
                name: 'keep_me'
                presence: {min_count: 1}
              }
            }
          }
        }
      }
      feature: {
        name: 'user'
        struct_domain: {
          feature: {
            name: 'friends'
            presence: {min_count: 1}
            domain: 'abcde'
          }
        }
      }
      """, schema_pb2.Schema())


def create_deep_prensor():
  """Creates prensor with three layers: root, event, and doc.

  Returns:
    a prensor expression representing:
    {
      foo: 9,
      foorepeated: [9],
      user: [{friends:["a"]}],
      event: [{doc:[{bar:["a"], keep_me:False}]}]
    },
    {
      foo: 8,
      foorepeated: [8,7],
      user: [{friends:["b", "c"]}, {friends:["d"]}],
      event: [{doc:[{bar:["b","c"], keep_me:True},{bar:["d"]}]}]
    },
    {
      foo:7,
      foorepeated: [6],
      user: [{friends:["e"]}],
      event: [{}]
    }
  """
  return prensor.create_prensor_from_descendant_nodes({
      path.Path([]):
          create_root_node(3),
      path.Path(["event"]):
          create_child_node([0, 1, 2], True),
      path.Path(["event", "doc"]):
          create_child_node([0, 1, 1], True),
      path.Path(["event", "doc", "bar"]):
          create_repeated_leaf_node([0, 1, 1, 2], ["a", "b", "c", "d"]),
      path.Path(["event", "doc", "keep_me"]):
          create_optional_leaf_node([0, 1], [False, True]),
      path.Path(["foo"]):
          create_optional_leaf_node([0, 1, 2], [9, 8, 7]),
      path.Path(["foorepeated"]):
          create_repeated_leaf_node([0, 1, 1, 2], [9, 8, 7, 6]),
      path.Path(["user"]):
          create_child_node([0, 1, 1, 2], True),
      path.Path(["user", "friends"]):
          create_repeated_leaf_node([0, 1, 1, 2, 3], ["a", "b", "c", "d", "e"])
  })


def create_four_layer_prensor():
  """Creates prensor with four layers: root, event, doc, and nested_child.

  Returns:
    a prensor expression representing:
    {
      event: {
        doc: {
          nested_child: {
            bar:["a"],
            keep_me:False
          }
        }
      }
    }
    {
      event: {
        doc: {
          nested_child: {
            bar:["b","c"],
            keep_me:True
          },
          nested_child: {
            bar:["d"]
          }
        },
        doc: {
          nested_child: {}
        }
      }
    }
    {event: {}}
  """
  return prensor.create_prensor_from_descendant_nodes({
      path.Path([]):
          create_root_node(3),
      path.Path(["event"]):
          create_child_node([0, 1, 2], True),
      path.Path(["event", "doc"]):
          create_child_node([0, 1, 1], True),
      path.Path(["event", "doc", "nested_child"]):
          create_child_node([0, 1, 1, 2], True),
      path.Path(["event", "doc", "nested_child", "bar"]):
          create_repeated_leaf_node([0, 1, 1, 2], ["a", "b", "c", "d"]),
      path.Path(["event", "doc", "nested_child", "keep_me"]):
          create_optional_leaf_node([0, 1], [False, True])
  })
