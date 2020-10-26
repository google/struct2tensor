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
"""Utilities for tests on proto expressions."""

import tensorflow as tf

from struct2tensor.test import test_pb2
from struct2tensor.expression_impl import proto
from google.protobuf import text_format


def text_to_tensor(text_list, example_proto_clz):
  as_protos = [text_format.Parse(x, example_proto_clz()) for x in text_list]
  serialized = [x.SerializeToString() for x in as_protos]
  return tf.constant(serialized)


def text_to_expression(text_list, example_proto_clz):
  """Create an expression from a list of text format protos."""
  return proto.create_expression_from_proto(
      text_to_tensor(text_list, example_proto_clz),
      example_proto_clz().DESCRIPTOR)


def _get_expression_from_session_empty_user_info():
  r"""Run create_root_prensor on a very deep tree.

  In addition, the user_info is empty.

  ```
               ------*-----------------
              /                        \
        ---session0----             session1
       /      \        \             /      \
    event0  event1   event2      event3   event4
    /    \    |  \     |  \       /       /  |  \
  act0 act1 act2 act3 act4 act5 act6   act7 act8 act9
    |   |     |        |    |     |      |    |   |
    a   b     c        e    f     g      h    i   j

  ```

  Returns:
    A RootPrensor with the above structure.
  """
  return text_to_expression([
      """
          event:{
            action:{
              doc_id:"a"
            }
            action:{
              doc_id:"b"
            }
            event_id: "A"
          }
          event:{
            action:{
              doc_id:"c"
            }
            action:{
            }
            event_id: "B"
          }
          event:{
            action:{
              doc_id:"e"
            }
            action:{
              doc_id:"f"
            }
            event_id: "C"
          }""", """
          event:{
            action:{
              doc_id:"g"
            }
          }
          event:{
            event_id: "D"
            action:{
              doc_id:"h"
            }
            action:{
              doc_id:"i"
            }
            action:{
              doc_id:"j"
            }
          }"""
  ], test_pb2.Session)
