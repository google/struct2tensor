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
"""Given an expression, calculate the prensor and source paths."""

from typing import List, NamedTuple, Optional, Sequence, Set, Tuple

from struct2tensor import calculate
from struct2tensor import calculate_options
from struct2tensor import expression
from struct2tensor import path
from struct2tensor import prensor
from struct2tensor.expression_impl import proto
from struct2tensor.proto import query_metadata_pb2
import tensorflow as tf

from google.protobuf import descriptor

# A proto summary represents the tensor of protos and descriptor, along with
# all the paths associated with it.
# tensor is tf.Tensor
# descriptor is descriptor.Descriptor
# paths is List[path.Path]
ProtoRequirements = NamedTuple("ProtoRequirements",
                               [("tensor", tf.Tensor),
                                ("descriptor", descriptor.Descriptor),
                                ("paths", List[path.Path])])


def calculate_prensors_with_source_paths(
    trees: Sequence[expression.Expression],
    options: Optional[calculate_options.Options] = None
) -> Tuple[Sequence[prensor.Prensor], Sequence[ProtoRequirements]]:
  """Returns a list of prensor trees, and proto summaries."""
  prensors, graph = calculate.calculate_prensors_with_graph(
      trees, options=options)
  proto_expressions = [
      x for x in graph.get_expressions_needed() if proto.is_proto_expression(x)
  ]
  summaries = _get_proto_summaries(proto_expressions)
  return prensors, summaries


def _dedup_paths(paths: Sequence[path.Path]) -> List[path.Path]:
  """Deduplicates paths including prefixes.

  Args:
    paths: a list of paths to dedup.

  Returns:
    A new list l where:
      each x in l is in paths
      if x in l, then no prefix of x is in l (except x itself).
      if x in paths, then there exists a y in l where x is a prefix of y.
  """
  # Sort new_paths in increasing length.
  # If x is a strict prefix of y, then x < y.
  new_paths = sorted(paths, key=len)
  new_paths.reverse()
  # If x is a strict prefix of y, now x is after y.
  result = set()  #  type: Set[path.Path]
  result_and_prefixes = set()
  for p in new_paths:
    if p not in result_and_prefixes:
      result.add(p)
      for sub_path in [p.prefix(i) for i in range(len(p) + 1)]:
        result_and_prefixes.add(sub_path)

  return list(result)


def requirements_to_metadata_proto(
    inp: Sequence[ProtoRequirements],
    result: query_metadata_pb2.QueryMetadata) -> None:
  """Populates result with a proto representation of the ProtoRequirements.

  This drops the link to the individual tensor of protos objects, which cannot
  be meaningfully serialized. If there are two protos of the same type, there
  will be two parsed_proto_info submessages with the same message_name.

  Args:
    inp: Sequence[ProtoRequirements]
    result: a proto to be populated.
  """

  for x in inp:
    _requirement_to_parsed_proto_info(x, result.parsed_proto_info.add())  # pytype: disable=wrong-arg-types


def _get_proto_summaries(
    proto_expressions: Sequence[proto.ProtoExpression]
) -> Sequence[ProtoRequirements]:
  """Gets the proto summaries."""
  result = []  # type: List[ProtoRequirements]
  for expr in proto_expressions:

    def get_summary(tensor_of_protos, desc):
      for summary in result:
        if id(summary.tensor) == id(
            tensor_of_protos) and summary.descriptor == desc:
          return summary
      new_summary = ProtoRequirements(
          tensor=tensor_of_protos, descriptor=desc, paths=[])
      result.append(new_summary)
      return new_summary

    tensor_of_protos, desc = expr.get_proto_source()
    get_summary(tensor_of_protos, desc).paths.append(expr.get_path())
  return [
      ProtoRequirements(
          tensor=x.tensor, descriptor=x.descriptor, paths=_dedup_paths(x.paths))
      for x in result
  ]


def _requirement_to_parsed_proto_info(
    inp: ProtoRequirements, result: query_metadata_pb2.ParsedProtoInfo) -> None:
  result.message_name = inp.descriptor.full_name
  for p in inp.paths:
    result.field_paths.add().MergeFrom(p.as_proto())
