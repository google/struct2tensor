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
r"""Benchmarks for struct2tensor.


"""

from absl.testing import parameterized
import struct2tensor as s2t
from struct2tensor.benchmarks import benchmark_pb2
from struct2tensor.benchmarks import struct2tensor_benchmark_util
import tensorflow as tf


class ProjectBenchmarks(struct2tensor_benchmark_util.ProtoDataBenchmarks):
  """Benchmarks for projecting fields."""

  # pylint: disable=g-complex-comprehension
  @parameterized.named_parameters(*[
      dict(
          testcase_name="project_1_deep_int_fields",
          fn_name="project_1_deep_int_fields",
          fn_args=[
              benchmark_pb2.DeepProto.DESCRIPTOR,
              [s2t.path.Path(["int_values_1"])]
          ],
          proto_list_key="deep_protos",
      ),
      dict(
          testcase_name="project_2_deep_int_fields",
          fn_name="project_2_deep_int_fields",
          fn_args=[
              benchmark_pb2.DeepProto.DESCRIPTOR,
              [s2t.path.Path(["child_1", "int_values_2"])]
          ],
          proto_list_key="deep_protos",
      ),
      dict(
          testcase_name="project_3_deep_int_fields",
          fn_name="project_3_deep_int_fields",
          fn_args=[
              benchmark_pb2.DeepProto.DESCRIPTOR,
              [s2t.path.Path(["child_1", "child_2", "int_values_3"])]
          ],
          proto_list_key="deep_protos",
      ),
      dict(
          testcase_name="project_4_deep_int_fields",
          fn_name="project_4_deep_int_fields",
          fn_args=[
              benchmark_pb2.DeepProto.DESCRIPTOR,
              [
                  s2t.path.Path(
                      ["child_1", "child_2", "child_3", "int_values_4"])
              ]
          ],
          proto_list_key="deep_protos",
      ),
      dict(
          testcase_name="project_5_deep_int_fields",
          fn_name="project_5_deep_int_fields",
          fn_args=[
              benchmark_pb2.DeepProto.DESCRIPTOR,
              [
                  s2t.path.Path([
                      "child_1", "child_2", "child_3", "child_4", "int_values_5"
                  ])
              ]
          ],
          proto_list_key="deep_protos",
      ),
      dict(
          testcase_name="project_1_flat_int_fields",
          fn_name="project_1_flat_int_fields",
          fn_args=[
              benchmark_pb2.FlatProto.DESCRIPTOR,
              [s2t.path.Path(["int_values_1"])]
          ],
          proto_list_key="flat_protos",
      ),
      dict(
          testcase_name="project_2_flat_int_fields",
          fn_name="project_2_flat_int_fields",
          fn_args=[
              benchmark_pb2.FlatProto.DESCRIPTOR,
              [
                  s2t.path.Path(["int_values_1"]),
                  s2t.path.Path(["int_values_2"])
              ]
          ],
          proto_list_key="flat_protos",
      ),
      dict(
          testcase_name="project_3_flat_int_fields",
          fn_name="project_3_flat_int_fields",
          fn_args=[
              benchmark_pb2.FlatProto.DESCRIPTOR,
              [
                  s2t.path.Path(["int_values_1"]),
                  s2t.path.Path(["int_values_2"]),
                  s2t.path.Path(["int_values_3"])
              ]
          ],
          proto_list_key="flat_protos",
      ),
      dict(
          testcase_name="project_4_flat_int_fields",
          fn_name="project_4_flat_int_fields",
          fn_args=[
              benchmark_pb2.FlatProto.DESCRIPTOR,
              [
                  s2t.path.Path(["int_values_1"]),
                  s2t.path.Path(["int_values_2"]),
                  s2t.path.Path(["int_values_3"]),
                  s2t.path.Path(["int_values_4"])
              ]
          ],
          proto_list_key="flat_protos",
      ),
      dict(
          testcase_name="project_5_flat_int_fields",
          fn_name="project_5_flat_int_fields",
          fn_args=[
              benchmark_pb2.FlatProto.DESCRIPTOR,
              [
                  s2t.path.Path(["int_values_1"]),
                  s2t.path.Path(["int_values_2"]),
                  s2t.path.Path(["int_values_3"]),
                  s2t.path.Path(["int_values_4"]),
                  s2t.path.Path(["int_values_5"])
              ]
          ],
          proto_list_key="flat_protos",
      ),
      dict(
          testcase_name="project_1_deep_float_fields",
          fn_name="project_1_deep_float_fields",
          fn_args=[
              benchmark_pb2.DeepProto.DESCRIPTOR,
              [s2t.path.Path(["float_values_1"])]
          ],
          proto_list_key="deep_protos",
      ),
      dict(
          testcase_name="project_2_deep_float_fields",
          fn_name="project_2_deep_float_fields",
          fn_args=[
              benchmark_pb2.DeepProto.DESCRIPTOR,
              [s2t.path.Path(["child_1", "float_values_2"])]
          ],
          proto_list_key="deep_protos",
      ),
      dict(
          testcase_name="project_3_deep_float_fields",
          fn_name="project_3_deep_float_fields",
          fn_args=[
              benchmark_pb2.DeepProto.DESCRIPTOR,
              [s2t.path.Path(["child_1", "child_2", "float_values_3"])]
          ],
          proto_list_key="deep_protos",
      ),
      dict(
          testcase_name="project_4_deep_float_fields",
          fn_name="project_4_deep_float_fields",
          fn_args=[
              benchmark_pb2.DeepProto.DESCRIPTOR,
              [
                  s2t.path.Path(
                      ["child_1", "child_2", "child_3", "float_values_4"])
              ]
          ],
          proto_list_key="deep_protos",
      ),
      dict(
          testcase_name="project_5_deep_float_fields",
          fn_name="project_5_deep_float_fields",
          fn_args=[
              benchmark_pb2.DeepProto.DESCRIPTOR,
              [
                  s2t.path.Path([
                      "child_1", "child_2", "child_3", "child_4",
                      "float_values_5"
                  ])
              ]
          ],
          proto_list_key="deep_protos",
      ),
      dict(
          testcase_name="project_1_flat_float_fields",
          fn_name="project_1_flat_float_fields",
          fn_args=[
              benchmark_pb2.FlatProto.DESCRIPTOR,
              [s2t.path.Path(["float_values_1"])]
          ],
          proto_list_key="flat_protos",
      ),
      dict(
          testcase_name="project_2_flat_float_fields",
          fn_name="project_2_flat_float_fields",
          fn_args=[
              benchmark_pb2.FlatProto.DESCRIPTOR,
              [
                  s2t.path.Path(["float_values_1"]),
                  s2t.path.Path(["float_values_2"])
              ]
          ],
          proto_list_key="flat_protos",
      ),
      dict(
          testcase_name="project_3_flat_float_fields",
          fn_name="project_3_flat_float_fields",
          fn_args=[
              benchmark_pb2.FlatProto.DESCRIPTOR,
              [
                  s2t.path.Path(["float_values_1"]),
                  s2t.path.Path(["float_values_2"]),
                  s2t.path.Path(["float_values_3"])
              ]
          ],
          proto_list_key="flat_protos",
      ),
      dict(
          testcase_name="project_4_flat_float_fields",
          fn_name="project_4_flat_float_fields",
          fn_args=[
              benchmark_pb2.FlatProto.DESCRIPTOR,
              [
                  s2t.path.Path(["float_values_1"]),
                  s2t.path.Path(["float_values_2"]),
                  s2t.path.Path(["float_values_3"]),
                  s2t.path.Path(["float_values_4"])
              ]
          ],
          proto_list_key="flat_protos",
      ),
      dict(
          testcase_name="project_5_flat_float_fields",
          fn_name="project_5_flat_float_fields",
          fn_args=[
              benchmark_pb2.FlatProto.DESCRIPTOR,
              [
                  s2t.path.Path(["float_values_1"]),
                  s2t.path.Path(["float_values_2"]),
                  s2t.path.Path(["float_values_3"]),
                  s2t.path.Path(["float_values_4"]),
                  s2t.path.Path(["float_values_5"])
              ]
          ],
          proto_list_key="flat_protos",
      ),
      dict(
          testcase_name="project_1_deep_bytes_fields",
          fn_name="project_1_deep_bytes_fields",
          fn_args=[
              benchmark_pb2.DeepProto.DESCRIPTOR,
              [s2t.path.Path(["bytes_values_1"])]
          ],
          proto_list_key="deep_protos",
      ),
      dict(
          testcase_name="project_2_deep_bytes_fields",
          fn_name="project_2_deep_bytes_fields",
          fn_args=[
              benchmark_pb2.DeepProto.DESCRIPTOR,
              [s2t.path.Path(["child_1", "bytes_values_2"])]
          ],
          proto_list_key="deep_protos",
      ),
      dict(
          testcase_name="project_3_deep_bytes_fields",
          fn_name="project_3_deep_bytes_fields",
          fn_args=[
              benchmark_pb2.DeepProto.DESCRIPTOR,
              [s2t.path.Path(["child_1", "child_2", "bytes_values_3"])]
          ],
          proto_list_key="deep_protos",
      ),
      dict(
          testcase_name="project_4_deep_bytes_fields",
          fn_name="project_4_deep_bytes_fields",
          fn_args=[
              benchmark_pb2.DeepProto.DESCRIPTOR,
              [
                  s2t.path.Path(
                      ["child_1", "child_2", "child_3", "bytes_values_4"])
              ]
          ],
          proto_list_key="deep_protos",
      ),
      dict(
          testcase_name="project_5_deep_bytes_fields",
          fn_name="project_5_deep_bytes_fields",
          fn_args=[
              benchmark_pb2.DeepProto.DESCRIPTOR,
              [
                  s2t.path.Path([
                      "child_1", "child_2", "child_3", "child_4",
                      "bytes_values_5"
                  ])
              ]
          ],
          proto_list_key="deep_protos",
      ),
      dict(
          testcase_name="project_1_flat_bytes_fields",
          fn_name="project_1_flat_bytes_fields",
          fn_args=[
              benchmark_pb2.FlatProto.DESCRIPTOR,
              [s2t.path.Path(["bytes_values_1"])]
          ],
          proto_list_key="flat_protos",
      ),
      dict(
          testcase_name="project_2_flat_bytes_fields",
          fn_name="project_2_flat_bytes_fields",
          fn_args=[
              benchmark_pb2.FlatProto.DESCRIPTOR,
              [
                  s2t.path.Path(["bytes_values_1"]),
                  s2t.path.Path(["bytes_values_2"])
              ]
          ],
          proto_list_key="flat_protos",
      ),
      dict(
          testcase_name="project_3_flat_bytes_fields",
          fn_name="project_3_flat_bytes_fields",
          fn_args=[
              benchmark_pb2.FlatProto.DESCRIPTOR,
              [
                  s2t.path.Path(["bytes_values_1"]),
                  s2t.path.Path(["bytes_values_2"]),
                  s2t.path.Path(["bytes_values_3"])
              ]
          ],
          proto_list_key="flat_protos",
      ),
      dict(
          testcase_name="project_4_flat_bytes_fields",
          fn_name="project_4_flat_bytes_fields",
          fn_args=[
              benchmark_pb2.FlatProto.DESCRIPTOR,
              [
                  s2t.path.Path(["bytes_values_1"]),
                  s2t.path.Path(["bytes_values_2"]),
                  s2t.path.Path(["bytes_values_3"]),
                  s2t.path.Path(["bytes_values_4"])
              ]
          ],
          proto_list_key="flat_protos",
      ),
      dict(
          testcase_name="project_5_flat_bytes_fields",
          fn_name="project_5_flat_bytes_fields",
          fn_args=[
              benchmark_pb2.FlatProto.DESCRIPTOR,
              [
                  s2t.path.Path(["bytes_values_1"]),
                  s2t.path.Path(["bytes_values_2"]),
                  s2t.path.Path(["bytes_values_3"]),
                  s2t.path.Path(["bytes_values_4"]),
                  s2t.path.Path(["bytes_values_5"])
              ]
          ],
          proto_list_key="flat_protos",
      ),
  ])
  # pylint: enable=g-complex-comprehension
  def test_project(self, fn_name, fn_args, proto_list_key):
    self.run_benchmarks(fn_name, _get_project_fn, fn_args, proto_list_key)


class PromoteBenchmarks(struct2tensor_benchmark_util.ProtoDataBenchmarks):
  """Benchmarks for promoting fields."""

  @parameterized.named_parameters(*[
      dict(
          testcase_name="promote_1",
          fn_name="promote_1",
          fn_args=[
              benchmark_pb2.DeepProto.DESCRIPTOR,
              s2t.path.Path(["child_1", "int_values_2"])
          ],
          proto_list_key="deep_protos"),
      dict(
          testcase_name="promote_2",
          fn_name="promote_2",
          fn_args=[
              benchmark_pb2.DeepProto.DESCRIPTOR,
              s2t.path.Path(["child_1", "child_2", "int_values_3"])
          ],
          proto_list_key="deep_protos"),
      dict(
          testcase_name="promote_3",
          fn_name="promote_3",
          fn_args=[
              benchmark_pb2.DeepProto.DESCRIPTOR,
              s2t.path.Path(["child_1", "child_2", "child_3", "int_values_4"])
          ],
          proto_list_key="deep_protos"),
      dict(
          testcase_name="promote_4",
          fn_name="promote_4",
          fn_args=[
              benchmark_pb2.DeepProto.DESCRIPTOR,
              s2t.path.Path(
                  ["child_1", "child_2", "child_3", "child_4", "int_values_5"])
          ],
          proto_list_key="deep_protos"),
  ])
  def test_promote(self, fn_name, fn_args, proto_list_key):
    self.run_benchmarks(fn_name, _get_promote_fn, fn_args, proto_list_key)


class BroadcastBenchmarks(struct2tensor_benchmark_util.ProtoDataBenchmarks):
  """Benchmarks for broadcasting fields."""

  @parameterized.named_parameters(*[
      dict(
          testcase_name="broadcast_2",
          fn_name="broadcast_2",
          fn_args=[
              benchmark_pb2.DeepProto.DESCRIPTOR,
              s2t.path.Path(["int_values_1"]), "child_1"
          ],
          proto_list_key="deep_protos"),
      dict(
          testcase_name="broadcast_3",
          fn_name="broadcast_3",
          fn_args=[
              benchmark_pb2.DeepProto.DESCRIPTOR,
              s2t.path.Path(["child_1", "int_values_2"]), "child_2"
          ],
          proto_list_key="deep_protos"),
      dict(
          testcase_name="broadcast_4",
          fn_name="broadcast_4",
          fn_args=[
              benchmark_pb2.DeepProto.DESCRIPTOR,
              s2t.path.Path(["child_1", "child_2", "int_values_3"]), "child_3"
          ],
          proto_list_key="deep_protos"),
      dict(
          testcase_name="broadcast_5",
          fn_name="broadcast_5",
          fn_args=[
              benchmark_pb2.DeepProto.DESCRIPTOR,
              s2t.path.Path(["child_1", "child_2", "child_3", "int_values_4"]),
              "child_4"
          ],
          proto_list_key="deep_protos"),
  ])
  def test_broadcast(self, fn_name, fn_args, proto_list_key):
    self.run_benchmarks(fn_name, _get_broadcast_fn, fn_args, proto_list_key)


class RerootBenchmarks(struct2tensor_benchmark_util.ProtoDataBenchmarks):
  """Benchmarks for rerooting fields."""

  @parameterized.named_parameters(*[
      dict(
          testcase_name="reroot_1",
          fn_name="reroot_1",
          fn_args=[
              benchmark_pb2.DeepProto.DESCRIPTOR,
              s2t.path.Path(["child_1"]), [s2t.path.Path(["int_values_2"])]
          ],
          proto_list_key="deep_protos"),
      dict(
          testcase_name="reroot_2",
          fn_name="reroot_2",
          fn_args=[
              benchmark_pb2.DeepProto.DESCRIPTOR,
              s2t.path.Path(["child_1", "child_2"]),
              [s2t.path.Path(["int_values_3"])]
          ],
          proto_list_key="deep_protos"),
      dict(
          testcase_name="reroot_3",
          fn_name="reroot_3",
          fn_args=[
              benchmark_pb2.DeepProto.DESCRIPTOR,
              s2t.path.Path(["child_1", "child_2", "child_3"]),
              [s2t.path.Path(["int_values_4"])]
          ],
          proto_list_key="deep_protos"),
      dict(
          testcase_name="reroot_4",
          fn_name="reroot_4",
          fn_args=[
              benchmark_pb2.DeepProto.DESCRIPTOR,
              s2t.path.Path(["child_1", "child_2", "child_3", "child_4"]),
              [s2t.path.Path(["int_values_5"])]
          ],
          proto_list_key="deep_protos"),
  ])
  def test_reroot(self, fn_name, fn_args, proto_list_key):
    self.run_benchmarks(fn_name, _get_reroot_fn, fn_args, proto_list_key)


class PrensorToTensorBenchmarks(
    struct2tensor_benchmark_util.ProtoDataBenchmarks):
  """Benchmarks for converting prensor to tensors."""

  # pylint: disable=g-complex-comprehension
  @parameterized.named_parameters(*[
      dict(
          testcase_name="flat_to_dense_{}_features".format(n),
          fn_name="flat_to_dense_{}_features".format(n),
          fn_args=[
              benchmark_pb2.FlatProto.DESCRIPTOR,
              [
                  s2t.path.Path(["int_values_{}".format(i + 1)])
                  for i in range(n)
              ]
          ],
          proto_list_key="flat_protos")
      for n in [1, 2, 3, 4, 5]
  ] + [
      dict(
          testcase_name="flat_to_dense_1_feature_100_values",
          fn_name="flat_to_dense_1_feature_100_values",
          fn_args=[
              benchmark_pb2.FlatProto.DESCRIPTOR,
              [
                  s2t.path.Path(["int_values_1"])
              ]
          ],
          proto_list_key="flat_protos_1_feature_100_values"),
      dict(
          testcase_name="flat_to_dense_100_features",
          fn_name="flat_to_dense_100_features",
          fn_args=[
              benchmark_pb2.FlatProto100.DESCRIPTOR,
              [
                  s2t.path.Path(["int_values_{}".format(i)])
                  for i in range(1, 101)
              ]
          ],
          proto_list_key="flat_protos_100_features"),
      dict(
          testcase_name="flat_to_dense_100_features_100_values",
          fn_name="flat_to_dense_100_features_100_values",
          fn_args=[
              benchmark_pb2.FlatProto100.DESCRIPTOR,
              [
                  s2t.path.Path(["int_values_{}".format(i)])
                  for i in range(1, 101)
              ]
          ],
          proto_list_key="flat_protos_100_features_100_values"),
      dict(
          testcase_name="deep_to_dense_1",
          fn_name="deep_to_dense_1",
          fn_args=[
              benchmark_pb2.DeepProto.DESCRIPTOR,
              [s2t.path.Path(["int_values_1"])]
          ],
          proto_list_key="deep_protos"),
      dict(
          testcase_name="deep_to_dense_2",
          fn_name="deep_to_dense_2",
          fn_args=[
              benchmark_pb2.DeepProto.DESCRIPTOR,
              [s2t.path.Path(["child_1", "int_values_2"])]
          ],
          proto_list_key="deep_protos"),
      dict(
          testcase_name="deep_to_dense_3",
          fn_name="deep_to_dense_3",
          fn_args=[
              benchmark_pb2.DeepProto.DESCRIPTOR,
              [s2t.path.Path(["child_1", "child_2", "int_values_3"])]
          ],
          proto_list_key="deep_protos"),
      dict(
          testcase_name="deep_to_dense_4",
          fn_name="deep_to_dense_4",
          fn_args=[
              benchmark_pb2.DeepProto.DESCRIPTOR,
              [
                  s2t.path.Path(
                      ["child_1", "child_2", "child_3", "int_values_4"])
              ]
          ],
          proto_list_key="deep_protos"),
      dict(
          testcase_name="deep_to_dense_5",
          fn_name="deep_to_dense_5",
          fn_args=[
              benchmark_pb2.DeepProto.DESCRIPTOR,
              [
                  s2t.path.Path([
                      "child_1", "child_2", "child_3", "child_4", "int_values_5"
                  ])
              ]
          ],
          proto_list_key="deep_protos"),
  ])
  # pylint: enable=g-complex-comprehension
  def test_to_dense(self, fn_name, fn_args, proto_list_key):
    """This benchmark converts prensors to dense tensors."""
    self.run_benchmarks(fn_name, _get_prensor_to_dense_tensor_fn, fn_args,
                        proto_list_key)

  # pylint: disable=g-complex-comprehension
  @parameterized.named_parameters(*[
      dict(
          testcase_name="flat_to_ragged_{}_features".format(n),
          fn_name="flat_to_ragged_{}_features".format(n),
          fn_args=[
              benchmark_pb2.FlatProto.DESCRIPTOR,
              [
                  s2t.path.Path(["int_values_{}".format(i + 1)])
                  for i in range(n)
              ]
          ],
          proto_list_key="flat_protos")
      for n in [1, 2, 3, 4, 5]
  ] + [
      dict(
          testcase_name="flat_to_ragged_1_feature_100_values",
          fn_name="flat_to_ragged_1_feature_100_values",
          fn_args=[
              benchmark_pb2.FlatProto.DESCRIPTOR,
              [
                  s2t.path.Path(["int_values_1"])
              ]
          ],
          proto_list_key="flat_protos_1_feature_100_values"),
      dict(
          testcase_name="flat_to_ragged_100_features",
          fn_name="flat_to_ragged_100_features",
          fn_args=[
              benchmark_pb2.FlatProto100.DESCRIPTOR,
              [
                  s2t.path.Path(["int_values_{}".format(i)])
                  for i in range(1, 101)
              ]
          ],
          proto_list_key="flat_protos_100_features"),
      dict(
          testcase_name="flat_to_ragged_100_features_100_values",
          fn_name="flat_to_ragged_100_features_100_values",
          fn_args=[
              benchmark_pb2.FlatProto100.DESCRIPTOR,
              [
                  s2t.path.Path(["int_values_{}".format(i)])
                  for i in range(1, 101)
              ]
          ],
          proto_list_key="flat_protos_100_features_100_values"),
      dict(
          testcase_name="deep_to_ragged_1",
          fn_name="deep_to_ragged_1",
          fn_args=[
              benchmark_pb2.DeepProto.DESCRIPTOR,
              [s2t.path.Path(["int_values_1"])]
          ],
          proto_list_key="deep_protos"),
      dict(
          testcase_name="deep_to_ragged_2",
          fn_name="deep_to_ragged_2",
          fn_args=[
              benchmark_pb2.DeepProto.DESCRIPTOR,
              [s2t.path.Path(["child_1", "int_values_2"])]
          ],
          proto_list_key="deep_protos"),
      dict(
          testcase_name="deep_to_ragged_3",
          fn_name="deep_to_ragged_3",
          fn_args=[
              benchmark_pb2.DeepProto.DESCRIPTOR,
              [s2t.path.Path(["child_1", "child_2", "int_values_3"])]
          ],
          proto_list_key="deep_protos"),
      dict(
          testcase_name="deep_to_ragged_4",
          fn_name="deep_to_ragged_4",
          fn_args=[
              benchmark_pb2.DeepProto.DESCRIPTOR,
              [
                  s2t.path.Path(
                      ["child_1", "child_2", "child_3", "int_values_4"])
              ]
          ],
          proto_list_key="deep_protos"),
      dict(
          testcase_name="deep_to_ragged_5",
          fn_name="deep_to_ragged_5",
          fn_args=[
              benchmark_pb2.DeepProto.DESCRIPTOR,
              [
                  s2t.path.Path([
                      "child_1", "child_2", "child_3", "child_4", "int_values_5"
                  ])
              ]
          ],
          proto_list_key="deep_protos"),
  ])
  # pylint: enable=g-complex-comprehension
  def test_to_ragged(self, fn_name, fn_args, proto_list_key):
    """This benchmark converts prensors to ragged tensors."""
    self.run_benchmarks(fn_name, _get_prensor_to_ragged_tensor_fn, fn_args,
                        proto_list_key)

  # pylint: disable=g-complex-comprehension
  @parameterized.named_parameters(*[
      dict(
          testcase_name="flat_to_sparse_{}_features".format(n),
          fn_name="flat_to_sparse_{}_features".format(n),
          fn_args=[
              benchmark_pb2.FlatProto.DESCRIPTOR,
              [
                  s2t.path.Path(["int_values_{}".format(i + 1)])
                  for i in range(n)
              ]
          ],
          proto_list_key="flat_protos")
      for n in [1, 2, 3, 4, 5]
  ] + [
      dict(
          testcase_name="flat_to_sparse_1_feature_100_values",
          fn_name="flat_to_sparse_1_feature_100_values",
          fn_args=[
              benchmark_pb2.FlatProto.DESCRIPTOR,
              [
                  s2t.path.Path(["int_values_1"])
              ]
          ],
          proto_list_key="flat_protos_1_feature_100_values"),
      dict(
          testcase_name="flat_to_sparse_100_features",
          fn_name="flat_to_sparse_100_features",
          fn_args=[
              benchmark_pb2.FlatProto100.DESCRIPTOR,
              [
                  s2t.path.Path(["int_values_{}".format(i)])
                  for i in range(1, 101)
              ]
          ],
          proto_list_key="flat_protos_100_features"),
      dict(
          testcase_name="flat_to_sparse_100_features_100_values",
          fn_name="flat_to_sparse_100_features_100_values",
          fn_args=[
              benchmark_pb2.FlatProto100.DESCRIPTOR,
              [
                  s2t.path.Path(["int_values_{}".format(i)])
                  for i in range(1, 101)
              ]
          ],
          proto_list_key="flat_protos_100_features_100_values"),
      dict(
          testcase_name="deep_to_sparse_1",
          fn_name="deep_to_sparse_1",
          fn_args=[
              benchmark_pb2.DeepProto.DESCRIPTOR,
              [s2t.path.Path(["int_values_1"])]
          ],
          proto_list_key="deep_protos"),
      dict(
          testcase_name="deep_to_sparse_2",
          fn_name="deep_to_sparse_2",
          fn_args=[
              benchmark_pb2.DeepProto.DESCRIPTOR,
              [s2t.path.Path(["child_1", "int_values_2"])]
          ],
          proto_list_key="deep_protos"),
      dict(
          testcase_name="deep_to_sparse_3",
          fn_name="deep_to_sparse_3",
          fn_args=[
              benchmark_pb2.DeepProto.DESCRIPTOR,
              [s2t.path.Path(["child_1", "child_2", "int_values_3"])]
          ],
          proto_list_key="deep_protos"),
      dict(
          testcase_name="deep_to_sparse_4",
          fn_name="deep_to_sparse_4",
          fn_args=[
              benchmark_pb2.DeepProto.DESCRIPTOR,
              [
                  s2t.path.Path(
                      ["child_1", "child_2", "child_3", "int_values_4"])
              ]
          ],
          proto_list_key="deep_protos"),
      dict(
          testcase_name="deep_to_sparse_5",
          fn_name="deep_to_sparse_5",
          fn_args=[
              benchmark_pb2.DeepProto.DESCRIPTOR,
              [
                  s2t.path.Path([
                      "child_1", "child_2", "child_3", "child_4", "int_values_5"
                  ])
              ]
          ],
          proto_list_key="deep_protos"),
  ])
  # pylint: enable=g-complex-comprehension
  def test_to_sparse(self, fn_name, fn_args, proto_list_key):
    """This benchmark converts prensors to sparse tensors."""
    self.run_benchmarks(fn_name, _get_prensor_to_sparse_tensor_fn, fn_args,
                        proto_list_key)


class TfExampleBenchmarks(struct2tensor_benchmark_util.ProtoDataBenchmarks):
  """Benchmarks for converting tf.example to tensors."""

  # pylint: disable=g-complex-comprehension
  @parameterized.named_parameters(*[
      dict(
          testcase_name="to_fixed_len_feature_{}_features_1_value".format(n),
          fn_name="tf_example_to_fixed_len_feature_{}".format(n),
          fn_args=[{
              "int_values_1": tf.io.FixedLenFeature(shape=[1], dtype=tf.int64)
          }],
          proto_list_key="tf_examples") for n in [1, 2, 3, 4, 5]
  ] + [
      dict(
          testcase_name="to_fixed_len_feature_1_feature_100_value",
          fn_name="tf_example_to_fixed_len_feature_1_feature_100_value",
          fn_args=[{
              "int_values_1": tf.io.FixedLenFeature(
                  shape=[100], dtype=tf.int64)
          }],
          proto_list_key="tf_examples_1_feature_100_values"),
      dict(
          testcase_name="to_fixed_len_feature_100_features_1_value",
          fn_name="tf_example_to_fixed_len_feature_100_features_1_value",
          fn_args=[{
              "int_values_{}".format(i): tf.io.FixedLenFeature(
                  shape=[1], dtype=tf.int64) for i in range(1, 101)
          }],
          proto_list_key="tf_examples_100_features"),
      dict(
          testcase_name="to_fixed_len_feature_100_features_100_values",
          fn_name="tf_example_to_fixed_len_feature_100_features_100_values",
          fn_args=[{
              "int_values_{}".format(i): tf.io.FixedLenFeature(
                  shape=[100], dtype=tf.int64) for i in range(1, 101)
          }],
          proto_list_key="tf_examples_100_features_100_values")
  ])
  # pylint: enable=g-complex-comprehension
  def test_parse_tf_example(self, fn_name, fn_args, proto_list_key):
    self.run_benchmarks(fn_name, _get_parse_tf_example_fn, fn_args,
                        proto_list_key)

  @parameterized.named_parameters(*[
      dict(
          testcase_name="to_var_len_feature_{}_features_1_value".format(n),
          fn_name="tf_example_to_var_len_feature_{}".format(n),
          fn_args=[{
              "int_values_1": tf.io.VarLenFeature(dtype=tf.int64)
          }],
          proto_list_key="tf_examples") for n in [1, 2, 3, 4, 5]
  ] + [
      dict(
          testcase_name="to_var_len_feature_1_feature_100_value",
          fn_name="tf_example_to_var_len_feature_1_feature_100_value",
          fn_args=[{
              "int_values_1": tf.io.VarLenFeature(dtype=tf.int64)
          }],
          proto_list_key="tf_examples_1_feature_100_values"),
      dict(
          testcase_name="to_var_len_feature_100_features_1_value",
          fn_name="tf_example_to_var_len_feature_100_features_1_value",
          fn_args=[{
              "int_values_{}".format(i): tf.io.VarLenFeature(dtype=tf.int64)
              for i in range(1, 101)
          }],
          proto_list_key="tf_examples_100_features"),
      dict(
          testcase_name="to_var_len_feature_100_features_100_values",
          fn_name="tf_example_to_var_len_feature_100_features_100_values",
          fn_args=[{
              "int_values_{}".format(i): tf.io.VarLenFeature(dtype=tf.int64)
              for i in range(1, 101)
          }],
          proto_list_key="tf_examples_100_features_100_values")
  ])
  def test_parse_tf_example_to_sparse(self, fn_name, fn_args, proto_list_key):
    self.run_benchmarks(fn_name, _get_parse_tf_example_to_sparse_fn, fn_args,
                        proto_list_key)

  @parameterized.named_parameters(*[
      dict(
          testcase_name="to_ragged_feature_{}_features_1_value".format(n),
          fn_name="tf_example_to_ragged_feature_{}".format(n),
          fn_args=[{
              "int_values_1":
                  tf.io.RaggedFeature(value_key="int_values_1", dtype=tf.int64)
          }],
          proto_list_key="tf_examples") for n in [1, 2, 3, 4, 5]
  ] + [
      dict(
          testcase_name="to_ragged_feature_1_feature_100_value",
          fn_name="tf_example_to_ragged_feature_1_feature_100_value",
          fn_args=[{
              "int_values_1":
                  tf.io.RaggedFeature(value_key="int_values_1", dtype=tf.int64)
          }],
          proto_list_key="tf_examples_1_feature_100_values"),
      dict(
          testcase_name="to_ragged_feature_100_features_1_value",
          fn_name="tf_example_to_ragged_feature_100_features_1_value",
          fn_args=[{
              "int_values_{}".format(i): tf.io.RaggedFeature(
                  value_key="int_values_{}".format(i), dtype=tf.int64)
              for i in range(1, 101)
          }],
          proto_list_key="tf_examples_100_features"),
      dict(
          testcase_name="to_ragged_feature_100_features_100_values",
          fn_name="tf_example_to_ragged_feature_100_features_100_values",
          fn_args=[{
              "int_values_{}".format(i): tf.io.RaggedFeature(
                  value_key="int_values_{}".format(i), dtype=tf.int64)
              for i in range(1, 101)
          }],
          proto_list_key="tf_examples_100_features_100_values")
  ])
  def test_parse_tf_example_to_ragged(self, fn_name, fn_args, proto_list_key):
    self.run_benchmarks(fn_name, _get_parse_tf_example_fn, fn_args,
                        proto_list_key)


def _get_project_fn(session, proto_descriptor, path):
  """Returns a callable that projects the path, and creates ragged tensors."""
  protos = tf.compat.v1.placeholder(dtype=tf.string, shape=(None,))
  expr = s2t.expression_impl.proto.create_expression_from_proto(
      protos, proto_descriptor)
  expr = expr.project(path)
  [prensor] = s2t.calculate.calculate_prensors(
      [expr], options=s2t.calculate_options.get_options_with_minimal_checks())
  rt = prensor.get_ragged_tensors()
  with tf.control_dependencies(rt.values()):
    x = tf.constant(1)
  return session.make_callable(x, feed_list=[protos])


def _get_promote_fn(session, proto_descriptor, path_to_promote):
  """Returns a callable that promotes a field, and creates ragged tensors."""
  protos = tf.compat.v1.placeholder(dtype=tf.string, shape=(None,))
  expr = s2t.expression_impl.proto.create_expression_from_proto(
      protos, proto_descriptor).promote(path_to_promote, "new_child").project(
          [path_to_promote.prefix(-2).concat(s2t.path.Path(["new_child"]))])
  [prensor] = s2t.calculate.calculate_prensors(
      [expr], options=s2t.calculate_options.get_options_with_minimal_checks())
  rt = prensor.get_ragged_tensors()
  with tf.control_dependencies(rt.values()):
    x = tf.constant(1)
  return session.make_callable(x, feed_list=[protos])


def _get_broadcast_fn(session, proto_descriptor, path_to_broadcast, sibling):
  """Returns a callable that broadcasts a field, and creates ragged tensors."""
  protos = tf.compat.v1.placeholder(dtype=tf.string, shape=(None,))
  expr = s2t.expression_impl.proto.create_expression_from_proto(
      protos,
      proto_descriptor).broadcast(path_to_broadcast, sibling,
                                  "new_child").project([
                                      path_to_broadcast.get_parent().concat(
                                          s2t.path.Path([sibling, "new_child"]))
                                  ])
  [prensor] = s2t.calculate.calculate_prensors(
      [expr], options=s2t.calculate_options.get_options_with_minimal_checks())
  rt = prensor.get_ragged_tensors()
  with tf.control_dependencies(rt.values()):
    x = tf.constant(1)
  return session.make_callable(x, feed_list=[protos])


def _get_reroot_fn(session, proto_descriptor, path_to_reroot, path_to_project):
  """Returns a callable that reroots a field, and creates ragged tensors."""
  protos = tf.compat.v1.placeholder(dtype=tf.string, shape=(None,))
  expr = s2t.expression_impl.proto.create_expression_from_proto(
      protos, proto_descriptor).reroot(path_to_reroot).project(path_to_project)
  [prensor] = s2t.calculate.calculate_prensors(
      [expr], options=s2t.calculate_options.get_options_with_minimal_checks())
  rt = prensor.get_ragged_tensors()
  with tf.control_dependencies(rt.values()):
    x = tf.constant(1)
  return session.make_callable(x, feed_list=[protos])


def _get_prensor_to_dense_tensor_fn(session, proto_descriptor, path):
  """Returns a callable that projects a field, and creates ragged tensors."""
  protos = tf.compat.v1.placeholder(dtype=tf.string, shape=(None,))
  expr = s2t.expression_impl.proto.create_expression_from_proto(
      protos, proto_descriptor)
  expr = expr.project(path)
  [prensor] = s2t.calculate.calculate_prensors(
      [expr], options=s2t.calculate_options.get_options_with_minimal_checks())
  ragged_tensors = prensor.get_ragged_tensors()
  dt = [rt.to_tensor() for rt in ragged_tensors.values()]
  with tf.control_dependencies(dt):
    x = tf.constant(1)
  return session.make_callable(x, feed_list=[protos])


def _get_prensor_to_ragged_tensor_fn(session, proto_descriptor, path):
  """Returns a callable that projects a field, and creates ragged tensors."""
  protos = tf.compat.v1.placeholder(dtype=tf.string, shape=(None,))
  expr = s2t.expression_impl.proto.create_expression_from_proto(
      protos, proto_descriptor)
  expr = expr.project(path)
  [prensor] = s2t.calculate.calculate_prensors(
      [expr], options=s2t.calculate_options.get_options_with_minimal_checks())
  rt = prensor.get_ragged_tensors()
  with tf.control_dependencies(rt.values()):
    x = tf.constant(1)
  return session.make_callable(x, feed_list=[protos])


def _get_prensor_to_sparse_tensor_fn(session, proto_descriptor, path):
  """Returns a callable that projects a field, and creates sparse tensors."""
  protos = tf.compat.v1.placeholder(dtype=tf.string, shape=(None,))
  expr = s2t.expression_impl.proto.create_expression_from_proto(
      protos, proto_descriptor)
  expr = expr.project(path)
  [prensor] = s2t.calculate.calculate_prensors(
      [expr], options=s2t.calculate_options.get_options_with_minimal_checks())
  st = prensor.get_sparse_tensors()
  with tf.control_dependencies(
      [sparse_tensor.indices for sparse_tensor in st.values()]):
    x = tf.constant(1)
  return session.make_callable(x, feed_list=[protos])


def _get_parse_tf_example_fn(session, features):
  protos = tf.compat.v1.placeholder(dtype=tf.string, shape=(None,))

  tensor_map = tf.io.parse_example(protos, features)
  with tf.control_dependencies(tensor_map.values()):
    x = tf.constant(1)

  return session.make_callable(x, feed_list=[protos])


def _get_parse_tf_example_to_sparse_fn(session, features):
  protos = tf.compat.v1.placeholder(dtype=tf.string, shape=(None,))

  tensor_map = tf.io.parse_example(protos, features)
  with tf.control_dependencies(
      [sparse_tensor.indices for sparse_tensor in tensor_map.values()]):
    x = tf.constant(1)

  return session.make_callable(x, feed_list=[protos])


if __name__ == "__main__":
  tf.test.main()
