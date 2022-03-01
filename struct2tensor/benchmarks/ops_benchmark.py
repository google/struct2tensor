# Copyright 2022 Google LLC
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
# pylint: disable=line-too-long
r"""Benchmarks for struct2tensor.


Usage:
blaze run -c opt --dynamic_mode=off \
  --run_under='perflab \
  --constraints=arch=x86_64,platform_family=iota,platform_genus=sandybridge' \
  //struct2tensor/benchmarks:ops_benchmark \
  -- --notest_mode

Results:
Num Iterations|Total (wall) Time (s)|Wall Time avg(ms)|Wall Time std|User CPU avg (ms)|User CPU std|System CPU avg (ms)|System CPU std
equi_join_any_indices_monotonic_increasing_1000: 10000|12.046009018551558|1.2046009018551558|0.9495328669791991|1.2590000000000146|6.046119048963948|0.09000000000014552|5.595813730989087
equi_join_any_indices_random_1000: 10000|12.026614569593221|1.2026614569593221|0.7186884686309326|1.2809999999997672|17.961518911690458|0.10100000000093132|10.87068154147507
equi_join_indices_monotonic_increasing_1000: 10000|12.022087568882853|1.2022087568882853|0.6371426815230887|1.256000000000131|9.673456340434292|0.10300000000061119|13.443108707001388
equi_join_indices_random_1000: 10000|12.04043987870682|1.2040439878706821|0.6657185129990686|1.2420000000001892|7.272474742950086|0.08600000000005821|5.508487482377499
"""
# pylint: disable=line-too-long

from absl.testing import parameterized
from struct2tensor.benchmarks import struct2tensor_benchmark_util
from struct2tensor.ops import struct2tensor_ops
import tensorflow as tf


class EquiJoinIndicesBenchmarks(struct2tensor_benchmark_util.OpsBenchmarks):
  """Benchmarks for EquiJoinIndices."""

  @parameterized.named_parameters(*[
      dict(
          testcase_name="equi_join_indices_monotonic_increasing",
          fn_name="equi_join_indices_monotonic_increasing",
          fn_args=[],
          data_key="monotonic_increasing",
      ),
      dict(
          testcase_name="equi_join_indices_random",
          fn_name="equi_join_indices_random",
          fn_args=[],
          data_key="random",
      ),
  ])
  def test_equi_join_indices(self, fn_name, fn_args, data_key):

    def benchmark_fn(session):
      a = tf.compat.v1.placeholder(dtype=tf.int64, shape=(None,))
      b = tf.compat.v1.placeholder(dtype=tf.int64, shape=(None,))
      result = struct2tensor_ops.equi_join_indices(a, b)
      with tf.control_dependencies(result):
        x = tf.constant(1)
      return session.make_callable(x, feed_list=[a, b])

    self.run_benchmarks(fn_name, benchmark_fn, fn_args, data_key)


class EquiJoinAnyIndicesBenchmarks(struct2tensor_benchmark_util.OpsBenchmarks):
  """Benchmarks for EquiJoinAnyIndices."""

  @parameterized.named_parameters(*[
      dict(
          testcase_name="equi_join_any_indices_monotonic_increasing",
          fn_name="equi_join_any_indices_monotonic_increasing",
          fn_args=[],
          data_key="monotonic_increasing",
      ),
      dict(
          testcase_name="equi_join_any_indices_random",
          fn_name="equi_join_any_indices_random",
          fn_args=[],
          data_key="random",
      ),
  ])
  def test_equi_join_indices(self, fn_name, fn_args, data_key):

    def benchmark_fn(session):
      a = tf.compat.v1.placeholder(dtype=tf.int64, shape=(None,))
      b = tf.compat.v1.placeholder(dtype=tf.int64, shape=(None,))
      result = struct2tensor_ops.equi_join_any_indices(a, b)
      with tf.control_dependencies(result):
        x = tf.constant(1)
      return session.make_callable(x, feed_list=[a, b])

    self.run_benchmarks(fn_name, benchmark_fn, fn_args, data_key)


if __name__ == "__main__":
  tf.test.main()
