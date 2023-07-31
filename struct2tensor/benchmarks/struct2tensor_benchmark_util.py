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
"""Struct2tensor benchmarks util."""

import os
import random
import statistics
import timeit

from absl import flags
from absl.testing import parameterized
import cpuinfo
import psutil
import tensorflow as tf

from tensorflow.core.protobuf import rewriter_config_pb2  # pylint: disable=g-direct-tensorflow-import


FLAGS = flags.FLAGS

flags.DEFINE_bool(
    "test_mode", False,
    "if True, run all benchmarks with two iterations."
)

_BASE_DIR = os.path.join(os.path.dirname(__file__), "testdata")
_CPU_INFO = cpuinfo.get_cpu_info()


class Struct2tensorBenchmarksBase(parameterized.TestCase):
  """Base Class for Struct2tensor benchmarks.

  These are tensorflow based benchmarks, and ensures that tensors are always
  evaluated.

  The derived class should call
  self.run_benchmarks(fn_name, get_benchmark_fn, fn_args, data_key).
  """

  def _discard_runs(self, benchmark_fn, inputs):
    benchmark_fn(*inputs)

  def run_benchmarks(self, fn_name, get_benchmark_fn, fn_args, data_key):
    """This benchmarks the function specified by `get_benchmark_fn`.

    Args:
      fn_name: A string of the name of the function to benchmark.
      get_benchmark_fn: A function that returns a tf.session callable.
      fn_args: A list of arguments for get_benchmark_fn.
      data_key: A string that determines what data to use for benchmarks. See
        child class for possible defined data_keys.

    """
    print(f"BEGIN {fn_name}:\tNum Iterations\tTotal (wall) Time (s)\t"
          "Wall Time avg(ms)\tWall Time std\tUser CPU avg (ms)\t"
          "User CPU std\tSystem CPU avg (ms)\tSystem CPU std")

    iterations = 1000
    # This is the number of iterations in a sample. We will benchmark the
    # total compute time per sample. And find std across all samples.
    sample_size = 100
    if FLAGS.test_mode:
      print("WARNING: --test_mode is True. Setting iterations to 2 with sample "
            "size of 1.")
      iterations = 2  # 2 iterations so we can calculate stdev.
      sample_size = 1

    with tf.Graph().as_default():
      # Disable control dependency optimization. Otherwise we cannot use the
      # trick to force the parsing to happen without using its output. e.g.:
      # with tf.control_dependencies([parsed_tensors]):
      #   return tf.constant(1)
      with tf.compat.v1.Session(
          config=tf.compat.v1.ConfigProto(
              graph_options=tf.compat.v1.GraphOptions(
                  rewrite_options=rewriter_config_pb2.RewriterConfig(
                      dependency_optimization=rewriter_config_pb2.RewriterConfig
                      .OFF)))) as sess:
        benchmark_fn = get_benchmark_fn(sess, *fn_args)

        for input_data in self._data[data_key]:
          self._discard_runs(benchmark_fn, input_data)

          # Collect timings and run the benchmark function
          name = f"{fn_name}_{len(input_data[0])}"
          wall_times, user_cpu_times, system_cpu_times = [], [], []
          for _ in range(int(iterations / sample_size)):
            start_cpu = psutil.cpu_times()
            start_time = timeit.default_timer()
            for _ in range(sample_size):
              _ = benchmark_fn(*input_data)
            end_cpu = psutil.cpu_times()
            duration = timeit.default_timer() - start_time
            cpu_user_duration = end_cpu.user - start_cpu.user
            cpu_system_duration = end_cpu.system - start_cpu.system

            wall_times.append(duration * 1000)
            user_cpu_times.append(cpu_user_duration * 1000)
            system_cpu_times.append(cpu_system_duration * 1000)

          total_duration = sum(wall_times)
          cpu_user_duration = sum(user_cpu_times)
          cpu_system_duration = sum(system_cpu_times)
          # The columns are |function name|Num Iterations|Total (wall) Time (s)|
          # Wall Time avg(ms)|Wall Time std|User CPU avg (ms)|User CPU std|
          # System CPU avg (ms)|System CPU std|
          print(
              f"{name}: \t{iterations}\t{total_duration / 1000}\t"
              f"{total_duration / iterations}\t{statistics.stdev(wall_times)}\t"
              f"{cpu_user_duration / iterations}\t"
              f"{statistics.stdev(user_cpu_times)}\t"
              f"{cpu_system_duration / iterations}\t"
              f"{statistics.stdev(system_cpu_times)}"
          )


class ProtoDataBenchmarks(Struct2tensorBenchmarksBase):
  """Base class for benchmarks that take proto data as input."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    print(f"cpuinfo: {_CPU_INFO}")

    shuffle_size = 2048

    # Batch sizes are powers of two. i.e. 1, 2, 4, 8, .. up to 2048.
    batch_sizes = [2**i for i in range(11)]

    if FLAGS.test_mode:
      print("WARNING: --test_mode is True. Setting batch_size to 1.")
      shuffle_size = 1
      batch_sizes = [1]

    # load tensor of serialized DeepProtos into memory.
    ds = tf.data.TFRecordDataset(
        os.path.join(_BASE_DIR, "deep_all_types_4096_positive.tfrecord.gz"),
        "GZIP").shuffle(shuffle_size)
    cls._deep_protos = [
        [list(ds.take(batch_size).as_numpy_iterator())]
        for batch_size in batch_sizes
    ]

    # load tensor of serialized FlatProtos into memory.
    ds = tf.data.TFRecordDataset(
        os.path.join(_BASE_DIR, "flat_all_types_4096_positive.tfrecord.gz"),
        "GZIP").shuffle(shuffle_size)
    cls._flat_protos = [
        [list(ds.take(batch_size).as_numpy_iterator())]
        for batch_size in batch_sizes
    ]

    # load tensor of serialized FlatProtos into memory. These protos each have
    # 1 feature, each with 100 values.
    ds = tf.data.TFRecordDataset(
        os.path.join(_BASE_DIR,
                     "flat_all_types_100_int_values_4096.tfrecord.gz"),
        "GZIP").shuffle(shuffle_size)
    cls._flat_protos_1_feature_100_values = [
        [list(ds.take(batch_size).as_numpy_iterator())]
        for batch_size in batch_sizes
    ]

    # load tensor of serialized FlatProtos100 into memory. These protos each
    # have 100 int features, where each feature has 1 value.
    ds = tf.data.TFRecordDataset(
        os.path.join(_BASE_DIR, "flat_100_int_features_4096.tfrecord.gz"),
        "GZIP").shuffle(shuffle_size)
    cls._flat_protos_100_features = [
        [list(ds.take(batch_size).as_numpy_iterator())]
        for batch_size in batch_sizes
    ]

    # load tensor of serialized FlatProtos100 into memory. These protos each
    # have 100 int features, where each feature has 100 values.
    ds = tf.data.TFRecordDataset(
        os.path.join(_BASE_DIR,
                     "flat_100_int_features_100_values_4096.tfrecord.gz"),
        "GZIP").shuffle(shuffle_size)
    cls._flat_protos_100_features_100_values = [
        [list(ds.take(batch_size).as_numpy_iterator())]
        for batch_size in batch_sizes
    ]

    # load tf example into memory. Each example has 1 feature of list length 1.
    ds = tf.data.TFRecordDataset(
        os.path.join(_BASE_DIR, "tf_example_all_types_4096.tfrecord.gz"),
        "GZIP").shuffle(shuffle_size)
    cls._tf_examples = [
        [list(ds.take(batch_size).as_numpy_iterator())]
        for batch_size in batch_sizes
    ]

    # load tf example into memory. Each example has 1 feature of 100 values.
    ds = tf.data.TFRecordDataset(
        os.path.join(_BASE_DIR,
                     "tf_example_1_int_feature_100_values_4096.tfrecord.gz"),
        "GZIP").shuffle(shuffle_size)
    cls._tf_examples_1_feature_100_values = [
        [list(ds.take(batch_size).as_numpy_iterator())]
        for batch_size in batch_sizes
    ]

    # load tf example into memory. Each example has 100 features, where each
    # feature has list length of 1.
    ds = tf.data.TFRecordDataset(
        os.path.join(_BASE_DIR, "tf_example_100_int_features_4096.tfrecord.gz"),
        "GZIP").shuffle(shuffle_size)
    cls._tf_examples_100_features = [
        [list(ds.take(batch_size).as_numpy_iterator())]
        for batch_size in batch_sizes
    ]

    # load tf example into memory. Each example has 100 features, where each
    # feature has list length of 100.
    ds = tf.data.TFRecordDataset(
        os.path.join(_BASE_DIR,
                     "tf_example_100_int_features_100_values_4096.tfrecord.gz"),
        "GZIP").shuffle(shuffle_size)
    cls._tf_examples_100_features_100_values = [
        [list(ds.take(batch_size).as_numpy_iterator())]
        for batch_size in batch_sizes
    ]

    # pylint: disable=protected-access
    cls._data = {
        "deep_protos":
            cls._deep_protos,
        "flat_protos":
            cls._flat_protos,
        "flat_protos_1_feature_100_values":
            cls._flat_protos_1_feature_100_values,
        "flat_protos_100_features":
            cls._flat_protos_100_features,
        "flat_protos_100_features_100_values":
            cls._flat_protos_100_features_100_values,
        "tf_examples":
            cls._tf_examples,
        "tf_examples_1_feature_100_values":
            cls._tf_examples_1_feature_100_values,
        "tf_examples_100_features":
            cls._tf_examples_100_features,
        "tf_examples_100_features_100_values":
            cls._tf_examples_100_features_100_values,
    }
    # pylint: enable=protected-access


class OpsBenchmarks(Struct2tensorBenchmarksBase):
  """Base class for benchmarks that take tensor data as input."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()

    print(f"cpuinfo: {_CPU_INFO}")
    a = list(range(1000))
    b = list(range(1000))

    rand_a = list(range(1000))
    rand_b = list(range(1000))
    random.shuffle(rand_a)
    random.shuffle(rand_b)

    cls._data = {"monotonic_increasing": [[a, b]],
                 "random": [[rand_a, rand_b]]}
