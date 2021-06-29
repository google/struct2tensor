/* Copyright 2020 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Benchmarks to compare the serialization cost of tf.Example to that of a proto
// with similar structure but no maps.
//
// To run on perflab:
// blaze run -c opt --dynamic_mode=off \
// third_party/py/struct2tensor/google/benchmarks:serialization_benchmark \
// --run_under="perflab\
//   --constraints=arch=x86_64,platform_family=iota,platform_genus=sandybridge"
//
// Results (Last Update: 11/05/2020):
//

/**
Benchmarking blaze-out/k8-opt/bin/third_party/py/struct2tensor/google/benchmarks/serialization_benchmark
Run on iky92 (32 X 2600 MHz CPUs) [iota-sandybridge]; 2020-11-05T16:11:59.206047213-08:00
CPU: Intel Sandybridge with HyperThreading (16 cores) dL1:32KB dL2:256KB dL3:20MB
Benchmark                       Time(ns)        CPU(ns)     Iterations
----------------------------------------------------------------------
BM_SerializeExample/1/1              327            326         172049 84.747MB/s
BM_SerializeExample/1/100            801            800          74890 610.700MB/s
BM_SerializeExample/100/1           6650           6634           8701 398.488MB/s
BM_SerializeExample/100/100        64032          63869            911 758.661MB/s
BM_SerializeFlatProto/1/1            363            362         161172 15.803MB/s
BM_SerializeFlatProto/1/100         1069           1067          53452 432.651MB/s
BM_SerializeFlatProto/100/1         1456           1453          40177 503.333MB/s
BM_SerializeFlatProto/100/100      53926          53725           1000 860.641MB/s
*/

#include <iostream>

#include "google/protobuf/descriptor.h"
#include "google/protobuf/message.h"
#include "testing/base/public/benchmark.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/random/distributions.h"
#include "absl/random/random.h"
#include "struct2tensor/benchmarks/benchmark.pb.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/example/feature_util.h"

namespace struct2tensor {
namespace benchmark {
namespace {
using ::benchmark::State;
using ::tensorflow::Example;

struct PayloadSize {
  int num_features;
  int num_feature_values;
};

PayloadSize GetPayloadSize(const State& state) {
  return PayloadSize{state.range(0), state.range(1)};
}

std::string GetFeatureName(int index) {
  return absl::StrCat("int_values_", index);
}

// This generates a random int64 with 9 digits. 9 digits because that was the
// restriction on the generated protos for the prensor benchmarks.
int64 GetRandomInt64() {
  absl::BitGen gen;
  return absl::Uniform<int>(gen, 100000000, 1000000000);
}

// Creates a list of random int64 with 9 digits.
std::vector<int64> GetIntFeatureList(int num_values) {\
  std::vector<int64> feature_list(num_values);
  std::generate(feature_list.begin(), feature_list.end(), GetRandomInt64);
  return feature_list;
}


// Builds a tf.Example with specified number of features, and each feature has
// num_feature_values values. Each feature is an int64 list.
Example GetExample(const PayloadSize payload_size) {
  absl::BitGen gen;
  Example result;
  for (int i = 0; i < payload_size.num_features; ++i) {
    auto* features = result.mutable_features();
    tensorflow::AppendFeatureValues(
        GetIntFeatureList(payload_size.num_feature_values),
        GetFeatureName(i), features);
  }
  return result;
}

// Builds a FlatProto100 which is a flattened version of the tf.Example, in the
// sense that there is no map, and no keys.
FlatProto100 GetFlatProto(const PayloadSize payload_size) {
  absl::BitGen gen;
  FlatProto100 result;
  const auto* feature_descriptor = FlatProto100::descriptor();
  const auto* flat_proto_reflection = result.GetReflection();

  // Unfortunately, FlatProto100's features are 1 indexed.
  for (int i = 1; i <= payload_size.num_features; ++i) {
    const auto* feature_field = feature_descriptor->FindFieldByNumber(i);
    for (int j = 0; j < payload_size.num_feature_values; ++j) {
      flat_proto_reflection->AddInt64(&result, feature_field, GetRandomInt64());
    }
  }
  return result;
}

// Benchmark cost of serialization of a tf.Example.
void BM_SerializeExample(State& state) {
  const Example example = GetExample(GetPayloadSize(state));
  size_t total_size = 0;
  const size_t size_one = example.ByteSizeLong();
  for (auto _ : state) {
    std::string s;
    example.SerializeToString(&s);
    testing::DoNotOptimize(s);
    total_size += size_one;
  }
  state.SetBytesProcessed(total_size);
}

// Benchmark cost of serialization of a FlatProto.
void BM_SerializeFlatProto(State& state) {
  const FlatProto100 flat_proto = GetFlatProto(GetPayloadSize(state));
  size_t total_size = 0;
  const size_t size_one = flat_proto.ByteSizeLong();
  for (auto _ : state) {
    std::string s;
    flat_proto.SerializeToString(&s);
    testing::DoNotOptimize(s);
    total_size += size_one;
  }
  state.SetBytesProcessed(total_size);
}

BENCHMARK(BM_SerializeExample)
    ->Args({1, 1})
    ->Args({1, 100})
    ->Args({100, 1})
    ->Args({100, 100});

BENCHMARK(BM_SerializeFlatProto)
    ->Args({1, 1})
    ->Args({1, 100})
    ->Args({100, 1})
    ->Args({100, 100});

}  // namespace
}  // namespace benchmark
}  // namespace struct2tensor

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  absl::SetFlag(&FLAGS_benchmarks, "all");
  RunSpecifiedBenchmarks();
}
