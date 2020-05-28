/* Copyright 2019 Google LLC

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
// DecodeProto is a TensorFlow Op which extracts arbitrary fields
// from protos serialized as strings.
//
// See docs in ../ops/decode_proto_op.cc.
//
// This implementation reads the serialized format using a handful of
// calls from the WireFormatLite API used by generated proto code.
// WireFormatLite is marked as an "internal" proto2 API but is widely
// used in practice and highly unlikely to change.
// This will be much faster than the previous implementation based on
// constructing a temporary dynamic message in memory and using the
// proto reflection api to read it.
// It can be used with any proto whose descriptors are available at
// runtime but should be competitive in speed with approaches that
// compile in the proto definitions.

#include <atomic>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/descriptor.pb.h"
#include "google/protobuf/descriptor.h"
#include "google/protobuf/descriptor_database.h"
#include "google/protobuf/dynamic_message.h"
#include "google/protobuf/message.h"
#include "google/protobuf/text_format.h"
#include "google/protobuf/wire_format.h"
#include "absl/memory/memory.h"
#include "struct2tensor/kernels/vector_to_tensor.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"


namespace struct2tensor {
namespace {
using ::absl::string_view;
using ::google::protobuf::Descriptor;
using ::google::protobuf::DescriptorPool;
using ::google::protobuf::DynamicMessageFactory;
using ::google::protobuf::FieldDescriptor;
using ::google::protobuf::FileDescriptorSet;
using ::google::protobuf::Message;
using ::google::protobuf::TextFormat;
using ::google::protobuf::internal::WireFormatLite;
using ::google::protobuf::io::CodedInputStream;
using ::std::vector;
using ::tensorflow::DataType;
using ::tensorflow::DEVICE_CPU;
using ::tensorflow::OpKernel;
using ::tensorflow::OpKernelConstruction;
using ::tensorflow::OpKernelContext;
using ::tensorflow::Status;
using ::tensorflow::Tensor;
using ::tensorflow::TensorShape;
using ::tensorflow::TensorShapeUtils;
using ::tensorflow::tstring;
using ::tensorflow::errors::DataLoss;
using ::tensorflow::errors::InvalidArgument;
const bool kFailOnDecodeError = true;

// Creates the output tensor of index `output_index` and populates it with
// contents in `vec`.
// If T is int64_t, it will create a tensor of type tensorflow::int64.
// If T is uint64_t, it will create a tensor of type tensorflow::uint64.
template <typename T>
::tensorflow::Status ToOutputTensor(OpKernelContext* context,
                                    const int output_index,
                                    const vector<T>& vec) {
  TensorShape output_shape;
  const tensorflow::int64 tensor_size = vec.size();
  TF_RETURN_IF_ERROR(
      TensorShapeUtils::MakeShape(&tensor_size, 1, &output_shape));

  Tensor* result = nullptr;
  TF_RETURN_IF_ERROR(
      context->allocate_output(output_index, output_shape, &result));

  if (tensor_size > 0) {
    VectorToTensor(vec, result);
  }
  return tensorflow::Status::OK();
}

// Makes `value` refer to bytes for a length-delimited field in the buffer
// backing `input`, and advances `input` to the next field.
// Returns false if there's an irrecoverable error decoding the field.
bool LengthDelimitedFieldToStringView(CodedInputStream* input,
                                      string_view* value) {
  int length = -1;
  // ReadVarintSizeAsInt returns false when the value read can't fit into an
  // int. In this case the message is malformed.
  if (!input->ReadVarintSizeAsInt(&length)) return false;

  // It's possible to have a zero-length field (e.g. an empty submessage).
  if (length == 0) {
    *value = string_view(nullptr, 0);
    return true;
  }

  int total_buffer_size = -1;
  const void* data = nullptr;
  input->GetDirectBufferPointerInline(&data, &total_buffer_size);
  // The buffer must have at least `length` bytes remaining, otherwise the
  // message is malformed.
  if (total_buffer_size < length) return false;

  *value = string_view(static_cast<const char*>(data), length);
  // Now we've "consumed" the data, advance `input` to the next field.
  // Skip might fail if the remaining bytes in the buffer is less than `length`
  // in which case the message is malformed.
  if (!input->Skip(length)) return false;
  return true;
}

// Read from `input` into `value`. Returns false if parsing fails.
template <typename T, enum WireFormatLite::FieldType DataType>
bool ReadFieldValue(CodedInputStream* input, T* value) {
  return WireFormatLite::ReadPrimitive<T, DataType>(input, value);
}

// Specializations for length-delimited fields (string, bytes, message). Avoid
// reading into strings, instead, read into a string_view that refers to bytes
// on the wire.
template <>
bool ReadFieldValue<string_view, WireFormatLite::TYPE_STRING>(
    CodedInputStream* input, string_view* value) {
  return LengthDelimitedFieldToStringView(input, value);
}

template <>
bool ReadFieldValue<string_view, WireFormatLite::TYPE_MESSAGE>(
    CodedInputStream* input, string_view* value) {
  return LengthDelimitedFieldToStringView(input, value);
}

template <>
bool ReadFieldValue<string_view, WireFormatLite::TYPE_BYTES>(
    CodedInputStream* input, string_view* value) {
  return LengthDelimitedFieldToStringView(input, value);
}

// Specialization for tag-delimited fields (group). Avoid reading
// into strings, instead, read into a string_view that refers to bytes on the
// wire.
template <>
bool ReadFieldValue<string_view, WireFormatLite::TYPE_GROUP>(
    CodedInputStream* input, string_view* value) {
  int total_buffer_size = -1;
  const void* data = nullptr;
  const int position_before_skip = input->CurrentPosition();
  input->GetDirectBufferPointerInline(&data, &total_buffer_size);
  // SkipMessage is not trivial (and potentially recursive, and expensive)
  // because every tag between GROUP_BEGIN and GROUP_END needs to be parsed.
  if (!WireFormatLite::SkipMessage(input)) return false;

  const int skipped_length = input->CurrentPosition() - position_before_skip;
  // This condition should never be hit, unless the buffer backing `input` is
  // not a flat array that contains the entire serialized message, in which
  // case the string_view approach is invalid.
  if (total_buffer_size < skipped_length) return false;

  *value = string_view(static_cast<const char*>(data), skipped_length);
  return true;
}

// Returns true iff the field is an extension and its wire format is proto1
// message set wire format (see WireFormatLite::kMessageSetItemNumber).
bool IsMessageSetWireFormatExtension(const FieldDescriptor& field_descriptor) {
  return field_descriptor.is_extension() && field_descriptor.containing_type()
                                                ->options()
                                                .message_set_wire_format();
}

// Abstract class that consumes protocol buffer field values and produces
// tensors.
class FieldBuilder {
 public:
  explicit FieldBuilder(const int wire_number,
                        const int output_index_parent_index,
                        const int output_index_value, const bool is_repeated,
                        const size_t hint_max_num_values)
      : output_index_parent_index_(output_index_parent_index),
        output_index_value_(output_index_value),
        wire_number_(wire_number),
        is_repeated_(is_repeated) {
    parent_indices_.reserve(hint_max_num_values);
  }
  virtual ~FieldBuilder() = default;
  // Consumes a token.
  // input: the coded input stream, where the tag has been consumed and the
  // token is the next thing to read.
  // wire_type: the wire type of the previously read tag.
  // message_index: the index of the message in input.
  virtual tensorflow::Status Consume(CodedInputStream* input,
                                     WireFormatLite::WireType wire_type,
                                     int64_t message_index) = 0;
  // Produces the tensor at the end.
  // Clears the internal state.
  // context is the context of the kernel where we are creating the output.
  virtual tensorflow::Status Produce(OpKernelContext* context) = 0;

  int wire_number() const { return wire_number_; }

  // Returns the number of values collected so far.
  size_t num_values() const { return parent_indices_.size(); }

 protected:
  // The output index of the parent index tensor (in terms of allocate_output).
  const int output_index_parent_index_;
  // The output index of the value tensor (in terms of allocate_output).
  const int output_index_value_;
  // Collected parent indices.
  vector<int64_t> parent_indices_;

  const int wire_number_;
  // Whether or not the field is repeated.
  const bool is_repeated_;
};

// Implementation of FieldBuilder for <cpp type, proto data type> pairs.
template <typename T, enum WireFormatLite::FieldType DataType>
class FieldBuilderImpl : public FieldBuilder {
 public:
  FieldBuilderImpl(const int wire_number, const int output_index_parent_index,
                   const int output_index_value, const bool is_repeated,
                   const size_t hint_max_num_values)
      : FieldBuilder(wire_number, output_index_parent_index, output_index_value,
                     is_repeated, hint_max_num_values) {
    values_.reserve(hint_max_num_values);
  }

  ~FieldBuilderImpl() override = default;

  // Returns true if this builder builds a field that can be packed.
  // A field could be packed iff its default wire type is *not*
  // WIRETYPE_LENGTH_DELIMITED (basically primitive types excluding bytes and
  // strings). This is to shortcut at compilation time the logic in Consume()
  // for DataType that cannot be packed.
  static constexpr bool IsPackableField() {
    return DataType != WireFormatLite::FieldType::TYPE_GROUP &&
           DataType != WireFormatLite::FieldType::TYPE_MESSAGE &&
           DataType != WireFormatLite::FieldType::TYPE_STRING &&
           DataType != WireFormatLite::FieldType::TYPE_BYTES;
  }

  tensorflow::Status Consume(CodedInputStream* input,
                             WireFormatLite::WireType wire_type,
                             int64_t message_index) override {
    const WireFormatLite::WireType schema_wire_type =
        WireFormatLite::WireTypeForFieldType(DataType);
    bool is_packed_primitive = false;
    if (wire_type != schema_wire_type) {
      // We encounter a packed field here.
      // According to the protobuf standard, we can't trust
      // desc->is_packed() to tell us if the repeated field is packed, and
      // must go by the wire format.
      if (IsPackableField() &&
          wire_type == WireFormatLite::WIRETYPE_LENGTH_DELIMITED) {
        is_packed_primitive = true;
      } else if (WireFormatLite::SkipField(
                     input, WireFormatLite::MakeTag(wire_number_, wire_type))) {
        return Status::OK();
      } else {
        return DataLoss("Failed skipping malformed field");
      }
    }
    return is_packed_primitive ? CollectPackedValues(input, message_index)
                               : CollectValue(input, message_index);
  }

  tensorflow::Status Produce(OpKernelContext* context) override {
    TF_RETURN_IF_ERROR(ToOutputTensor(context, output_index_value_, values_));
    TF_RETURN_IF_ERROR(
        ToOutputTensor(context, output_index_parent_index_, parent_indices_));
    return Status::OK();
  }

 private:
  // Parses packed values from `input` and updates `values_` and
  // `parent_indices`. If the field is not repeated but appears multiple times
  // on the wire, only the last value in the pack will be collected.
  Status CollectPackedValues(CodedInputStream* input, int64_t message_index) {
    int length;
    if (!input->ReadVarintSizeAsInt(&length)) {
      return DataLoss("Failed reading length for packed field.");
    }
    const CodedInputStream::Limit limit = input->PushLimit(length);
    while (input->BytesUntilLimit() > 0) {
      TF_RETURN_IF_ERROR(CollectValue(input, message_index));
    }
    input->PopLimit(limit);
    return Status::OK();
  }

  // Parses one value from `input`, then updates `values_` and
  // `parent_indices_`. The collected value might override the last collected
  // value if the field is not repeated but appears multiple times on the wire.
  Status CollectValue(CodedInputStream* input, int64_t message_index) {
    T value;
    if (!ReadFieldValue<T, DataType>(input, &value)) {
      return DataLoss("Failed to parse field.");
    }
    if (is_repeated_ || parent_indices_.empty() ||
        parent_indices_.back() != message_index) {
      values_.push_back(value);
      parent_indices_.push_back(message_index);
    } else {
      values_.back() = value;
    }
    return Status::OK();
  }

  // Collected field values.
  vector<T> values_;
};

// Abstract class for creating FieldBuilder objects.
class FieldBuilderFactory {
 public:
  explicit FieldBuilderFactory(const int wire_number)
      : max_num_values_(0), wire_number_(wire_number) {}
  virtual ~FieldBuilderFactory() = default;
  // Creates a builder, local to a single run of the op.
  virtual std::unique_ptr<FieldBuilder> Create() = 0;

  int wire_number() const { return wire_number_; }

  size_t max_num_values() const {
    // Use the strictest memory order here per recommendation of the cpp primer.
    // As this won't be called with high concurrency.
    return max_num_values_.load(std::memory_order_seq_cst);
  }

  void compare_and_set_max_num_values(size_t current_num_values) {
    // Note: This could be implemented atomically with a compare and swap (like
    // a spin lock), but the current implementation is not atomic. One
    // thread might commit a value that's less than what was just committed
    // after the load(). We are fine with that because it's for an optimization:
    // an incorrect number leads to degraded performance but not incorrect
    // results.
    if (max_num_values() < current_num_values) {
      max_num_values_.store(current_num_values, std::memory_order_seq_cst);
    }
  }

 private:
  // Used for memorizing the maximum size seen so far of the value collecting
  // vectors in the FieldBuilder corresponding to this field. Note that it's the
  // kernel instance that owns all factories, and a kernel instance might be
  // invoked concurrently thus this field might be accessed concurrently.
  std::atomic<size_t> max_num_values_;
  // The wire number of the field to be built.
  const int wire_number_;
};

template <typename T, enum WireFormatLite::FieldType DataType>
class FieldBuilderFactoryImpl : public FieldBuilderFactory {
 public:
  FieldBuilderFactoryImpl(const FieldDescriptor* field_desc,
                          int output_index_parent_index, int output_index_value)
      : FieldBuilderFactory(field_desc->number()),
        output_index_parent_index_(output_index_parent_index),
        output_index_value_(output_index_value),
        is_repeated_(field_desc->is_repeated()) {}
  ~FieldBuilderFactoryImpl() override {}

  std::unique_ptr<FieldBuilder> Create() override {
    return absl::make_unique<FieldBuilderImpl<T, DataType>>(
        wire_number(), output_index_parent_index_, output_index_value_,
        is_repeated_, max_num_values());
  }

 protected:
  // The output index of the parent index tensor (in terms of allocate_output).
  const int output_index_parent_index_;
  // The output index of the value tensor (in terms of allocate_output).
  const int output_index_value_;
  // Whether or not the field is repeated.
  const bool is_repeated_;
};

// Creates a field builder factory for the descriptor.
// descriptor: the field descriptor of the input.
// output_index_parent_index: the index in the op of the parent index
// output tensor.
// output_index_value: the index in the op of the value output tensor.
// dtype: the output data type.
// If the input and output type do not match, return null.
std::unique_ptr<FieldBuilderFactory> CreateFieldBuilderFactory(
    const FieldDescriptor* descriptor, int output_index_parent_index,
    int output_index_value, DataType dtype) {
  // Being very careful here to only create FieldBuilderFactories that are
  // actually valid.
  // Also, note that signed and unsigned ints cannot be cast here.
  switch (descriptor->type()) {
    case WireFormatLite::TYPE_BOOL:
      if (dtype == DataType::DT_BOOL) {
        return absl::make_unique<
            FieldBuilderFactoryImpl<bool, WireFormatLite::TYPE_BOOL>>(
            descriptor, output_index_parent_index, output_index_value);
      } else {
        return nullptr;
      }
    case WireFormatLite::TYPE_INT32:
      if (dtype == DataType::DT_INT32) {
        return absl::make_unique<
            FieldBuilderFactoryImpl<int32_t, WireFormatLite::TYPE_INT32>>(
            descriptor, output_index_parent_index, output_index_value);
      } else {
        return nullptr;
      }
    case WireFormatLite::TYPE_SFIXED32:
      if (dtype == DataType::DT_INT32) {
        return absl::make_unique<
            FieldBuilderFactoryImpl<int32_t, WireFormatLite::TYPE_SFIXED32>>(
            descriptor, output_index_parent_index, output_index_value);
      } else {
        return nullptr;
      }
    case WireFormatLite::TYPE_SINT32:
      if (dtype == DataType::DT_INT32) {
        return absl::make_unique<
            FieldBuilderFactoryImpl<int32_t, WireFormatLite::TYPE_SINT32>>(
            descriptor, output_index_parent_index, output_index_value);
      } else {
        return nullptr;
      }
    case WireFormatLite::TYPE_UINT32:
      if (dtype == DataType::DT_UINT32) {
        return absl::make_unique<
            FieldBuilderFactoryImpl<uint32_t, WireFormatLite::TYPE_UINT32>>(
            descriptor, output_index_parent_index, output_index_value);
      } else {
        return nullptr;
      }
    case WireFormatLite::TYPE_FIXED32:
      if (dtype == DataType::DT_UINT32) {
        return absl::make_unique<
            FieldBuilderFactoryImpl<uint32_t, WireFormatLite::TYPE_FIXED32>>(
            descriptor, output_index_parent_index, output_index_value);
      } else {
        return nullptr;
      }
    case WireFormatLite::TYPE_SFIXED64:
      if (dtype == DataType::DT_INT64) {
        return absl::make_unique<
            FieldBuilderFactoryImpl<int64_t, WireFormatLite::TYPE_SFIXED64>>(
            descriptor, output_index_parent_index, output_index_value);
      } else {
        return nullptr;
      }
    case WireFormatLite::TYPE_SINT64:
      if (dtype == DataType::DT_INT64) {
        return absl::make_unique<
            FieldBuilderFactoryImpl<int64_t, WireFormatLite::TYPE_SINT64>>(
            descriptor, output_index_parent_index, output_index_value);
      } else {
        return nullptr;
      }

    case WireFormatLite::TYPE_INT64:
      if (dtype == DataType::DT_INT64) {
        return absl::make_unique<
            FieldBuilderFactoryImpl<int64_t, WireFormatLite::TYPE_INT64>>(
            descriptor, output_index_parent_index, output_index_value);
      } else {
        return nullptr;
      }
    case WireFormatLite::TYPE_UINT64:
      if (dtype == DataType::DT_UINT64) {
        return absl::make_unique<
            FieldBuilderFactoryImpl<uint64_t, WireFormatLite::TYPE_UINT64>>(
            descriptor, output_index_parent_index, output_index_value);
      } else {
        return nullptr;
      }
    case WireFormatLite::TYPE_FIXED64:
      if (dtype == DataType::DT_UINT64) {
        return absl::make_unique<
            FieldBuilderFactoryImpl<uint64_t, WireFormatLite::TYPE_FIXED64>>(
            descriptor, output_index_parent_index, output_index_value);
      } else {
        return nullptr;
      }
    case WireFormatLite::TYPE_FLOAT:
      if (dtype == DataType::DT_FLOAT) {
        return absl::make_unique<
            FieldBuilderFactoryImpl<float, WireFormatLite::TYPE_FLOAT>>(
            descriptor, output_index_parent_index, output_index_value);
      } else {
        return nullptr;
      }
    case WireFormatLite::TYPE_DOUBLE:
      if (dtype == DataType::DT_DOUBLE) {
        return absl::make_unique<
            FieldBuilderFactoryImpl<double, WireFormatLite::TYPE_DOUBLE>>(
            descriptor, output_index_parent_index, output_index_value);
      } else {
        return nullptr;
      }

    case WireFormatLite::TYPE_STRING:
      if (dtype == DataType::DT_STRING) {
        return std::unique_ptr<FieldBuilderFactory>(
            new FieldBuilderFactoryImpl<string_view,
                                        WireFormatLite::TYPE_STRING>(
                descriptor, output_index_parent_index, output_index_value));
      } else {
        return nullptr;
      }

    case WireFormatLite::TYPE_GROUP:
      if (dtype == DataType::DT_STRING) {
        return std::unique_ptr<FieldBuilderFactory>(
            new FieldBuilderFactoryImpl<string_view,
                                        WireFormatLite::TYPE_GROUP>(
                descriptor, output_index_parent_index, output_index_value));
      } else {
        return nullptr;
      }

    case WireFormatLite::TYPE_MESSAGE:
      if (dtype == DataType::DT_STRING) {
        return std::unique_ptr<FieldBuilderFactory>(
            new FieldBuilderFactoryImpl<string_view,
                                        WireFormatLite::TYPE_MESSAGE>(
                descriptor, output_index_parent_index, output_index_value));
      } else {
        return nullptr;
      }

    case WireFormatLite::TYPE_BYTES:
      if (dtype == DataType::DT_STRING) {
        return std::unique_ptr<FieldBuilderFactory>(
            new FieldBuilderFactoryImpl<string_view,
                                        WireFormatLite::TYPE_BYTES>(
                descriptor, output_index_parent_index, output_index_value));
      } else {
        return nullptr;
      }
    case WireFormatLite::TYPE_ENUM:
      if (dtype == DataType::DT_INT32) {
        return absl::make_unique<
            FieldBuilderFactoryImpl<int, WireFormatLite::TYPE_ENUM>>(
            descriptor, output_index_parent_index, output_index_value);
      } else {
        return nullptr;
      }
  }
}

// Returns a FieldDescriptor for a step, whether it is a normal field
// or an extension. If the field is not well-formed, returns nullptr.
const FieldDescriptor* FindFieldByName(const DescriptorPool* pool,
                                       const Descriptor* descriptor,
                                       const std::string& field_name) {
  if (field_name.empty()) {
    return nullptr;
  } else if (field_name[0] == '(' && field_name[field_name.size() - 1] == ')') {
    // If the first and last characters are different, field_name.size() >= 2.
    return pool->FindExtensionByName(
        field_name.substr(1, field_name.size() - 2));
  } else {
    return descriptor->FindFieldByName(field_name);
  }
}

// Binds a field builder with the factory that creates it. Later a
// field builder will report its number of collected values to its factory.
struct FieldBuilderAndFactory {
  FieldBuilder* field_builder;
  FieldBuilderFactory* field_builder_factory;
};

class DecodeProtoSparseOp : public OpKernel {
 public:
  explicit DecodeProtoSparseOp(OpKernelConstruction* context)
      : OpKernel(context) {
    std::string descriptor_literal;
    OP_REQUIRES_OK(context,
                   context->GetAttr("descriptor_literal", &descriptor_literal));
    std::string descriptor_source;
    OP_REQUIRES_OK(context,
                   context->GetAttr("descriptor_source", &descriptor_source));

    int num_fields_attr;
    OP_REQUIRES_OK(context, context->GetAttr("num_fields", &num_fields_attr));

    OP_REQUIRES(
        context, !descriptor_literal.empty(),
        InvalidArgument(
            "descriptor_literal must be a serialized file_descriptor_set."));
    FileDescriptorSet file_descriptor_set;
    OP_REQUIRES(context,
                file_descriptor_set.ParseFromString(descriptor_literal),
                tensorflow::errors::InvalidArgument(
                    "descriptor_literal is neither empty nor a "
                    "serialized file_descriptor_set."));
    file_descriptor_set.ParseFromString(descriptor_literal);
    desc_pool_ = absl::make_unique<DescriptorPool>();
    for (const auto& file : file_descriptor_set.file()) {
      // Note, the order of the files matters: early files cannot depend on
      // later files.
      OP_REQUIRES(
          context, desc_pool_->BuildFile(file),
          tensorflow::errors::InvalidArgument("could not create DescriptorPool "
                                              "from descriptor_literal."));
    }

    std::string message_type;
    OP_REQUIRES_OK(context, context->GetAttr("message_type", &message_type));

    const Descriptor* message_desc =
        desc_pool_->FindMessageTypeByName(message_type);
    OP_REQUIRES(
        context, message_desc != nullptr,
        InvalidArgument("No descriptor found for message type ", message_type));

    vector<std::string> field_names;
    OP_REQUIRES_OK(context, context->GetAttr("field_names", &field_names));
    OP_REQUIRES(
        context, field_names.size() == num_fields_attr,
        InvalidArgument("field_names.size() must equal num_fields, but ",
                        field_names.size(), " != ", num_fields_attr));

    vector<DataType> output_types;
    OP_REQUIRES_OK(context, context->GetAttr("output_types", &output_types));
    OP_REQUIRES(context, field_names.size() == output_types.size(),
                InvalidArgument("field_names and output_types attributes must "
                                "have the same length"));

    // Gather the field descriptors and check that requested output types
    // match.

    const int field_count = field_names.size();
    int field_index = 0;
    for (const std::string& name : field_names) {
      const auto* fd = FindFieldByName(desc_pool_.get(), message_desc, name);
      if (IsMessageSetWireFormatExtension(*fd)) {
        has_message_set_wire_format_extension_ = true;
      }
      OP_REQUIRES(context, fd != nullptr,
                  InvalidArgument("Unknown field: ", name, " in message type ",
                                  message_type));

      std::unique_ptr<FieldBuilderFactory> factory =
          CreateFieldBuilderFactory(fd, field_index + field_count, field_index,
                                    output_types[field_index]);
      OP_REQUIRES(
          context, factory,
          InvalidArgument("Unexpected output type for ", fd->full_name(), ": ",
                          fd->cpp_type(), " to ", output_types[field_index]));

      field_builder_factories_.push_back(std::move(factory));
      ++field_index;
    }

    // Reorder factories in the order of the wire numbers.
    // We want field_builders sorted by their number on the wire.
    // But the field_builder_factories_ are allocated in the order given by
    // the caller.
    std::sort(field_builder_factories_.begin(), field_builder_factories_.end(),
              [](const std::unique_ptr<FieldBuilderFactory>& a,
                 const std::unique_ptr<FieldBuilderFactory>& b) {
                return a->wire_number() < b->wire_number();
              });

    message_prototype_ = message_factory_.GetPrototype(message_desc);
    OP_REQUIRES(context, message_prototype_ != nullptr,
                InvalidArgument("Couldn't get prototype message: ",
                                message_desc->full_name()));
    std::string format;
    OP_REQUIRES_OK(context, context->GetAttr("message_format", &format));
    OP_REQUIRES(context, format == "binary" || format == "text",
                InvalidArgument("format must be one of binary or text"));
    is_binary_ = format == "binary";

    // Enable the initial protobuf sanitizer, which is much
    // more expensive than the decoder.
    // TODO(nix): Remove this once the fast decoder
    // has passed security review.
    OP_REQUIRES_OK(context, context->GetAttr("sanitize", &sanitize_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& buf_tensor = ctx->input(0);
    const int message_count = buf_tensor.NumElements();
    const int field_count = field_builder_factories_.size();

    OP_REQUIRES(ctx, ctx->num_outputs() == field_count * 2,
                InvalidArgument(
                    "Number of outputs is not twice the number of fields."));

    // This is used to allocate binary bufs if used. It serves only
    // to define memory ownership.
    vector<tstring> tmp_binary_bufs;

    // These are the actual buffers to use, which may be in tmp_binary_bufs
    // or may be pointers into the buf_tensor. Either way they are not owned
    // here.
    vector<const tstring*> bufs;
    bufs.reserve(message_count);

    if (is_binary_ && !sanitize_) {
      // Fast path.
      for (int mi = 0; mi < message_count; ++mi) {
        const tstring* buf = &buf_tensor.flat<tstring>()(mi);
        bufs.push_back(buf);
      }
    } else {
      tmp_binary_bufs.reserve(message_count);
      // We will have to allocate a copy, either to convert from text to
      // binary or to sanitize a binary proto.
      for (int mi = 0; mi < message_count; ++mi) {
        tmp_binary_bufs.emplace_back();
        ReserializeMessage(ctx, buf_tensor.flat<tstring>()(mi),
                           &tmp_binary_bufs.back());
        if (!ctx->status().ok()) {
          return;
        }
        bufs.push_back(&tmp_binary_bufs[mi]);
      }
    }

    // Create builders.
    vector<std::unique_ptr<FieldBuilder>> builders;
    builders.reserve(field_builder_factories_.size());

    vector<FieldBuilderAndFactory> field_builders_and_factories;
    field_builders_and_factories.reserve(field_builder_factories_.size());
    for (const auto& factory : field_builder_factories_) {
      builders.push_back(factory->Create());
      field_builders_and_factories.push_back(
          FieldBuilderAndFactory{builders.back().get(), factory.get()});
    }

    // Let builders collect the field values.
    ConsumeProtos(ctx, bufs, builders);
    // This is the wire number order. I am counting on the fact that it does
    // not matter the order in which you optimize fields.
    for (const auto& builder : builders) {
      OP_REQUIRES_OK(ctx, builder->Produce(ctx));
    }

    // Collect maximum number of collected values from each field builder.
    // Must happen after ConsumeProtos().
    for (const auto& builder_and_factory : field_builders_and_factories) {
      builder_and_factory.field_builder_factory->compare_and_set_max_num_values(
          builder_and_factory.field_builder->num_values());
    }
  }

 private:
  // Copy a serialized message to binary, e.g. to handle text proto inputs.
  void ReserializeMessage(OpKernelContext* ctx, const tstring& buf,
                          tstring* binary_buf) {
    // Handle text protos by translating them to binary.
    std::unique_ptr<Message> message(message_prototype_->New());
    OP_REQUIRES(ctx, message, DataLoss("Initializing message failed"));

    if (is_binary_) {
      // If we get here we are sanitizing the input protobuf by parsing
      // and reserializing it with a trusted (but very slow) library.
      OP_REQUIRES(ctx, message->ParseFromString(buf),
                  DataLoss("Unable to parse binary protobuf"));
    } else {
      OP_REQUIRES(ctx, TextFormat::ParseFromString(buf, message.get()),
                  DataLoss("Unable to parse text protobuf"));
    }

    OP_REQUIRES(ctx, ::tensorflow::SerializeToTString(*message, binary_buf),
                DataLoss("Unable to reserialize text proto as binary"));
  }

  // Parse fields from a serialized message into vectors.
  void ConsumeProtos(
      OpKernelContext* ctx, const vector<const tstring*>& bufs,
      const vector<std::unique_ptr<FieldBuilder>>& field_builders) {
    for (int message_index = 0; message_index < bufs.size(); ++message_index) {
      const tstring& buf = *bufs[message_index];
      // When collecting field values, we don't want to copy values of string
      // types (string fields, sub messages, etc). Instead we want to collect
      // string_views pointing back to the wire format. Therefore `input` must
      // be backed by a flat array that contains the entire message. This
      // c'tor ensures that, and the IsFlat() check below verifies that.
      CodedInputStream input(reinterpret_cast<const uint8_t*>(buf.c_str()),
                             buf.size());

      OP_REQUIRES(ctx, input.IsFlat(),
                  DataLoss("Failed to construct a flat CodedInputStream"));

      Status st = ConsumeOneProto(&input, message_index, field_builders);

      if (st.ok() && !input.ConsumedEntireMessage()) {
        st = DataLoss("Failed to consume entire buffer");
      }
      if (kFailOnDecodeError) {
        OP_REQUIRES_OK(ctx, st);  // NOLINT
      }
      if (!st.ok()) {
        // This code suppresses the corrupt proto, treating it as empty
        // to avoid crashing training.
        LOG(WARNING) << "Proto counting error for message type "
                     << message_type_ << ": " << st;
      }
    }
  }

  // Look up the FieldBuilder for a particular field number.
  bool LookupFieldBuilder(
      int field_number, int* field_index,
      const vector<std::unique_ptr<FieldBuilder>>& field_builders) {
    // Look up the FieldDescriptor using linear search.
    // TODO(nix): this could be sped up with binary search, but we are
    // already way off the fastpath at this point. If you see a hotspot
    // here, somebody is sending you very inefficient protos.
    for (int fi = field_builders.size() - 1; fi >= 0; fi--) {
      if (field_number == field_builders[fi]->wire_number()) {
        *field_index = fi;
        return true;
      }
    }
    return false;
  }

  // Handles proto1 MessageSet wire format. `input` is expected to be at a
  // position just passing a kMessageSetItemStartTag.
  // The contents between the kMessageSetItemStartTag and the
  // kMessageSetItemEndTag will be consumed (inclusively).
  // `message_index` is the index of the proto message currently being parsed
  // in the input tensor of protos. Returns false if parsing fails.
  bool HandleMessageSetItemGroup(
      CodedInputStream* input, const int message_index,
      const vector<std::unique_ptr<FieldBuilder>>& field_builders) {
    uint32_t type_id = 0;
    string_view message_data;
    // The following logic attempts to parse a proto group as follows:
    // group MessageSetItem {
    //   // extension field number.
    //   required int32_t type_id = 1;
    //   // serialized extension message.
    //   required string message = 2;
    // }
    //
    // There might be multiple of each field on wire, the last appearence of
    // each will be taken. Unknown fields will be skipped.
    while (true) {
      const uint32_t tag = input->ReadTagNoLastTag();
      if (tag == 0) return false;
      switch (tag) {
        case WireFormatLite::kMessageSetTypeIdTag:
          if (!input->ReadVarint32(&type_id)) return false;
          break;

        case WireFormatLite::kMessageSetMessageTag: {
          // the message field is length-delimited:
          // <length in varint32><bytes of length>
          // The entire field (length + bytes) will be passed to the field
          // builder.
          const int position_before_skip = input->CurrentPosition();
          int total_buffer_size = -1;
          const void* data = nullptr;
          input->GetDirectBufferPointerInline(&data, &total_buffer_size);
          int length;
          if (!input->ReadVarintSizeAsInt(&length)) return false;
          if (!input->Skip(length)) return false;
          message_data =
              string_view(static_cast<const char*>(data),
                          input->CurrentPosition() - position_before_skip);
          break;
        }

        case WireFormatLite::kMessageSetItemEndTag: {
          int field_index;
          // Both fields are required so if not encountered yet it's a
          // malformed message. Note that message_data is not empty even if
          // the sub-message is empty because it contains the length.
          if (message_data.empty() || type_id == 0) return false;

          if (LookupFieldBuilder(type_id, &field_index, field_builders)) {
            CodedInputStream sub_input(
                reinterpret_cast<const uint8_t*>(message_data.data()),
                message_data.size());
            // We could do the optimization similar to that in
            // ConsumeOneProto() to look up the FieldBuilder. But it's
            // probably not worth it as there are usually not many
            // FieldBuilders (each would be for a requested message
            // extension).
            if (!field_builders[field_index]
                     ->Consume(&sub_input,
                               WireFormatLite::GetTagWireType(
                                   WireFormatLite::kMessageSetMessageTag),
                               message_index)
                     .ok()) {
              return false;
            }
          }
          return true;
        }

        default:
          if (!WireFormatLite::SkipField(input, tag)) return false;
      }
    }
  }

  // Traverses a serialized protobuf, dispatching values to the
  // field_builders. input contains the protobuf. index is the index of the
  // message. field_builders contains the builders.
  // field_builders must be sorted by increasing wire_number.
  Status ConsumeOneProto(
      CodedInputStream* input, int index,
      const vector<std::unique_ptr<FieldBuilder>>& field_builders) {
    // At the beginning of each loop, the last field number that was seen,
    // regardless of whether it was parsed or not, or -1 if no field has
    // been seen before.
    int last_seen_field_number = -1;
    // The field builder that is expected to be used next.
    // It was either used to parse the last seen field number, or if the
    // last seen field number was not in field_builders, it is the next
    // field builder after the last seen field number.
    // At the beginning it is the first field_builder.
    auto expected_field_builder_iter = field_builders.begin();

    // The 'tag' variable should always be treated as tainted.
    uint32_t tag;
    for (tag = input->ReadTag();
         tag != 0 && WireFormatLite::GetTagWireType(tag) !=
                         WireFormatLite::WIRETYPE_END_GROUP;
         tag = input->ReadTag()) {
      DCHECK(expected_field_builder_iter == field_builders.begin() ||
             last_seen_field_number >
                 (*(expected_field_builder_iter - 1))->wire_number());
      DCHECK(expected_field_builder_iter == field_builders.end() ||
             last_seen_field_number <=
                 (*expected_field_builder_iter)->wire_number());

      // Special handling for proto1 MessageSet wire format.
      // (proto2 MessageSet bridge is also serialized into this wire format
      // by default).
      if (has_message_set_wire_format_extension_ &&
          tag == WireFormatLite::kMessageSetItemStartTag) {
        if (!HandleMessageSetItemGroup(input, index, field_builders)) {
          return DataLoss("Unable to parse MessageSet wire format.");
        }
        continue;
      }

      // The field wire number.
      const int field_number = WireFormatLite::GetTagFieldNumber(tag);
      // The field associated with the field wire number.
      FieldBuilder* field_builder = nullptr;

      // field_builders are ordered by their field numbers. If the field numbers
      // on wire are also ordered (which is a convention), then we can
      // monotonically increment `expected_field_builder_iter` as the field
      // numbers on wire get larger. If we detect any out-of-order
      // field number, we reset `expected_field_builder_iter`, and expect that
      // future wire numbers are ordered. This algorithm is quadratic in the
      // worst case where field numbers on wire are in descending order, however
      // it works well in the case where two serialized protobufs are
      // concatenated together.
      if (field_number < last_seen_field_number) {
        expected_field_builder_iter = field_builders.begin();
      }

      // Advance expected_field_builder_iter until
      // field_number <= expected_field_number.
      for (; expected_field_builder_iter != field_builders.end();
           ++expected_field_builder_iter) {
        DCHECK(expected_field_builder_iter == field_builders.begin() ||
               field_number >
                   (*(expected_field_builder_iter - 1))->wire_number());
        FieldBuilder* expected_field_builder =
            expected_field_builder_iter->get();
        if (field_number <= expected_field_builder->wire_number()) {
          if (field_number == expected_field_builder->wire_number()) {
            field_builder = expected_field_builder;
          }
          break;
        }
      }

      last_seen_field_number = field_number;

      if (!field_builder) {
        // This DCHECK verifies that if we skip a field, we didn't want it.
        // In particular, field_builders is empty or the field_number is either:
        // before field_builders.begin().wire_number() or
        // after (field_builders.end() - 1).wire_number() or
        // in-between expected_field_builder_iter and
        // expected_field_builder_iter - 1.
        DCHECK(field_builders.empty() ||
               (field_number < (*field_builders.begin())->wire_number()) ||
               (field_number > (*(field_builders.end() - 1))->wire_number()) ||
               (((*(expected_field_builder_iter - 1))->wire_number() <
                 field_number) &&
                (field_number <
                 (*(expected_field_builder_iter))->wire_number())));
        // Unknown and unrequested field_builders are skipped.
        if (!WireFormatLite::SkipField(input, tag)) {
          return DataLoss("Failed skipping unrequested field");
        }
        continue;
      }

      DCHECK(field_number == field_builder->wire_number());
      TF_RETURN_IF_ERROR(field_builder->Consume(
          input, WireFormatLite::GetTagWireType(tag), index));
    }
    // If the last read tag is END_GROUP it should be the very last thing left
    // in the buffer.
    if (WireFormatLite::GetTagWireType(tag) ==
            WireFormatLite::WIRETYPE_END_GROUP &&
        input->ReadTag() != 0) {
      return DataLoss(
          "Encountered WIRETYPE_END_GROUP but the message did not end with "
          "it.");
    }

    return Status::OK();
  }

  std::string message_type_;
  // Fields are ordered by wire number.
  vector<std::unique_ptr<FieldBuilderFactory>> field_builder_factories_;

  // Owned_desc_pool_ is null when using descriptor_source=local.
  std::unique_ptr<DescriptorPool> desc_pool_;
  DynamicMessageFactory message_factory_;
  const Message* message_prototype_;

  // True if decoding binary format, false if decoding text format.
  bool is_binary_;

  // True if the protos should be sanitized before parsing.
  // Enables the initial protobuf sanitizer, which is much
  // more expensive than the decoder. The flag defaults to true
  // but can be set to false for trusted sources.
  // TODO(nix): flip the default to false when the fast decoder
  // has passed security review.
  bool sanitize_;

  // True iff a extension field is requested *and* the containing message
  // has "proto2.MessageOptions.message_set_wire_format" enabled.
  // With that option enabled, the extensions will be serialized into a
  // wire format which needs special handling.
  bool has_message_set_wire_format_extension_ = false;

  TF_DISALLOW_COPY_AND_ASSIGN(DecodeProtoSparseOp);
};

REGISTER_KERNEL_BUILDER(Name("DecodeProtoSparseV2").Device(DEVICE_CPU),
                        DecodeProtoSparseOp);

}  // namespace
}  // namespace struct2tensor
