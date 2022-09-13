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
// An op to decode serialized protobuf map entries into tensors.
//
// High-level picture of how it works:
// A kernel instance owns a MapEntryCollector. A MapEntryCollector owns a
// KeyDecoder<FieldTypeOfKey>. The KeyDecoder keeps internally a hash map
// from key to the index of that key in the "keys" attribute.
//
// On each Compute() call, the MapEntryCollector creates a
// ValueCollector<FieldTypeOfValue>. The ValueCollector keeps internally
// "num_keys" vectors of collected values, each corresponds to a key in the
// "keys" attribute. Similarly, it also keeps internally "num_keys" vectors of
// collected parent indices.
//
// In Compute(), for each serialized map entry, the MapEntryCollector asks
// the KeyDecoder to look up the key in its internal hash map and tells
// ValueCollector which of its internal vectors (by the index) is to received
// the parsed value (if the key is found).
//
// The separation of KeyDecoder and ValueCollector allows KeyDecoder to be
// initialized only once for the lifetime of the kernel and be immutable while
// ValueCollector to be per Compute() call and stateful.
//
// By using only the index to communicate between KeyDecoder and ValueCollector,
// we can decopule the key type and value type (as both are template arguments).
#include <memory>
#include <vector>

#include "google/protobuf/descriptor.pb.h"
#include "google/protobuf/descriptor.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/numbers.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "struct2tensor/kernels/streaming_proto_reader.h"
#include "struct2tensor/kernels/vector_to_tensor.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"

namespace struct2tensor {
namespace {
using ::google::protobuf::Descriptor;
using ::google::protobuf::DescriptorPool;
using ::google::protobuf::FieldDescriptor;
using ::google::protobuf::FileDescriptorSet;
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
namespace errors = ::tensorflow::errors;

constexpr int kKeyFieldNumber = 1;
constexpr int kValueFieldNumber = 2;

constexpr int kNotFoundIndex = -1;

// Parses `str` into a value of type `T`.
// The catch-all implementation assumes `T` is an integral type.
template <typename T>
Status ParseStringAs(const std::string& str, T* val) {
  if (!absl::SimpleAtoi(str, val)) {
    return errors::InvalidArgument(
        absl::StrCat("Failed to parse string: ", str, " as integer."));
  }
  return tensorflow::OkStatus();
}

// Specialization for parsing into a string_view.
template <>
Status ParseStringAs<absl::string_view>(const std::string& str,
                                        absl::string_view* val) {
  *val = str;
  return tensorflow::OkStatus();
}

// Specialization for parsing into a boolean.
template <>
Status ParseStringAs<bool>(const std::string& str, bool* val) {
  if (str == "0") {
    *val = false;
    return tensorflow::OkStatus();
  }
  if (str == "1") {
    *val = true;
    return tensorflow::OkStatus();
  }
  return errors::InvalidArgument(
      absl::StrCat("Failed to parse string: ", str, " as bool."));
}

// Statically bind proto field types with in-memory types (which the wire bytes
// will be parsed into and stored inside a vector), and tensorflow::Tensor's
// dtype (which has a 1:1 mapping to Tensor's in-memory type, through
// tensorflow::EnumToDataType<>).
// We use a static_assert to make sure we won't make a mistake when copying from
// our vector containing collected values into the output Tensors.
template <FieldDescriptor::Type kFieldType>
struct FieldTypeTraits {};
#define MATCH_TYPES(FIELD_TYPE_ENUM, FIELD_CPP_TYPE, TENSOR_DTYPE_ENUM)       \
  template <>                                                                 \
  struct FieldTypeTraits<FIELD_TYPE_ENUM> {                                   \
    using FieldCppType = FIELD_CPP_TYPE;                                      \
    static constexpr DataType kTFDataType = TENSOR_DTYPE_ENUM;                \
    using TensorCppType =                                                     \
        typename tensorflow::EnumToDataType<kTFDataType>::Type;               \
    static_assert(sizeof(TensorCppType) == sizeof(FieldCppType) ||            \
                      (std::is_same<TensorCppType, tstring>::value &&         \
                       std::is_same<FieldCppType, absl::string_view>::value), \
                  "Unexpected FIELD_CPP_TYPE and TENSOR_DTYPE_ENUM pair");    \
  };                                                                          \
  constexpr DataType FieldTypeTraits<FIELD_TYPE_ENUM>::kTFDataType;

MATCH_TYPES(FieldDescriptor::TYPE_DOUBLE, double, tensorflow::DT_DOUBLE);
MATCH_TYPES(FieldDescriptor::TYPE_FLOAT, float, tensorflow::DT_FLOAT);
MATCH_TYPES(FieldDescriptor::TYPE_INT64, int64_t, tensorflow::DT_INT64);
MATCH_TYPES(FieldDescriptor::TYPE_UINT64, uint64_t, tensorflow::DT_UINT64);
MATCH_TYPES(FieldDescriptor::TYPE_INT32, int32_t, tensorflow::DT_INT32);
MATCH_TYPES(FieldDescriptor::TYPE_FIXED64, uint64_t, tensorflow::DT_UINT64);
MATCH_TYPES(FieldDescriptor::TYPE_FIXED32, uint32_t, tensorflow::DT_UINT32);
MATCH_TYPES(FieldDescriptor::TYPE_BOOL, bool, tensorflow::DT_BOOL);
// We store string_view for string-typed fields, and only copy the string
// when populating the output tensors. (Also see VectorToTensor()).
MATCH_TYPES(FieldDescriptor::TYPE_STRING, absl::string_view,
            tensorflow::DT_STRING);
// Undefined for FieldDescriptor::TYPE_GROUP because it cannot appear in a map.
MATCH_TYPES(FieldDescriptor::TYPE_BYTES, absl::string_view,
            tensorflow::DT_STRING);
MATCH_TYPES(FieldDescriptor::TYPE_MESSAGE, absl::string_view,
            tensorflow::DT_STRING);
MATCH_TYPES(FieldDescriptor::TYPE_UINT32, uint32_t, tensorflow::DT_UINT32);
MATCH_TYPES(FieldDescriptor::TYPE_ENUM, int32_t, tensorflow::DT_INT32);
MATCH_TYPES(FieldDescriptor::TYPE_SFIXED32, int32_t, tensorflow::DT_INT32);
MATCH_TYPES(FieldDescriptor::TYPE_SFIXED64, int64_t, tensorflow::DT_INT64);
MATCH_TYPES(FieldDescriptor::TYPE_SINT32, int32_t, tensorflow::DT_INT32);
MATCH_TYPES(FieldDescriptor::TYPE_SINT64, int64_t, tensorflow::DT_INT64);

// We also build a map containing allowed pairs of field type and dtype so
// we can do run-time check at Op construction time.
// TODO(martinz): We need a better way to make a dynamic argument a
// static one.
#define FIELD_TYPE_ENUM_TO_TENSOR_TYPE_CASE(FIELD_TYPE_ENUM) \
  case FIELD_TYPE_ENUM:                                      \
    return &FieldTypeTraits<FIELD_TYPE_ENUM>::kTFDataType

const DataType* FieldTypeEnumToDType(const FieldDescriptor::Type field_type) {
  switch (field_type) {
    FIELD_TYPE_ENUM_TO_TENSOR_TYPE_CASE(FieldDescriptor::TYPE_DOUBLE);
    FIELD_TYPE_ENUM_TO_TENSOR_TYPE_CASE(FieldDescriptor::TYPE_FLOAT);
    FIELD_TYPE_ENUM_TO_TENSOR_TYPE_CASE(FieldDescriptor::TYPE_INT64);
    FIELD_TYPE_ENUM_TO_TENSOR_TYPE_CASE(FieldDescriptor::TYPE_UINT64);
    FIELD_TYPE_ENUM_TO_TENSOR_TYPE_CASE(FieldDescriptor::TYPE_INT32);
    FIELD_TYPE_ENUM_TO_TENSOR_TYPE_CASE(FieldDescriptor::TYPE_FIXED64);
    FIELD_TYPE_ENUM_TO_TENSOR_TYPE_CASE(FieldDescriptor::TYPE_FIXED32);
    FIELD_TYPE_ENUM_TO_TENSOR_TYPE_CASE(FieldDescriptor::TYPE_BOOL);
    FIELD_TYPE_ENUM_TO_TENSOR_TYPE_CASE(FieldDescriptor::TYPE_STRING);
    FIELD_TYPE_ENUM_TO_TENSOR_TYPE_CASE(FieldDescriptor::TYPE_BYTES);
    FIELD_TYPE_ENUM_TO_TENSOR_TYPE_CASE(FieldDescriptor::TYPE_MESSAGE);
    FIELD_TYPE_ENUM_TO_TENSOR_TYPE_CASE(FieldDescriptor::TYPE_UINT32);
    FIELD_TYPE_ENUM_TO_TENSOR_TYPE_CASE(FieldDescriptor::TYPE_ENUM);
    FIELD_TYPE_ENUM_TO_TENSOR_TYPE_CASE(FieldDescriptor::TYPE_SFIXED32);
    FIELD_TYPE_ENUM_TO_TENSOR_TYPE_CASE(FieldDescriptor::TYPE_SFIXED64);
    FIELD_TYPE_ENUM_TO_TENSOR_TYPE_CASE(FieldDescriptor::TYPE_SINT32);
    FIELD_TYPE_ENUM_TO_TENSOR_TYPE_CASE(FieldDescriptor::TYPE_SINT64);
    default:
      return nullptr;
  }
}

// Given the type of the map value, the dtype of the output tensor is fixed.
bool FieldTypeMatchesOutputTensorType(const FieldDescriptor::Type field_type,
                                      const DataType output_tensor_type) {
  const DataType* allowed_dtype = FieldTypeEnumToDType(field_type);
  return allowed_dtype && *allowed_dtype == output_tensor_type;
}

class KeyDecoderBase {
 public:
  virtual ~KeyDecoderBase() {}

  // Consumes and parses bytes from the wire (`reader`) into a map key, and
  // looks up the key's index in the "keys" attribute and stores it in
  // `key_index`.
  // Returns an error on parsing error. key_index could be assigned with
  // kNotFoundIndex.
  virtual Status Decode(StreamingProtoReader* reader, int* key_index) const = 0;
};

// Thread-safe.
template <FieldDescriptor::Type kFieldType>
class KeyDecoder : public KeyDecoderBase {
 public:
  using KeyCppType = typename FieldTypeTraits<kFieldType>::FieldCppType;
  static Status Create(const std::vector<std::string>& keys_as_strings,
                       std::unique_ptr<KeyDecoderBase>* key_decoder) {
    // We double-parse these strings just to validate them in this factory
    // method.
    // Note that we can not construct key_to_value_index_ here because the
    // key in that hash map could be absl::string_view pointing to strings
    // in keys_as_strings_.
    for (const std::string& s : keys_as_strings) {
      KeyCppType key;
      TF_RETURN_IF_ERROR(ParseStringAs(s, &key));
    }

    *key_decoder = absl::WrapUnique(new KeyDecoder(keys_as_strings));
    return tensorflow::OkStatus();
  }

  Status Decode(StreamingProtoReader* reader, int* value_index) const override {
    KeyCppType key;
    if (!reader->ReadValue(kFieldType, &key)) {
      return errors::DataLoss("Corrupted key field.");
    }
    auto it = key_to_value_index_.find(key);
    if (it == key_to_value_index_.end()) {
      *value_index = kNotFoundIndex;
    } else {
      *value_index = it->second;
    }
    return tensorflow::OkStatus();
  }

 private:
  explicit KeyDecoder(const std::vector<std::string>& keys_as_strings)
      : keys_as_strings_(keys_as_strings), key_to_value_index_([this]() {
          absl::flat_hash_map<KeyCppType, int> result;
          for (int i = 0; i < keys_as_strings_.size(); ++i) {
            KeyCppType key;
            // Will never fail because we have validated keys_as_strings in
            // Create().
            TF_CHECK_OK(ParseStringAs(keys_as_strings_[i], &key));
            result[key] = i;
          }
          return result;
        }()) {}

  const std::vector<std::string> keys_as_strings_;
  const absl::flat_hash_map<KeyCppType, int> key_to_value_index_;
};

class ValueCollectorBase {
 public:
  virtual ~ValueCollectorBase() {}

  // Consumes bytes from the wire (`reader`), and keeps the parsed value
  // internally.
  virtual Status Consume(StreamingProtoReader* reader) = 0;
  // Commit the currently kept value into values_[key_index] and
  // `parent_index` into parent_indices_[key_index]
  virtual void Commit(int key_index, int64_t parent_index) = 0;
  // Populates `t` with values_[key_index].
  virtual void PopulateValueTensor(const int key_index, Tensor* t,
                                   bool produce_string_view) const = 0;
  // Populates `t` with parent_indices_[key_index].
  virtual void PopulateParentIndicesTensor(const int key_index,
                                             Tensor* t) const = 0;
  // How many values have been collected for key at `key_index`?
  virtual size_t NumCollectedValues(const int key_index) const = 0;
};

// Thread-compatible. But it's expected to be created per Compute() (thus per
// thread).
template <FieldDescriptor::Type kFieldType>
class ValueCollector : public ValueCollectorBase {
 public:
  using ValueCppType = typename FieldTypeTraits<kFieldType>::FieldCppType;
  explicit ValueCollector(int num_keys)
      : values_per_key_(num_keys), parent_indices_per_key_(num_keys) {}

  Status Consume(StreamingProtoReader* reader) override {
    if (!reader->ReadValue(kFieldType, &current_value_)) {
      return errors::DataLoss("Corrupted value field.");
    }
    return tensorflow::OkStatus();
  }
  void Commit(const int key_index, const int64_t parent_index) override {
    values_per_key_[key_index].push_back(current_value_);
    parent_indices_per_key_[key_index].push_back(parent_index);
  }

  void PopulateValueTensor(const int key_index, Tensor* t,
                           bool produce_string_view) const override {
    VectorToTensor(values_per_key_[key_index], t,
                   produce_string_view &&
                       (kFieldType == google::protobuf::FieldDescriptor::TYPE_MESSAGE));
  }

  void PopulateParentIndicesTensor(const int key_index,
                                   Tensor* t) const override {
    VectorToTensor(parent_indices_per_key_[key_index], t, false);
  }

  size_t NumCollectedValues(const int key_index) const override {
    return parent_indices_per_key_[key_index].size();
  }

 private:
  ValueCppType current_value_;
  std::vector<std::vector<ValueCppType>> values_per_key_;
  std::vector<std::vector<int64_t>> parent_indices_per_key_;
};

// Thread-safe.
class MapEntryCollector {
 public:
  static Status Create(
      const std::vector<std::string>& keys_as_strings,
      const FieldDescriptor::Type key_type,
      const FieldDescriptor::Type value_type,
      const DataType output_tensor_dtype,
      std::unique_ptr<const MapEntryCollector>* map_entry_collector) {
    if (!FieldTypeMatchesOutputTensorType(value_type, output_tensor_dtype)) {
      return errors::InvalidArgument(
          absl::StrCat("Value field is of type ", value_type,
                       " but the output tensor type is ", output_tensor_dtype,
                       " which did not match."));
    }
    std::unique_ptr<KeyDecoderBase> key_decoder;
    switch (key_type) {
      case FieldDescriptor::TYPE_INT64:
        TF_RETURN_IF_ERROR(KeyDecoder<FieldDescriptor::TYPE_INT64>::Create(
            keys_as_strings, &key_decoder));
        break;
      case FieldDescriptor::TYPE_INT32:
        TF_RETURN_IF_ERROR(KeyDecoder<FieldDescriptor::TYPE_INT32>::Create(
            keys_as_strings, &key_decoder));
        break;
      case FieldDescriptor::TYPE_UINT64:
        TF_RETURN_IF_ERROR(KeyDecoder<FieldDescriptor::TYPE_UINT64>::Create(
            keys_as_strings, &key_decoder));
        break;
      case FieldDescriptor::TYPE_UINT32:
        TF_RETURN_IF_ERROR(KeyDecoder<FieldDescriptor::TYPE_UINT32>::Create(
            keys_as_strings, &key_decoder));
        break;
      case FieldDescriptor::TYPE_FIXED64:
        TF_RETURN_IF_ERROR(KeyDecoder<FieldDescriptor::TYPE_FIXED64>::Create(
            keys_as_strings, &key_decoder));
        break;
      case FieldDescriptor::TYPE_FIXED32:
        TF_RETURN_IF_ERROR(KeyDecoder<FieldDescriptor::TYPE_FIXED32>::Create(
            keys_as_strings, &key_decoder));
        break;
      case FieldDescriptor::TYPE_SFIXED64:
        TF_RETURN_IF_ERROR(KeyDecoder<FieldDescriptor::TYPE_SFIXED64>::Create(
            keys_as_strings, &key_decoder));
        break;
      case FieldDescriptor::TYPE_SFIXED32:
        TF_RETURN_IF_ERROR(KeyDecoder<FieldDescriptor::TYPE_SFIXED32>::Create(
            keys_as_strings, &key_decoder));
        break;
      case FieldDescriptor::TYPE_SINT64:
        TF_RETURN_IF_ERROR(KeyDecoder<FieldDescriptor::TYPE_SINT64>::Create(
            keys_as_strings, &key_decoder));
        break;
      case FieldDescriptor::TYPE_SINT32:
        TF_RETURN_IF_ERROR(KeyDecoder<FieldDescriptor::TYPE_SINT32>::Create(
            keys_as_strings, &key_decoder));
        break;
      case FieldDescriptor::TYPE_STRING:
        TF_RETURN_IF_ERROR(KeyDecoder<FieldDescriptor::TYPE_STRING>::Create(
            keys_as_strings, &key_decoder));
        break;
      case FieldDescriptor::TYPE_BOOL:
        TF_RETURN_IF_ERROR(KeyDecoder<FieldDescriptor::TYPE_BOOL>::Create(
            keys_as_strings, &key_decoder));
        break;
      default:
        return errors::InvalidArgument(
            absl::StrCat("Unexpected field type for map key: ", key_type));
    }
    *map_entry_collector = absl::WrapUnique(new MapEntryCollector(
        keys_as_strings.size(), std::move(key_decoder), value_type));
    return tensorflow::OkStatus();
  }

  ~MapEntryCollector() {}

  Status ConsumeAndPopulateOutputTensors(
      absl::Span<const tstring> serialized_protos,
      absl::Span<const tensorflow::int64> parent_indices,
      bool produce_string_view, OpKernelContext* op_kernel_contxt) const {
    std::unique_ptr<ValueCollectorBase> value_collector;
    TF_RETURN_IF_ERROR(MakeValueCollector(num_keys_, &value_collector));
    for (int i = 0; i < serialized_protos.size(); ++i) {
      const tstring& p = serialized_protos[i];
      const int64_t parent_index = parent_indices[i];
      StreamingProtoReader reader(p);
      bool key_field_found = false;
      bool value_field_found = false;
      int value_index = kNotFoundIndex;
      // It's possible that one field appear more than once, but only the last
      // appearence counts.
      for (int tag_number; reader.Next(&tag_number);) {
        if (tag_number == kKeyFieldNumber) {
          TF_RETURN_IF_ERROR(key_decoder_->Decode(&reader, &value_index));
          key_field_found = true;
        } else if (tag_number == kValueFieldNumber) {
          TF_RETURN_IF_ERROR(value_collector->Consume(&reader));
          value_field_found = true;
        }
        // Otherwise ignore -- reader.Next() will skip the field automatically.
      }
      // reader.Next() also returns false on parsing error.
      if (reader.ptr() != reader.end()) {
        return errors::DataLoss(
            "Failed to consume the entire serialized string.");
      }
      if (!key_field_found) {
        return errors::DataLoss("Key field not found in a map.");
      }
      // If value is not found, do not collect.
      // TODO(martinz): revisit if value_field_found == false.
      if (value_index >= 0) {
        value_collector->Commit(value_index, parent_index);
      }
    }
    return PopulateOutputTensors(*value_collector, op_kernel_contxt,
                                 produce_string_view);
  }

 private:
  MapEntryCollector(const int num_keys,
                    std::unique_ptr<KeyDecoderBase> key_decoder,
                    const FieldDescriptor::Type value_type)
      : num_keys_(num_keys),
        key_decoder_(std::move(key_decoder)),
        value_type_(value_type) {}

  Status MakeValueCollector(
      const int num_keys,
      std::unique_ptr<ValueCollectorBase>* value_collector) const {
    switch (value_type_) {
      case FieldDescriptor::TYPE_DOUBLE:
        *value_collector =
            absl::make_unique<ValueCollector<FieldDescriptor::TYPE_DOUBLE>>(
                num_keys);
        break;
      case FieldDescriptor::TYPE_FLOAT:
        *value_collector =
            absl::make_unique<ValueCollector<FieldDescriptor::TYPE_FLOAT>>(
                num_keys);
        break;
      case FieldDescriptor::TYPE_INT64:
        *value_collector =
            absl::make_unique<ValueCollector<FieldDescriptor::TYPE_INT64>>(
                num_keys);
        break;
      case FieldDescriptor::TYPE_UINT64:
        *value_collector =
            absl::make_unique<ValueCollector<FieldDescriptor::TYPE_UINT64>>(
                num_keys);
        break;
      case FieldDescriptor::TYPE_INT32:
        *value_collector =
            absl::make_unique<ValueCollector<FieldDescriptor::TYPE_INT32>>(
                num_keys);
        break;
      case FieldDescriptor::TYPE_FIXED64:
        *value_collector =
            absl::make_unique<ValueCollector<FieldDescriptor::TYPE_FIXED64>>(
                num_keys);
        break;
      case FieldDescriptor::TYPE_FIXED32:
        *value_collector =
            absl::make_unique<ValueCollector<FieldDescriptor::TYPE_FIXED32>>(
                num_keys);
        break;
      case FieldDescriptor::TYPE_BOOL:
        *value_collector =
            absl::make_unique<ValueCollector<FieldDescriptor::TYPE_BOOL>>(
                num_keys);
        break;
      case FieldDescriptor::TYPE_STRING:
        *value_collector =
            absl::make_unique<ValueCollector<FieldDescriptor::TYPE_STRING>>(
                num_keys);
        break;
      case FieldDescriptor::TYPE_MESSAGE:
        *value_collector =
            absl::make_unique<ValueCollector<FieldDescriptor::TYPE_MESSAGE>>(
                num_keys);
        break;
      case FieldDescriptor::TYPE_BYTES:
        *value_collector =
            absl::make_unique<ValueCollector<FieldDescriptor::TYPE_BYTES>>(
                num_keys);
        break;
      case FieldDescriptor::TYPE_UINT32:
        *value_collector =
            absl::make_unique<ValueCollector<FieldDescriptor::TYPE_UINT32>>(
                num_keys);
        break;
      case FieldDescriptor::TYPE_ENUM:
        *value_collector =
            absl::make_unique<ValueCollector<FieldDescriptor::TYPE_ENUM>>(
                num_keys);
        break;
      case FieldDescriptor::TYPE_SFIXED32:
        *value_collector =
            absl::make_unique<ValueCollector<FieldDescriptor::TYPE_SFIXED32>>(
                num_keys);
        break;
      case FieldDescriptor::TYPE_SFIXED64:
        *value_collector =
            absl::make_unique<ValueCollector<FieldDescriptor::TYPE_SFIXED64>>(
                num_keys);
        break;
      case FieldDescriptor::TYPE_SINT32:
        *value_collector =
            absl::make_unique<ValueCollector<FieldDescriptor::TYPE_SINT32>>(
                num_keys);
        break;
      case FieldDescriptor::TYPE_SINT64:
        *value_collector =
            absl::make_unique<ValueCollector<FieldDescriptor::TYPE_SINT64>>(
                num_keys);
        break;
      default:
        return errors::InvalidArgument(
            absl::StrCat("Unexpected map value type: ", value_type_));
    }
    return tensorflow::OkStatus();
  }

  Status PopulateOutputTensors(const ValueCollectorBase& value_collector,
                               OpKernelContext* op_kernel_context,
                               bool produce_string_view) const {
    for (int i = 0; i < num_keys_; ++i) {
      TensorShape output_shape;
      const tensorflow::int64 tensor_size =
          value_collector.NumCollectedValues(i);
      TF_RETURN_IF_ERROR(
          TensorShapeUtils::MakeShape(&tensor_size, 1, &output_shape));
      Tensor* output_values_tensor;
      TF_RETURN_IF_ERROR(op_kernel_context->allocate_output(
          i, output_shape, &output_values_tensor));
      value_collector.PopulateValueTensor(i, output_values_tensor,
                                          produce_string_view);

      Tensor* output_parent_indices_tensor;
      TF_RETURN_IF_ERROR(op_kernel_context->allocate_output(
          i + num_keys_, output_shape, &output_parent_indices_tensor));
      value_collector.PopulateParentIndicesTensor(
          i, output_parent_indices_tensor);
    }

    return tensorflow::OkStatus();
  }

  const int num_keys_;
  const std::unique_ptr<const KeyDecoderBase> key_decoder_;
  const FieldDescriptor::Type value_type_;
};

template <int kOpVersion>
class DecodeProtoMapOp : public OpKernel {
 public:
  explicit DecodeProtoMapOp(OpKernelConstruction* context) : OpKernel(context) {
    int num_keys;
    OP_REQUIRES_OK(context, context->GetAttr("num_keys", &num_keys));

    std::string descriptor_literal;
    OP_REQUIRES_OK(context,
                   context->GetAttr("descriptor_literal", &descriptor_literal));
    FileDescriptorSet file_descriptor_set;
    OP_REQUIRES(context,
                file_descriptor_set.ParseFromString(descriptor_literal),
                tensorflow::errors::InvalidArgument(
                    "descriptor_literal is neither empty nor a "
                    "serialized file_descriptor_set."));
    auto descriptor_pool = absl::make_unique<DescriptorPool>();
    for (const auto& file : file_descriptor_set.file()) {
      OP_REQUIRES(
          context, descriptor_pool->BuildFile(file),
          tensorflow::errors::InvalidArgument("could not create DescriptorPool "
                                              "from descriptor_literal."));
      // Note, the order of the files matters: early files cannot depend on
      // later files.
    }

    std::string message_type;
    OP_REQUIRES_OK(context, context->GetAttr("message_type", &message_type));

    const Descriptor* message_desc =
        descriptor_pool->FindMessageTypeByName(message_type);
    OP_REQUIRES(context, message_desc != nullptr,
                errors::InvalidArgument("No descriptor found for message type ",
                                        message_type));
    const FieldDescriptor* key_fd =
        message_desc->FindFieldByNumber(kKeyFieldNumber);
    OP_REQUIRES(context, key_fd != nullptr,
                errors::InvalidArgument("No descriptor found for key field"));
    OP_REQUIRES(
        context, key_fd->name() == "key",
        errors::InvalidArgument(absl::StrCat(
            "Field 1 is not named key -- is this a valid map entry proto?",
            message_desc->full_name())));
    const FieldDescriptor* value_fd =
        message_desc->FindFieldByNumber(kValueFieldNumber);
    OP_REQUIRES(context, value_fd != nullptr,
                errors::InvalidArgument("No descriptor found for value field"));
    OP_REQUIRES(
        context, value_fd->name() == "value",
        errors::InvalidArgument(absl::StrCat(
            "Field 2 is not name value -- is this a valid map entry proto?",
            message_desc->full_name())));

    std::vector<std::string> keys_as_strings;
    OP_REQUIRES_OK(context, context->GetAttr("keys", &keys_as_strings));
    OP_REQUIRES(
        context, keys_as_strings.size() == num_keys,
        errors::InvalidArgument("keys.size() must equal num_keys, but ",
                                keys_as_strings.size(), " != ", num_keys));

    // The number of outputs is enforced by the Op definition so we don't check
    // here.
    for (int i = 0; i < num_keys; ++i) {
      const DataType output_parent_index_dtype =
          context->output_type(i + num_keys);
      OP_REQUIRES(context, output_parent_index_dtype == tensorflow::DT_INT64,
                  errors::InvalidArgument(absl::StrCat(
                      "DType of parent index output ", i, " is not DT_INT64: ",
                      tensorflow::DataType_Name(output_parent_index_dtype))));
    }
    const DataType output_tensor_dtype = context->output_type(0);

    // MapEntryCollector::Create checks that the output_tensor_dtype matches
    // the type of the map values and returns an error if not.
    OP_REQUIRES_OK(context,
                   MapEntryCollector::Create(
                       keys_as_strings, key_fd->type(), value_fd->type(),
                       output_tensor_dtype, &map_entry_collector_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor* serialized_protos_tensor;
    OP_REQUIRES_OK(context, context->input("serialized_map_entries",
                                           &serialized_protos_tensor));
    const Tensor* parent_indices_tensor;
    OP_REQUIRES_OK(context, context->input("map_entries_parent_indices",
                                           &parent_indices_tensor));

    bool produce_string_view = false;
    if (kOpVersion > 1) {
      tensorflow::OpInputList backing_strings;
      OP_REQUIRES_OK(context,
                     context->input_list("backing_string", &backing_strings));
      produce_string_view = (backing_strings.size() != 0);
    }

    const int num_protos = serialized_protos_tensor->NumElements();
    OP_REQUIRES(
        context, num_protos == parent_indices_tensor->NumElements(),
        errors::InvalidArgument(
            "Num parent indices must be equal to number of input protos."));
    OP_REQUIRES_OK(
        context,
        map_entry_collector_->ConsumeAndPopulateOutputTensors(
            absl::MakeConstSpan(
                serialized_protos_tensor->flat<tstring>().data(), num_protos),
            absl::MakeConstSpan(
                parent_indices_tensor->flat<tensorflow::int64>().data(),
                num_protos),
            produce_string_view, context));
  }

  std::unique_ptr<const MapEntryCollector> map_entry_collector_;
};

REGISTER_KERNEL_BUILDER(Name("DecodeProtoMap").Device(DEVICE_CPU),
                        DecodeProtoMapOp<1>);
REGISTER_KERNEL_BUILDER(Name("DecodeProtoMapV2").Device(DEVICE_CPU),
                        DecodeProtoMapOp<2>);

}  // namespace
}  // namespace struct2tensor
