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
#ifndef STRUCT2TENSOR_PY_KERNELS_STREAMING_PROTO_READER_H_
#define STRUCT2TENSOR_PY_KERNELS_STREAMING_PROTO_READER_H_

// Low-level streaming parsing for protocol buffers. There are more
// user-friendly alternatives available in message.h
//
// LOG and CHECK free.


#include "absl/base/internal/endian.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/descriptor.h"
#include "google/protobuf/message.h"
#include "google/protobuf/wire_format_lite.h"
#include "tensorflow/core/lib/core/coding.h"
#include "tensorflow/core/platform/types.h"

namespace struct2tensor {

// Implements the most low-level streaming (pull) parsing of serialized protocol
// buffers. Typical loop looks as follows:
//
//   int field_number = 0;
//   while (reader.Next(&field_number)) {
//     switch (field_number) {
//       case TestProto::kNameFieldNumber:
//         CHECK(reader.ReadValue(...));  // optional
//         break;
//     }
//   }
//
// If field value was not read, it will be skipped automatically.
//
// StreamingProtoReader will never read past the end of input StringPiece.
class StreamingProtoReader {
 public:
  explicit StreamingProtoReader(absl::string_view proto);

  // Advances the stream to the next available field. Returns false when end of
  // stream is reached or stream is corrupt.
  inline bool Next(int* number);

  // Attempts to read the content. Returns false if value was not read
  // successfully i.e. there was field type and value type mismatch, or stream
  // was terminated unexpectedly. In this case internal pointer will not be
  // advanced.
  //
  // If field contains packed values, do the following:
  //   PackedValues values;
  //   ReadValue(google::protobuf::FieldDescriptor::TYPE_FLOAT, &values);
  //   PackedValuesReader<float> value_reader(values);
  //
  // This may report success while retrieving a wrong value, if the value
  // type is not the C++ type defined for the field type. See test case
  // UnmatchedOutputTypeSilentFailure for an example.
  template <class T>
  bool ReadValue(google::protobuf::FieldDescriptor::Type field_type, T* value);

  // Returns a wire type for the last read field descriptor.
  google::protobuf::internal::WireFormatLite::WireType wire_type() const {
    return wire_type_;
  }

  // Returns a position in serialized protobuf where this reader will be reading
  // from.
  const char* ptr() const { return ptr_; }

  // Returns a pointer to a first byte past serialized protobuf.
  const char* end() const { return end_; }

 private:
  const char* ptr_;
  const char* const end_;

  // Most recent tag's wire type tag read from the stream.
  google::protobuf::internal::WireFormatLite::WireType wire_type_;

  // true if it is possible to read field content using ReadValue/ReadField.
  bool content_available_;
};

// Copyable container for packed values. Data will stay valid until original
// serialized protocol buffer stays around.
class PackedValues {
 public:
  PackedValues() : field_type_(google::protobuf::FieldDescriptor::MAX_TYPE) {}

  PackedValues(google::protobuf::FieldDescriptor::Type field_type, absl::string_view data)
      : field_type_(field_type), data_(data) {}

  google::protobuf::FieldDescriptor::Type field_type() const { return field_type_; }

  absl::string_view data() const { return data_; }

 private:
  google::protobuf::FieldDescriptor::Type field_type_;
  absl::string_view data_;
};

// Reader for packed values.
template <class T>
class PackedValuesReader {
 public:
  explicit PackedValuesReader(const PackedValues& values);

  // Returns true if next value was read successfully.
  inline bool Next(T* value);

 private:
  const char* ptr_;
  const char* const end_;
  const google::protobuf::FieldDescriptor::Type field_type_;
  const google::protobuf::internal::WireFormatLite::WireType wire_type_;
};

// Utility function to parse message_set items. MessageSet item's wire format
// is equivalent to:
//   repeated group Item {
//     required int32 id = 2;
//     required bytes value = 3;
//   }
bool ParseMessageSetItem(absl::string_view msgset_item, int* id,
                         absl::string_view* value);

// Everything below this line is implementation-specific and may change at any
// time without notice.

namespace impl {

inline const char* VarintSkip32(const char* p) {
  const unsigned char* ptr = reinterpret_cast<const unsigned char*>(p);
  if (*ptr++ < 128) return reinterpret_cast<const char*>(ptr);
  if (*ptr++ < 128) return reinterpret_cast<const char*>(ptr);
  if (*ptr++ < 128) return reinterpret_cast<const char*>(ptr);
  if (*ptr++ < 128) return reinterpret_cast<const char*>(ptr);
  if (*ptr++ < 16) return reinterpret_cast<const char*>(ptr);
  return nullptr;  // value is too long to be a varint32
}

// Copied over from tensorflow/c/c_api.cc. However,
// using uint64_t instead of tensorflow::uint64.
// This code can be deleted if/when tensorflow uses uint64_t.
// This avoids an unnecessary copy.
// See https://github.com/tensorflow/tensorflow/pull/21042
inline const char* GetVarint64Ptr(const char* p, const char* limit,
                                  uint64_t* value) {
  uint64_t result = 0;
  for (uint32_t shift = 0; shift <= 63 && p < limit; shift += 7) {
    uint64_t byte = *(reinterpret_cast<const unsigned char*>(p));
    p++;
    if (byte & 128) {
      // More bytes are present
      result |= ((byte & 127) << shift);
    } else {
      result |= (byte << shift);
      *value = result;
      return reinterpret_cast<const char*>(p);
    }
  }
  return nullptr;
}

// Converts a runtime value of WireType to compile time constant and passes it
// as non-type template parameter to f.operator().
// The functor must have both result_type and kDefaultResultValue defined.
template <class F>
typename F::result_type DispatchByWireType(
    google::protobuf::internal::WireFormatLite::WireType type, F f) {
  switch (type) {
#define CASE_DISPATCH(TYPE)                               \
  case google::protobuf::internal::WireFormatLite::WIRETYPE_##TYPE: \
    return f.template                                     \
    operator()<google::protobuf::internal::WireFormatLite::WIRETYPE_##TYPE>()
    CASE_DISPATCH(VARINT);
    CASE_DISPATCH(FIXED64);
    CASE_DISPATCH(LENGTH_DELIMITED);
    CASE_DISPATCH(START_GROUP);
    CASE_DISPATCH(END_GROUP);
    CASE_DISPATCH(FIXED32);
#undef CASE_DISPATCH
  }
  // If we got here, the proto is malformed.
  return F::kDefaultResultValue;
}

// Attempts to skip a field given ptr where encoded field value starts. Returns
// a pointer just past field's content or nullptr if skip failed or crossed
// the limit.
const char* SkipField(const char* ptr, const char* limit,
                      google::protobuf::internal::WireFormatLite::WireType wire_type);

// Attempts to skip a group content. Returns a pointer just before END_GROUP tag
// or nullptr if skip failed or crossed the limit.
const char* SkipGroup(const char* ptr, const char* limit);

// Functions for decoding values from "raw" values. Raw data types are
// the following: uint32, uint64, StringPiece.
// Target value types are primitive values such as: int32, uint32, int64,
// uint64, bool, float, double, string, StringPiece and PackedValue.

inline void DecodeRawValue(uint32_t raw_value,
                           google::protobuf::FieldDescriptor::Type type,
                           uint32_t* value) {
  *value = raw_value;
}

inline void DecodeRawValue(uint64_t raw_value,
                           google::protobuf::FieldDescriptor::Type type,
                           uint64_t* value) {
  *value = raw_value;
}

inline void DecodeRawValue(absl::string_view raw_value,
                           google::protobuf::FieldDescriptor::Type type,
                           absl::string_view* value) {
  *value = raw_value;
}

inline void DecodeRawValue(absl::string_view raw_value,
                           google::protobuf::FieldDescriptor::Type type,
                           std::string* value) {
  value->assign(raw_value.data(), raw_value.size());
}

inline void DecodeRawValue(absl::string_view raw_value,
                           google::protobuf::FieldDescriptor::Type type,
                           PackedValues* value) {
  *value = PackedValues(type, raw_value);
}

inline void DecodeRawValue(uint32_t raw_value,
                           google::protobuf::FieldDescriptor::Type type, float* value) {
  *value = google::protobuf::internal::WireFormatLite::DecodeFloat(raw_value);
}

inline void DecodeRawValue(uint64_t raw_value,
                           google::protobuf::FieldDescriptor::Type type, double* value) {
  *value = google::protobuf::internal::WireFormatLite::DecodeDouble(raw_value);
}

inline void DecodeRawValue(uint32_t raw_value,
                           google::protobuf::FieldDescriptor::Type type, bool* value) {
  *value = (raw_value != 0);
}

inline void DecodeRawValue(uint32_t raw_value,
                           google::protobuf::FieldDescriptor::Type type, int32_t* value) {
  *value = (type == google::protobuf::FieldDescriptor::TYPE_SINT32)
               ? google::protobuf::internal::WireFormatLite::ZigZagDecode32(raw_value)
               : static_cast<int32_t>(raw_value);
}

inline void DecodeRawValue(uint64_t raw_value,
                           google::protobuf::FieldDescriptor::Type type, int64_t* value) {
  *value = (type == google::protobuf::FieldDescriptor::TYPE_SINT64)
               ? google::protobuf::internal::WireFormatLite::ZigZagDecode64(raw_value)
               : static_cast<int64_t>(raw_value);
}

// Defines raw value type for the given value type.
template <class T>
struct RawValueType;
template <>
struct RawValueType<bool> {
  typedef uint32_t type;
};
template <>
struct RawValueType<float> {
  typedef uint32_t type;
};
template <>
struct RawValueType<int32_t> {
  typedef uint32_t type;
};
template <>
struct RawValueType<uint32_t> {
  typedef uint32_t type;
};
template <>
struct RawValueType<int64_t> {
  typedef uint64_t type;
};
template <>
struct RawValueType<uint64_t> {
  typedef uint64_t type;
};
template <>
struct RawValueType<double> {
  typedef uint64_t type;
};
template <>
struct RawValueType<std::string> {
  typedef absl::string_view type;
};
template <>
struct RawValueType<absl::string_view> {
  typedef absl::string_view type;
};
template <>
struct RawValueType<PackedValues> {
  typedef absl::string_view type;
};

// Attempts to read a value starting from ptr. Reading will honor end limit.
// Returns new pointer just past value or nullptr if reading was not successful.
template <google::protobuf::internal::WireFormatLite::WireType WireType, class T>
inline const char* ReadRawValue(const char* ptr, const char* limit, T* value) {
  return nullptr;
}

template <>
inline const char*
ReadRawValue<google::protobuf::internal::WireFormatLite::WIRETYPE_FIXED32, uint32_t>(
    const char* ptr, const char* limit, uint32_t* value) {
  if (limit - ptr < 4) return nullptr;
  *value = absl::little_endian::Load32(ptr);
  return ptr + 4;
}

template <>
inline const char*
ReadRawValue<google::protobuf::internal::WireFormatLite::WIRETYPE_FIXED64, uint64_t>(
    const char* ptr, const char* limit, uint64_t* value) {
  if (limit - ptr < 8) return nullptr;
  *value = absl::little_endian::Load64(ptr);
  return ptr + 8;
}

template <>
inline const char*
ReadRawValue<google::protobuf::internal::WireFormatLite::WIRETYPE_VARINT, uint32_t>(
    const char* ptr, const char* limit, uint32_t* value) {
  // Special case where negative values in this encoding are actually
  // uint64 encoded as varint. So, decode uint64 and take first 4 bytes.
  uint64_t v;
  const char* a = GetVarint64Ptr(ptr, limit, &v);
  *value = static_cast<int32_t>(v);
  return a;
}

template <>
inline const char*
ReadRawValue<google::protobuf::internal::WireFormatLite::WIRETYPE_VARINT, uint64_t>(
    const char* ptr, const char* limit, uint64_t* value) {
  const char* result = GetVarint64Ptr(ptr, limit, value);
  return result;
}

template <>
inline const char*
ReadRawValue<google::protobuf::internal::WireFormatLite::WIRETYPE_LENGTH_DELIMITED,
             absl::string_view>(const char* ptr, const char* limit,
                                absl::string_view* value) {
  uint32_t length;
  ptr = tensorflow::core::GetVarint32Ptr(ptr, limit, &length);
  if (ptr == nullptr || limit - ptr < length) return nullptr;
  *value = absl::string_view(ptr, length);
  return ptr + length;
}

template <>
inline const char* ReadRawValue<
    google::protobuf::internal::WireFormatLite::WIRETYPE_START_GROUP, absl::string_view>(
    const char* ptr, const char* limit, absl::string_view* value) {
  const char* new_ptr = SkipGroup(ptr, limit);
  if (new_ptr == nullptr) return nullptr;
  *value = absl::string_view(ptr, new_ptr - ptr);
  // It is safe to skip 32 here because SkipGroup read past this skip already.
  return VarintSkip32(new_ptr);
}

template <class T>
struct ReadValueFn {
  typedef const char* result_type;
  static constexpr const char* kDefaultResultValue = nullptr;

  template <google::protobuf::internal::WireFormatLite::WireType WireType>
  result_type operator()() const {
    typename RawValueType<T>::type raw_value;
    const char* new_ptr = ReadRawValue<WireType>(ptr, end, &raw_value);
    if (new_ptr != nullptr) {
      DecodeRawValue(raw_value, field_type, value);
    }
    return new_ptr;
  }

  const char* ptr;
  const char* end;
  T* value;
  google::protobuf::FieldDescriptor::Type field_type;
};

// Attempts to read a value starting from ptr. Returns a pointer just past value
// content or nullptr if reading was unsuccessful. Will honor end limit.
template <class T>
const char* ReadValue(const char* ptr, const char* end,
                      google::protobuf::internal::WireFormatLite::WireType wire_type,
                      google::protobuf::FieldDescriptor::Type field_type, T* value) {
  return DispatchByWireType(wire_type,
                            ReadValueFn<T>{ptr, end, value, field_type});
}

// Attempts to read next field's tag. Returns a pointer just past tag value or
// nullptr if reading was unsuccessful. Will honor end limit.
inline const char* ReadTag(const char* ptr, const char* end, uint32_t* tag) {
  return tensorflow::core::GetVarint32Ptr(ptr, end, tag);
}

}  // namespace impl

bool StreamingProtoReader::Next(int* field_number) {
  const char* new_ptr = ptr_;

  // Skip field if needed.
  if (content_available_) {
    new_ptr = impl::SkipField(new_ptr, end_, wire_type_);
    if (new_ptr == nullptr) return false;
    ptr_ = new_ptr;
  }

  // Read a tag for the next field.
  uint32_t tag;
  new_ptr = impl::ReadTag(new_ptr, end_, &tag);
  if (new_ptr == nullptr) return false;

  // Decode field number and wire type.
  content_available_ = true;
  ptr_ = new_ptr;
  wire_type_ = google::protobuf::internal::WireFormatLite::GetTagWireType(tag);
  *field_number = google::protobuf::internal::WireFormatLite::GetTagFieldNumber(tag);
  return true;
}

template <class T>
bool StreamingProtoReader::ReadValue(google::protobuf::FieldDescriptor::Type field_type,
                                     T* value) {
  if (!content_available_) return false;
  const char* new_ptr =
      impl::ReadValue(ptr_, end_, wire_type_, field_type, value);
  if (new_ptr == nullptr) return false;

  content_available_ = false;
  ptr_ = new_ptr;
  return true;
}

template <class T>
PackedValuesReader<T>::PackedValuesReader(const PackedValues& values)
    : ptr_(values.data().begin()),
      end_(values.data().end()),
      field_type_(values.field_type()),
      wire_type_(google::protobuf::internal::WireFormatLite::WireTypeForFieldType(
          static_cast<google::protobuf::internal::WireFormatLite::FieldType>(
              field_type_))) {}

template <class T>
bool PackedValuesReader<T>::Next(T* value) {
  if (ptr_ == nullptr) return false;
  ptr_ = impl::ReadValue(ptr_, end_, wire_type_, field_type_, value);
  return ptr_ != nullptr;
}

}  // namespace struct2tensor

#endif  // STRUCT2TENSOR_PY_KERNELS_STREAMING_PROTO_READER_H_
