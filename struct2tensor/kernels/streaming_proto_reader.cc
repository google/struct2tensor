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
#include "struct2tensor/kernels/streaming_proto_reader.h"

#include "absl/base/integral_types.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/wire_format_lite.h"
#include "tensorflow/core/lib/core/coding.h"

namespace struct2tensor {
namespace {
using google::protobuf::FieldDescriptor;
using google::protobuf::internal::WireFormatLite;

const char* VarintSkip64(const char* p) {
  const unsigned char* ptr = reinterpret_cast<const unsigned char*>(p);
  if (*ptr++ < 128) return reinterpret_cast<const char*>(ptr);
  if (*ptr++ < 128) return reinterpret_cast<const char*>(ptr);
  if (*ptr++ < 128) return reinterpret_cast<const char*>(ptr);
  if (*ptr++ < 128) return reinterpret_cast<const char*>(ptr);
  if (*ptr++ < 128) return reinterpret_cast<const char*>(ptr);
  if (*ptr++ < 128) return reinterpret_cast<const char*>(ptr);
  if (*ptr++ < 128) return reinterpret_cast<const char*>(ptr);
  if (*ptr++ < 128) return reinterpret_cast<const char*>(ptr);
  if (*ptr++ < 128) return reinterpret_cast<const char*>(ptr);
  if (*ptr++ < 2) return reinterpret_cast<const char*>(ptr);
  return nullptr;  // value is too long to be a varint64
}

const char* VarintSkip64WithLimit(const char* ptr, const char* limit) {
  if (limit - ptr >= tensorflow::core::kMaxVarint64Bytes)
    return VarintSkip64(ptr);
  const unsigned char* p = reinterpret_cast<const unsigned char*>(ptr);
  const unsigned char* l = reinterpret_cast<const unsigned char*>(limit);
  if (p >= l) return nullptr;
  if (*p++ < 128) return reinterpret_cast<const char*>(p);
  if (p >= l) return nullptr;
  if (*p++ < 128) return reinterpret_cast<const char*>(p);
  if (p >= l) return nullptr;
  if (*p++ < 128) return reinterpret_cast<const char*>(p);
  if (p >= l) return nullptr;
  if (*p++ < 128) return reinterpret_cast<const char*>(p);
  if (p >= l) return nullptr;
  if (*p++ < 128) return reinterpret_cast<const char*>(p);
  if (p >= l) return nullptr;
  if (*p++ < 128) return reinterpret_cast<const char*>(p);
  if (p >= l) return nullptr;
  if (*p++ < 128) return reinterpret_cast<const char*>(p);
  if (p >= l) return nullptr;
  if (*p++ < 128) return reinterpret_cast<const char*>(p);
  if (p >= l) return nullptr;
  if (*p++ < 128) return reinterpret_cast<const char*>(p);
  if (p >= l) return nullptr;
  if (*p++ < 128) return reinterpret_cast<const char*>(p);
  return nullptr;  // value is too long to be a varint64
}

// Attempts to skip a field given ptr where encoded field value starts. Returns
// a pointer just past field's content or nullptr if skip failed or crossed
// end limit.
template <WireFormatLite::WireType WireType>
const char* SkipField(const char* ptr, const char* limit);

template <>
const char* SkipField<WireFormatLite::WIRETYPE_FIXED32>(const char* ptr,
                                                        const char* limit) {
  return limit - ptr < 4 ? nullptr : ptr + 4;
}

template <>
const char* SkipField<WireFormatLite::WIRETYPE_FIXED64>(const char* ptr,
                                                        const char* limit) {
  return limit - ptr < 8 ? nullptr : ptr + 8;
}

template <>
const char* SkipField<WireFormatLite::WIRETYPE_VARINT>(const char* ptr,
                                                       const char* limit) {
  ptr = VarintSkip64WithLimit(ptr, limit);
  return ptr > limit ? nullptr : ptr;
}

template <>
const char* SkipField<WireFormatLite::WIRETYPE_LENGTH_DELIMITED>(
    const char* ptr, const char* limit) {
  uint32_t length;
  ptr = tensorflow::core::GetVarint32Ptr(ptr, limit, &length);
  if (ptr == nullptr) return nullptr;
  return limit - ptr < length ? nullptr : ptr + length;
}

template <>
const char* SkipField<WireFormatLite::WIRETYPE_START_GROUP>(const char* ptr,
                                                            const char* limit) {
  ptr = impl::SkipGroup(ptr, limit);
  if (ptr == nullptr) return nullptr;
  // It is safe to skip 32 here because SkipGroup read past this skip already.
  return impl::VarintSkip32(ptr);
}

template <>
const char* SkipField<WireFormatLite::WIRETYPE_END_GROUP>(const char* ptr,
                                                          const char* limit) {
  return ptr;
}

struct SkipFieldFn {
  typedef const char* result_type;
  static constexpr const char* kDefaultResultValue = nullptr;

  const char* ptr;
  const char* limit;

  template <WireFormatLite::WireType WireType>
  result_type operator()() const {
    return SkipField<WireType>(ptr, limit);
  }
};

}  // namespace

namespace impl {

const char* SkipField(const char* ptr, const char* limit,
                      WireFormatLite::WireType wire_type) {
  return DispatchByWireType(wire_type, SkipFieldFn{ptr, limit});
}

// Reads group content without END_GROUP tag or returns nullptr on failure.
// Ignores END_GROUP tag numbers and relies only on balancing of START_GROUP and
// END_GROUP.
//
// Since groups are encoded with bracketing-pairs of wire tags, we must
// interpret their contents in order to skip them.  This means the input can
// force us to read arbitrarily-deeply-nested groups regardless of the message
// type being parsed, so we must be able to parse nested groups without
// introducing more stack frames, or risk stack overflows.  Do this by counting
// nesting depth in a wide integer rather than going through SkipField+SkipGroup
// recursively.
const char* SkipGroup(const char* ptr, const char* limit) {
  uint32_t tag = 0;

  uint64_t group_depth = 1;

  while (true) {
    const char* new_ptr = impl::ReadTag(ptr, limit, &tag);
    if (new_ptr == nullptr) break;
    WireFormatLite::WireType wire_type = WireFormatLite::GetTagWireType(tag);
    switch (wire_type) {
      case WireFormatLite::WIRETYPE_END_GROUP:
        if (--group_depth == 0) return ptr;  // Finished the top-level group.
        ptr = new_ptr;  // Consume the nested group's END_GROUP and continue.
        break;
      case WireFormatLite::WIRETYPE_START_GROUP:
        ++group_depth;  // Entered a nested group; keep skipping stuff.
        ptr = new_ptr;  // Consume the START_GROUP and continue.
        break;
      default:
        ptr = SkipField(new_ptr, limit, wire_type);
    }
    if (ptr == nullptr) break;
  }
  return nullptr;
}

}  // namespace impl

StreamingProtoReader::StreamingProtoReader(absl::string_view proto)
    : ptr_(proto.data()),
      end_(proto.end()),
      wire_type_(WireFormatLite::WIRETYPE_VARINT),
      content_available_(false) {}

bool ParseMessageSetItem(absl::string_view msgset_item, int* id,
                         absl::string_view* value) {
  // id and value fields may go in arbitrary order. Therefore, there is proper
  // loop instead of implying specific order.
  int field_number = 0;
  bool seen_id = false;
  bool seen_content = false;
  for (StreamingProtoReader reader(msgset_item); reader.Next(&field_number);) {
    switch (field_number) {
      case WireFormatLite::kMessageSetTypeIdNumber:
        if (seen_id || !reader.ReadValue(FieldDescriptor::TYPE_INT32, id)) {
          return false;
        }
        seen_id = true;
        break;
      case WireFormatLite::kMessageSetMessageNumber:
        if (seen_content ||
            !reader.ReadValue(FieldDescriptor::TYPE_BYTES, value)) {
          return false;
        }
        seen_content = true;
        break;
      default:
        continue;
    }
    if (seen_id && seen_content) return true;
  }
  return false;
}

}  // namespace struct2tensor
