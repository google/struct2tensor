load("//struct2tensor:struct2tensor.bzl", "s2t_dynamic_binary", "s2t_dynamic_library", "s2t_gen_op_wrapper_py", "s2t_pytype_library")
load(
    "@org_tensorflow//tensorflow:tensorflow.bzl",
    "tf_gen_op_wrapper_cc",
)

# TODO(martinz): Add back integration_with_servo_test.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files([
    "LICENSE",
    "__init__.py",
])

# NOTE: s2t_dynamic_binary is a NOOP internally (does not generate a target).
s2t_dynamic_binary(
    name = "_run_length_before_op.so",
    deps = [
        ":run_length_before_op_dynamic",
        "//struct2tensor/kernels:run_length_before_op_dynamic",
    ],
)

s2t_dynamic_binary(
    name = "_decode_proto_sparse_op.so",
    deps = [
        ":decode_proto_sparse_op_dynamic",
        "//struct2tensor/kernels:decode_proto_sparse_op_dynamic",
    ],
)

s2t_dynamic_binary(
    name = "_decode_proto_map_op.so",
    deps = [
        ":decode_proto_map_op_dynamic",
        "//struct2tensor/kernels:decode_proto_map_op_dynamic",
    ],
)

s2t_dynamic_binary(
    name = "_equi_join_indices_op.so",
    deps = [
        ":equi_join_indices_op_dynamic",
        "//struct2tensor/kernels:equi_join_indices_op_dynamic",
    ],
)

s2t_pytype_library(
    name = "struct2tensor_ops",
    srcs = ["struct2tensor_ops.py"],
    deps = [
        ":file_descriptor_set",
        ":gen_decode_proto_map_op_py",
        ":gen_decode_proto_sparse_py",
        ":gen_equi_join_indices_py",
        ":gen_run_length_before_py",
        "//struct2tensor:path",
        "@com_google_protobuf//:protobuf_python",
    ],
)

s2t_pytype_library(
    name = "file_descriptor_set",
    srcs = ["file_descriptor_set.py"],
    deps = [
        "//struct2tensor:path",
        "@com_google_protobuf//:protobuf_python",
    ],
)

cc_library(
    name = "run_length_before",
    srcs = [
        "run_length_before_op.cc",
    ],
    deps = [
        "@org_tensorflow//tensorflow/core:framework",
    ],
    alwayslink = 1,
)

s2t_dynamic_library(
    name = "run_length_before_op_dynamic",
    srcs = [
        "run_length_before_op.cc",
    ],
)

cc_library(
    name = "decode_proto_sparse",
    srcs = [
        "decode_proto_sparse_op.cc",
    ],
    deps = [
        "@org_tensorflow//tensorflow/core:framework",
    ],
    alwayslink = 1,
)

s2t_dynamic_library(
    name = "decode_proto_sparse_op_dynamic",
    srcs = [
        "decode_proto_sparse_op.cc",
    ],
)

cc_library(
    name = "decode_proto_map_op",
    srcs = [
        "decode_proto_map_op.cc",
    ],
    deps = [
        "@org_tensorflow//tensorflow/core:framework",
    ],
    alwayslink = 1,
)

s2t_dynamic_library(
    name = "decode_proto_map_op_dynamic",
    srcs = [
        "decode_proto_map_op.cc",
    ],
)

cc_library(
    name = "equi_join_indices",
    srcs = ["equi_join_indices_op.cc"],
    deps = [
        "@org_tensorflow//tensorflow/core:framework",
    ],
    alwayslink = 1,
)

s2t_dynamic_library(
    name = "equi_join_indices_op_dynamic",
    srcs = ["equi_join_indices_op.cc"],
)

s2t_gen_op_wrapper_py(
    name = "gen_run_length_before_py",
    out = "gen_run_length_before.py",
    dynamic_library = ":_run_length_before_op.so",
    static_library = ":run_length_before",
)

s2t_gen_op_wrapper_py(
    name = "gen_equi_join_indices_py",
    out = "gen_equi_join_indices.py",
    dynamic_library = ":_equi_join_indices_op.so",
    static_library = ":equi_join_indices",
)

s2t_gen_op_wrapper_py(
    name = "gen_decode_proto_sparse_py",
    out = "gen_decode_proto_sparse.py",
    dynamic_library = ":_decode_proto_sparse_op.so",
    static_library = ":decode_proto_sparse",
)

s2t_gen_op_wrapper_py(
    name = "gen_decode_proto_map_op_py",
    out = "gen_decode_proto_map_op.py",
    dynamic_library = ":_decode_proto_map_op.so",
    static_library = ":decode_proto_map_op",
)

cc_library(
    name = "parquet_dataset_op",
    srcs = ["parquet_dataset_op.cc"],
    deps = [
        "@org_tensorflow//tensorflow/core:dataset_ops_op_lib",
        "@org_tensorflow//tensorflow/core:framework",
    ],
    alwayslink = 1,
)

s2t_dynamic_library(
    name = "parquet_dataset_op_dynamic",
    srcs = [
        "parquet_dataset_op.cc",
    ],
)

s2t_dynamic_binary(
    name = "_parquet_dataset_op.so",
    deps = [
        ":parquet_dataset_op_dynamic",
        "//struct2tensor/kernels/parquet:parquet_dataset_kernel_dynamic",
    ],
)

s2t_gen_op_wrapper_py(
    name = "gen_parquet_dataset_py",
    out = "gen_parquet_dataset.py",
    dynamic_library = ":_parquet_dataset_op.so",
    static_library = ":parquet_dataset_op",
)

# Contains only registrations of Prensor ops. Programs that build but do not run
# Prensor graphs should depend on this target.
# Open source packages should depend upon this and
# struct2tensor/kernels:struct2tensor_kernels.
cc_library(
    name = "struct2tensor_op_registrations",
    visibility = ["//visibility:public"],
    deps = [
        ":decode_proto_map_op",
        ":decode_proto_sparse",
        ":equi_join_indices",
        ":run_length_before",
    ],
    alwayslink = 1,
)

cc_test(
    name = "op_kernel_registration_test",
    srcs = ["op_kernel_registration_test.cc"],
    deps = [
        ":parquet_dataset_op",  # build_cleaner: keep
        ":struct2tensor_op_registrations",  # build_cleaner: keep
        "//struct2tensor/kernels:struct2tensor_kernels",  # build_cleaner: keep
        "//struct2tensor/kernels/parquet:parquet_dataset_kernel",  # build_cleaner: keep
        "@com_google_googletest//:gtest_main",
        "@org_tensorflow//tensorflow/core:framework",
    ],
)
