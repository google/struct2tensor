package(
    default_visibility = ["@//:__subpackages__"],
)

load("//struct2tensor:struct2tensor.bzl", "s2t_proto_library", "s2t_proto_library_cc", "s2t_proto_library_py", "s2t_pytype_library")

licenses(["notice"])  # Apache 2.0

exports_files([
    "LICENSE",
    "__init__.py",
    "expression_test_util.py",
    "prensor_test_util.py",
])

s2t_proto_library(
    name = "test_proto",
    srcs = ["test.proto"],
    cc_api_version = 2,
)

s2t_proto_library_cc(
    name = "test_cc_proto",
    deps = [
        ":test_proto",
    ],
)

s2t_proto_library_py(
    name = "test_py_pb2",
    srcs = ["test.proto"],
    api_version = 2,
    proto_library = "test_proto",
    deps = [],
)

s2t_proto_library(
    name = "test_any_proto",
    srcs = ["test_any.proto"],
    cc_api_version = 2,
    deps = ["@com_google_protobuf//:cc_wkt_protos"],
)

s2t_proto_library_cc(
    name = "test_any_cc_proto",
    deps = [":test_any_proto"],
)

s2t_proto_library_py(
    name = "test_any_py_pb2",
    srcs = ["test_any.proto"],
    api_version = 2,
    proto_library = "test_any_proto",
    deps = [],
)

s2t_proto_library(
    name = "test_map_proto",
    srcs = ["test_map.proto"],
    cc_api_version = 2,
    deps = [],
)

s2t_proto_library_cc(
    name = "test_map_cc_proto",
    deps = [":test_map_proto"],
)

s2t_proto_library_py(
    name = "test_map_py_pb2",
    srcs = ["test_map.proto"],
    proto_library = "test_map_proto",
    deps = [],
)

s2t_proto_library(
    name = "test_extension_proto",
    srcs = ["test_extension.proto"],
    deps = [":test_proto"],
)

s2t_proto_library_py(
    name = "test_extension_py_pb2",
    srcs = ["test_extension.proto"],
    oss_deps = [":test_py_pb2"],
    proto_library = "test_extension_proto",
)

s2t_proto_library(
    name = "dependent_test_proto",
    srcs = ["dependent_test.proto"],
    deps = [":test_proto"],
)

s2t_proto_library_py(
    name = "dependent_test_py_pb2",
    srcs = ["dependent_test.proto"],
    oss_deps = [":test_py_pb2"],
    proto_library = "dependent_test_proto",
)

s2t_pytype_library(
    name = "expression_test_util",
    testonly = 1,
    srcs = [
        "expression_test_util.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//struct2tensor:expression",
        "//struct2tensor:path",
    ],
)

s2t_pytype_library(
    name = "prensor_test_util",
    testonly = 1,
    srcs = [
        "prensor_test_util.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//struct2tensor:expression",
        "//struct2tensor:path",
    ],
)
