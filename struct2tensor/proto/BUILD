package(default_visibility = ["@//:__subpackages__"])

licenses(["notice"])

load("//struct2tensor:struct2tensor.bzl", "s2t_proto_library", "s2t_proto_library_cc", "s2t_proto_library_py")

s2t_proto_library(
    name = "query_metadata_proto",
    srcs = ["query_metadata.proto"],
    cc_api_version = 2,
    deps = ["@com_github_tensorflow_metadata//tensorflow_metadata/proto/v0:cc_metadata_v0_proto_cc"],
)

s2t_proto_library_cc(
    name = "query_metadata_cc_proto",
    deps = [":query_metadata_proto"],
)

s2t_proto_library_py(
    name = "query_metadata_py_pb2",
    srcs = ["query_metadata.proto"],
    api_version = 2,
    oss_deps = [
        "@com_github_tensorflow_metadata//tensorflow_metadata/proto/v0:py_metadata_v0_proto_py",
    ],
    proto_library = "query_metadata_proto",
)
