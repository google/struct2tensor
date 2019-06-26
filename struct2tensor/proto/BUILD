package(default_visibility = ["@//:__subpackages__"])

licenses(["notice"])  # Apache 2.0

load("//struct2tensor:struct2tensor.bzl", "s2t_proto_library", "s2t_proto_library_cc", "s2t_proto_library_py")

s2t_proto_library(
    name = "query_metadata_proto",
    srcs = ["query_metadata.proto"],
    cc_api_version = 2,
    visibility = ["//learning/tfx/pipeline/example_validation:__subpackages__"],
    deps = ["@com_github_tensorflow_metadata//tensorflow_metadata/proto/v0:metadata_v0_proto_cc_pb2"],
)

s2t_proto_library_cc(
    name = "query_metadata_cc_proto",
    visibility = ["//learning/tfx/pipeline/example_validation:__subpackages__"],
    deps = [":query_metadata_proto"],
)

s2t_proto_library_py(
    name = "query_metadata_py_pb2",
    srcs = ["query_metadata.proto"],
    api_version = 2,
    oss_deps = [
        "@com_github_tensorflow_metadata//tensorflow_metadata/proto/v0:metadata_v0_proto_py_pb2",
    ],
    proto_library = "query_metadata_proto",
)
