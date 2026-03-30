package(default_visibility = ["//visibility:public"])

filegroup(
    name = "cuda_headers",
    srcs = [],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "using_nvcc",
    values = {"define": "using_nvcc=true"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "using_clang",
    values = {"define": "using_clang=true"},
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cuda",
)
