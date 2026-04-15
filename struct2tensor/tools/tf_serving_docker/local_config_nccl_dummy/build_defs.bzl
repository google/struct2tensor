def cuda_rdc_library(name, srcs = [], deps = [], **kwargs):
    native.cc_library(
        name = name,
        srcs = srcs,
        deps = deps,
        **kwargs
    )
