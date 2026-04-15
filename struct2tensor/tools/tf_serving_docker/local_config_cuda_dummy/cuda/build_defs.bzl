def if_cuda(if_true, if_false = []):
    return if_false

def cuda_library(copts = [], tags = [], deps = [], **kwargs):
    native.cc_library(
        copts = copts,
        tags = tags,
        deps = deps,
        **kwargs
    )

def if_cuda_exec(if_true, if_false = []):
    return if_false

def cuda_header_library(name, hdrs, **kwargs):
    native.cc_library(name = name, hdrs = hdrs, **kwargs)

def cuda_cc_test(copts = [], **kwargs):
    native.cc_test(copts = copts, **kwargs)

def if_cuda_is_configured(x, no_cuda = []):
    return no_cuda

def if_cuda_newer_than(version, if_true, if_false = []):
    return if_false

def cuda_gpu_architectures():
    return []

def cuda_default_copts():
    return []
