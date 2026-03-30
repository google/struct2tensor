def cuda_library(name, **kwargs):
    native.cc_library(name = name, **kwargs)

def if_cuda_exec(if_true, if_false = []):
    return if_false

def if_cuda(if_true, if_false = []):
    return if_false

def if_cuda_is_configured(if_true, if_false = []):
    return if_false

def if_cuda_newer_than(wanted_ver, if_true, if_false = []):
    return if_false

def cuda_gpu_architectures():
    return []

def cuda_default_copts():
    return []

