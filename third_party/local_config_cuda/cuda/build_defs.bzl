def if_cuda_is_configured(x, no_cuda = []):
    return no_cuda

def is_cuda_configured():
    return False

def if_cuda_newer_than(wanted_ver, if_true, if_false = []):
    return if_false

def if_cuda(if_true, if_false = []):
    return if_false

def cuda_library(**kwargs):
    native.cc_library(**kwargs)

def if_cuda_exec(if_true, if_false = []):
    return if_false

def cuda_gpu_architectures():
    return []
