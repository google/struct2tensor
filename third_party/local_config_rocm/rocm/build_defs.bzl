def if_rocm(if_true, if_false = []):
    return if_false

def get_rbe_amdgpu_pool(**kwargs):
    return ""

def is_rocm_configured():
    return False

def rocm_copts(**kwargs):
    return []

def if_rocm_is_configured(x, no_rocm = []):
    return no_rocm

def if_cuda_or_rocm(if_true, if_false = []):
    return if_false

def if_gpu_is_configured(if_true, if_false = []):
    return if_false

def rocm_gpu_architectures():
    return []

def if_rocm_hipblaslt(if_true, if_false = []):
    return if_false
