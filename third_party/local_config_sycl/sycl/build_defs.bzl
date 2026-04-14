load("@rules_cc//cc:cc_library.bzl", "cc_library")

def if_sycl_is_configured(if_true, if_false = []):
    return if_false

def sycl_library(name, **kwargs):
    cc_library(name = name)
