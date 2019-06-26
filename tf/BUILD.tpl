"""Template for a BUILD file that defines TF library tagets.
Dynamic libraries and the PIP package in Struct2Tensor depend upon Tensorflow
through these targets. The template is populated by tf_configure.bzl by
running ../configure.sh.

Static libraries for building a tensorflow serving binary do not depend upon
this.
"""


package(default_visibility = ["//visibility:public"])

cc_library(
    name = "tf_header_lib",
    hdrs = [":tf_header_include"],
    includes = ["include"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "libtensorflow_framework",
    srcs = [":libtensorflow_framework.so"],
    visibility = ["//visibility:public"],
)

%{TF_HEADER_GENRULE}
%{TF_SHARED_LIBRARY_GENRULE}
