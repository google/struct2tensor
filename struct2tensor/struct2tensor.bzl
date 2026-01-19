# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bazel macros used in OSS."""

def _py_proto_library_impl(ctx):
    """Implementation of py_proto_library rule."""
    proto_deps = ctx.attr.deps

    # Separate proto and Python dependencies
    all_sources = []
    py_infos = []

    for dep in proto_deps:
        if ProtoInfo in dep:
            # It's a proto_library - collect proto sources
            all_sources.extend(dep[ProtoInfo].direct_sources)
        elif PyInfo in dep:
            # It's already a py_library - collect its PyInfo for passthrough
            py_infos.append(dep[PyInfo])

    # Filter to only include sources from the workspace (not external packages)
    # We can only declare outputs in our own package
    workspace_sources = []
    for src in all_sources:
        # Filter out external sources (they start with external/ or ..)
        if not src.short_path.startswith("external/") and not src.short_path.startswith("../"):
            workspace_sources.append(src)

    # Generate Python output files from proto sources
    py_outputs = []
    for proto_src in workspace_sources:
        # Use just the basename to avoid path issues
        basename = proto_src.basename[:-6]  # Remove .proto
        py_file = ctx.actions.declare_file(basename + "_pb2.py")
        py_outputs.append(py_file)

    if py_outputs:
        # Build proto_path arguments for protoc
        # We need to include paths for workspace root and external dependencies
        proto_path_args = []

        # Add current directory to find workspace proto files
        proto_path_args.append("--proto_path=.")

        # Collect proto_path entries from all transitive dependencies
        # Use dictionary as a set (Starlark doesn't have set type)
        proto_paths = {".": True}

        # Also add directories of workspace sources so imports like "any.proto"
        # (in the same folder) resolve correctly.
        for ws in workspace_sources:
            ws_dir = "/".join(ws.short_path.split("/")[:-1])
            if ws_dir and ws_dir not in proto_paths:
                proto_paths[ws_dir] = True
                proto_path_args.append("--proto_path=" + ws_dir)

        for dep in proto_deps:
            if ProtoInfo in dep:
                # Add proto_source_root if available
                if hasattr(dep[ProtoInfo], 'proto_source_root'):
                    root = dep[ProtoInfo].proto_source_root
                    if root and root not in proto_paths:
                        proto_paths[root] = True
                        proto_path_args.append("--proto_path=" + root)

                # Also derive from file paths for more coverage
                for src in dep[ProtoInfo].transitive_sources.to_list():
                    # Use the directory containing the proto file's import root
                    # For external/com_google_protobuf/src/google/protobuf/any.proto,
                    # we want external/com_google_protobuf/src
                    if src.path.startswith("external/com_google_protobuf/"):
                        proto_path = "external/com_google_protobuf/src"
                        if proto_path not in proto_paths:
                            proto_paths[proto_path] = True
                            proto_path_args.append("--proto_path=" + proto_path)
                    elif src.path.startswith("external/"):
                        # For other external repos like tensorflow_metadata
                        # Extract external/repo_name
                        parts = src.path.split("/")
                        if len(parts) >= 2:
                            proto_path = "/".join(parts[:2])
                            if proto_path not in proto_paths:
                                proto_paths[proto_path] = True
                                proto_path_args.append("--proto_path=" + proto_path)

                    # Also add Bazel root paths
                    if src.root.path and src.root.path not in proto_paths:
                        proto_paths[src.root.path] = True
                        proto_path_args.append("--proto_path=" + src.root.path)

# Build list of proto file paths - only include workspace sources
        proto_file_args = []
        for src in workspace_sources:
            proto_file_args.append(src.short_path)

        # Run protoc to generate Python files
        # Use ctx.bin_dir.path as the output directory root
        output_root = ctx.bin_dir.path

        ctx.actions.run(
            # Include workspace sources plus all transitive dependencies for imports
            inputs = depset(direct = workspace_sources, transitive = [
                dep[ProtoInfo].transitive_sources for dep in proto_deps if ProtoInfo in dep
            ]),
            outputs = py_outputs,
            executable = ctx.executable._protoc,
            arguments = [
                "--python_out=" + output_root,
            ] + proto_path_args + proto_file_args,
            mnemonic = "ProtocPython",
        )

    # Collect transitive sources from both generated files and Python deps
    all_transitive_sources = [depset(py_outputs)]
    all_imports = [depset([ctx.bin_dir.path])] if py_outputs else []

    for py_info in py_infos:
        all_transitive_sources.append(py_info.transitive_sources)
        if hasattr(py_info, 'imports'):
            all_imports.append(py_info.imports)

    # Return PyInfo provider so this can be used as a py_library dependency
    # Merge proto-generated files with passthrough Python dependencies
    return [
        DefaultInfo(files = depset(py_outputs)),
        PyInfo(
            transitive_sources = depset(transitive = all_transitive_sources),
            imports = depset(transitive = all_imports),
            has_py2_only_sources = False,
            has_py3_only_sources = True,
        ),
    ]

_py_proto_library_rule = rule(
    implementation = _py_proto_library_impl,
    attrs = {
        "deps": attr.label_list(
            providers = [[ProtoInfo], [PyInfo]],  # Accept either ProtoInfo OR PyInfo
            doc = "Proto library or Python library dependencies",
        ),
        "_protoc": attr.label(
            default = "@com_google_protobuf//:protoc",
            executable = True,
            cfg = "exec",
        ),
    },
    provides = [PyInfo],
)

# Wrapper for cc_proto_library to maintain compatibility with old Protobuf 3.x API
def cc_proto_library(
        name,
        srcs = [],
        deps = [],
        cc_libs = [],
        protoc = None,
        default_runtime = None,
        use_grpc_plugin = None,
        testonly = 0,
        visibility = None,
        **kwargs):
    """Wrapper for cc_proto_library that works with Protobuf 4.x."""
    _ignore = [cc_libs, protoc, default_runtime, use_grpc_plugin, kwargs]

    # Create proto_library first
    native.proto_library(
        name = name + "_proto",
        srcs = srcs,
        deps = [d + "_proto" if not d.startswith("@") else d for d in deps],
        testonly = testonly,
        visibility = visibility,
    )

    # Create cc_proto_library that depends on proto_library
    native.cc_proto_library(
        name = name,
        deps = [":" + name + "_proto"],
        testonly = testonly,
        visibility = visibility,
    )

def s2t_pytype_library(
        name,
        srcs = [],
        deps = [],
        srcs_version = "PY3ONLY",
        testonly = False):
    """Python library that automatically wraps proto_library deps with PyInfo.

    This wrapper wraps all dependencies with our custom py_proto_library_rule.
    Dependencies that don't provide ProtoInfo will fail with a clear error.
    Dependencies that do provide ProtoInfo (proto_library targets) will get PyInfo.
    """
    # Process dependencies to wrap them all with our custom rule
    processed_deps = []
    for dep in deps:
        # Skip protobuf_python - it's already a proper Python library
        if dep == "@com_google_protobuf//:protobuf_python":
            processed_deps.append(dep)
            continue

        # Create a safe wrapper name for this dependency
        safe_dep_name = dep.replace(":", "_").replace("//", "").replace("/", "_").replace("@", "").replace("-", "_").replace(".", "_")
        wrapper_name = name + "_proto_wrapper_" + safe_dep_name

        # Wrap all dependencies with our custom py_proto_library rule
        # If the dep provides ProtoInfo, this will work and provide PyInfo
        # If it doesn't provide ProtoInfo, it will fail with a clear error
        _py_proto_library_rule(
            name = wrapper_name,
            deps = [dep],
            testonly = testonly,
        )
        processed_deps.append(":" + wrapper_name)

    native.py_library(
        name = name,
        srcs = srcs,
        deps = processed_deps,
        testonly = testonly,
    )


def s2t_proto_library(
        name,
        srcs = [],
        has_services = False,
        deps = [],
        visibility = None,
        testonly = 0,
        cc_grpc_version = None):
    """Opensource proto_library.

    Args:
      name: the name of the build target.
      srcs: .proto file sources.
      has_services: no effect
      deps: dependencies
      visibility: visibility constraints
      testonly: if true, only use in tests.
      cc_grpc_version: If set, use grpc plugin.
    """
    _ignore = [has_services]
    native.filegroup(
        name = name + "_proto_srcs",
        srcs = srcs,
        testonly = testonly,
    )

    # Create a native proto_library for Python generation
    # This is needed by s2t_proto_library_py
    proto_lib_deps = [d + "_proto" if not d.startswith("@") else d for d in deps]
    native.proto_library(
        name = name + "_proto",
        srcs = srcs,
        deps = proto_lib_deps,
        visibility = visibility,
        testonly = testonly,
    )

    # Create cc_proto_library that depends on the proto_library we just created
    # Don't use our cc_proto_library wrapper to avoid duplicate proto_library creation
    native.cc_proto_library(
        name = name,
        deps = [":" + name + "_proto"],
        testonly = testonly,
        visibility = visibility,
    )

DYNAMIC_COPTS = [
    "-pthread",
]

DYNAMIC_DEPS = ["@local_config_tf//:libtensorflow_framework", "@local_config_tf//:tf_header_lib"]

def s2t_dynamic_binary(name, deps):
    """Creates a .so file intended for linking with tensorflow_framework.so."""
    native.cc_binary(
        name = name,
        copts = DYNAMIC_COPTS,
        linkshared = 1,
        deps = deps + DYNAMIC_DEPS,
    )

def s2t_dynamic_library(
        name,
        srcs,
        deps = None):
    """Creates a static library intended for linking with tensorflow_framework.so."""
    true_deps = [] if deps == None else deps
    native.cc_library(
        name = name,
        srcs = srcs,
        alwayslink = 1,
        copts = DYNAMIC_COPTS,
        deps = true_deps + DYNAMIC_DEPS,
    )

def s2t_gen_op_wrapper_py(
        name,
        out,
        static_library,
        dynamic_library,
        visibility = None):
    """Applies gen_op_wrapper_py externally.

    Instead of a static library, links to a dynamic library.
    Instead of generating a file, one is provided.

    Args:
      name: name of the target
      out: a file that must be provided. Included as source.
      static_library: a static library (ignored).
      dynamic_library: a dynamic library included as data.
      visibility: The visibility attribute on a rule controls whether the rule can be used by other packages. Rules are always visible to other rules declared in the same package.
    """
    native.py_library(
        name = name,
        srcs = ([
            out,
        ]),
        data = [
            dynamic_library,
        ],
        srcs_version = "PY3ONLY",
        visibility = visibility,
    )

def s2t_proto_library_cc(
        name,
        srcs = [],
        has_services = False,
        deps = [],
        visibility = None,
        testonly = 0,
        cc_grpc_version = None):
    """Opensource cc_proto_library.

    Args:
        name: name of library
        srcs: .proto sources
        has_services: no effect
        deps: dependencies
        visibility: visibility constraints
        testonly: if true, can only be used in tests.
        cc_grpc_version: if set, use_grpc_version is True.
    """
    _ignore = [has_services]
    native.filegroup(
        name = name + "_proto_srcs",
        srcs = srcs,
        testonly = testonly,
    )

    use_grpc_plugin = None
    if cc_grpc_version:
        use_grpc_plugin = True
    cc_proto_library(
        name = name,
        srcs = srcs,
        deps = deps,
        cc_libs = ["@com_google_protobuf//:protobuf"],
        protoc = "@com_google_protobuf//:protoc",
        default_runtime = "@com_google_protobuf//:protobuf",
        use_grpc_plugin = use_grpc_plugin,
        testonly = testonly,
        visibility = visibility,
    )

def s2t_proto_library_py(name, proto_library, srcs = [], deps = [], oss_deps = [], visibility = None, testonly = 0, api_version = None):
    """Opensource py_proto_library.

    Uses a custom rule implementation that properly generates Python from proto_library
    and provides PyInfo for Python library dependencies.

    Note: s2t_proto_library creates {name}_proto for the proto_library, so we append _proto.
    """
    _ignore = [api_version, srcs, deps]

    if not proto_library:
        fail("proto_library parameter is required for s2t_proto_library_py")

    # s2t_proto_library creates a proto_library named {name}_proto
    # So we need to reference it correctly
    actual_proto_library = ":" + proto_library + "_proto"

    # Use our custom py_proto_library rule
    _py_proto_library_rule(
        name = name,
        deps = [actual_proto_library] + oss_deps,
        visibility = visibility,
    )
