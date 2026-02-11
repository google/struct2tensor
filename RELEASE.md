# Version 0.48.1

## Major Features and Improvements

*   N/A

## Bug Fixes and Other Changes

*   Add py_proto_library macro for OSS compatibility
*   Replace gunit_main with googletest gtest_main
*   Fix proto import to use workspace-relative path
*   Explicitly builds each dynamic library target that creates .so files
*   Update proto library generation to depend on generated
     cc_proto_library targets
*   Implement custom _tsl_py_proto_library_rule to replace the
    built-in py_proto_library removed in Protobuf 4.x
*   Refactor cc_proto_library to use native proto_library and
     cc_proto_library rules instead of custom proto_gen

## Breaking Changes

*   N/A

## Deprecations

*   N/A

# Version 0.48.0

## Major Features and Improvements

*   N/A

## Bug Fixes and Other Changes

*   Depends on `protobuf>=4.25.2,<6.0.0` for Python 3.11 and on `protobuf>=4.21.6,<6.0.0` for 3.9 and 3.10.
*   Depends on `tensorflow 2.17.1`.
*   macOS wheel publishing is temporarily paused due to missing ARM64 support.

## Breaking Changes

*   N/A

## Deprecations

*   N/A
