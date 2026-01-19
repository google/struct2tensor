# Third Party Dependencies

## TensorFlow Metadata Protobuf Downgrade Patch

### Background

TensorFlow Metadata v1.17.1 upgraded from Protobuf 3.21.9 to 4.25.6 in their Bazel build configuration. However, this creates compatibility issues with TensorFlow 2.17.1, which still uses an older Protobuf version range (`>=3.20.3,<5.0.0dev` with exclusions).

### Solution

To maintain compatibility with TensorFlow 2.17.1 while using TensorFlow Metadata v1.17.1, we apply a patch that reverts the Protobuf upgrade in TFMD back to version 3.21.9.

The patch file `tfmd_protobuf_downgrade.patch` reverts the following changes from TFMD v1.17.1:

1. **tensorflow_metadata/proto/v0/BUILD file**:
   - Reverts proto library definitions to use Protobuf 3.x syntax
   - Restores the old `cc_proto_library` and `py_proto_library` patterns

### Usage

The patch is automatically applied when building struct2tensor through the `patches` parameter in the `http_archive` rule for `com_github_tensorflow_metadata` in [workspace.bzl](../struct2tensor/workspace.bzl).

No manual intervention is required.

### References

- [TensorFlow Metadata v1.17.0...v1.17.1 comparison](https://github.com/tensorflow/metadata/compare/v1.17.0...v1.17.1)
- [TensorFlow 2.17.1 dependencies](https://github.com/tensorflow/tensorflow/blob/v2.17.1/tensorflow/tools/pip_package/setup.py)
