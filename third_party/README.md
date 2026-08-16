# Third-party patch notes

This directory contains local patches applied to upstream dependencies to keep
`struct2tensor` buildable with the dependency versions used in this repository.

## `tensorflow.patch`

Purpose:

- Adds compatibility changes for older/incompatible Abseil (`absl`) APIs used by
    TensorFlow/TSL in this workspace.
- Applies protobuf build-system updates needed for Protobuf `4.25.6` (the
    protobuf version currently used by `struct2tensor`).

Details:

- `absl` compatibility changes include fallback headers and minor include/deps
    adjustments (for example prefetch shims and status-message compatibility).
- Protobuf compatibility changes update proto-related Bazel logic/rules so code
    generation and proto library wiring work under Protobuf `4.25.6`.

Removal conditions:

- To remove this patch, TensorFlow must use Protobuf `4.25.6` natively in the
    version consumed by `struct2tensor`.
- TensorFlow `2.18` will also require a similar patch when used with this
    repository setup; the patch in this directory is not expected to apply
    unchanged to TensorFlow `2.18`.

## `tfmd.patch`

Purpose:

- Applies protobuf-related build changes in TensorFlow Metadata (TFMD) to make
    it compatible with Protobuf `4.25.6` used by `struct2tensor`.

Removal conditions:

- To remove this patch, TFMD/metadata should be upgraded to a version (or set of
    upstream changes) that supports Protobuf `4.25.6` directly.
- This is expected to be feasible, but requires dedicated work in metadata.
