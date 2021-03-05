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

"""get_file_descriptor_set_proto() gets a FileDescriptorSet from a descriptor.

1. It gets the file descriptor associated with the descriptor.
2. It gets all file descriptors that file depends upon.
3. It orders the file descriptors so that earlier file descriptors don't depend
   upon later ones (fails on circular dependencies but that won't happen).
4. It serializes the file descriptors into a FileDescriptorSet.
"""
from typing import Sequence, Set

from struct2tensor import path

from google.protobuf import descriptor_pb2
from google.protobuf import descriptor


def _are_dependencies_handled(file_descriptor: descriptor.FileDescriptor,
                              dependencies: Set[descriptor.Descriptor]) -> bool:
  """Returns True iff dependencies of descriptor are in dependencies."""
  for dependency in file_descriptor.dependencies:
    if dependency not in dependencies:
      return False
  return True


def _order_dependencies(file_descriptors: Set[descriptor.FileDescriptor]
                       ) -> Sequence[descriptor.FileDescriptor]:
  """Given a set of file descriptors, return them as an ordered list.

  Each file descriptor in the resulting list follows its dependencies.
  Args:
    file_descriptors: a list of file descriptors.
  Returns:
    file descriptors as an ordered list.
  Raises:
    ValueError: if there is a circular dependency.
  """
  # TODO(martinz): if you first create a graph, then you can calculate this in
  # linear (instead of quadratic) time. However, let's hold off on that
  # complexity for now.
  result = []
  result_set = set()
  progress = True
  # We sort the file descriptors so that the list of descriptors returned
  # is deterministic and the attr input to the DecodeProto* ops are stable.
  remaining_file_descriptors = sorted(list(file_descriptors),
                                      key=lambda x: x.name)
  while remaining_file_descriptors and progress:
    failed = []
    progress = False
    while remaining_file_descriptors:
      file_descriptor = remaining_file_descriptors.pop()
      if _are_dependencies_handled(file_descriptor, result_set):
        result.append(file_descriptor)
        result_set.add(file_descriptor)
        progress = True
      else:
        failed.append(file_descriptor)
    remaining_file_descriptors = failed
  if remaining_file_descriptors:
    # In theory, the module should not have even compiled.
    raise ValueError('Circular dependency ' + str(
        [file_desc.name for file_desc in remaining_file_descriptors]))
  return result


def _get_dependencies_recursively(
    initial_file_descriptors: Set[descriptor.FileDescriptor]
) -> Set[descriptor.FileDescriptor]:
  """Gets all dependencies of file_descriptors as a set (including itself)."""
  file_descriptor_set = set(initial_file_descriptors)
  boundary = list(initial_file_descriptors)
  while boundary:
    current = boundary.pop()
    for dependency in current.dependencies:
      if dependency not in file_descriptor_set:
        file_descriptor_set.add(dependency)
        boundary.append(dependency)
  return file_descriptor_set


def _create_file_descriptor_set_proto(
    file_descriptor_list: Sequence[descriptor.FileDescriptor]
) -> descriptor_pb2.FileDescriptorSet:
  """Creates a FileDescriptorSet proto from a list of file descriptors."""
  result = descriptor_pb2.FileDescriptorSet()
  for file_descriptor in file_descriptor_list:
    file_descriptor.CopyToProto(result.file.add())
  return result


def _get_initial_file_descriptor_set(
    descriptor_type: descriptor.Descriptor,
    field_names: Sequence[str]) -> Set[descriptor.FileDescriptor]:
  """Gets a set of file descriptors for a descriptor and extensions."""
  result = set()
  result.add(descriptor_type.file)
  for field_name in field_names:
    if path.is_extension(field_name):
      extension_field = descriptor_type.file.pool.FindExtensionByName(
          path.get_raw_extension_name(field_name))
      extension_file = extension_field.file
      if extension_file not in result:
        result.add(extension_file)
  return result


def get_file_descriptor_set_proto(
    descriptor_type: descriptor.Descriptor,
    field_names: Sequence[str]) -> descriptor_pb2.FileDescriptorSet:
  """Returns a FileDescriptorSet for parsing field_names in a descriptor_type.

  The FileDescriptorSet has file descriptors for the file of the
  descriptor_type, the files of any extensions in field_names, and any files
  they depend upon.

  Args:
    descriptor_type: the type to be parsed.
    field_names: fields to be parsed.
  """
  return _create_file_descriptor_set_proto(
      _order_dependencies(
          _get_dependencies_recursively(
              _get_initial_file_descriptor_set(descriptor_type, field_names))))
