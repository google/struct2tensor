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
"""Utility methods for using Prensors.

2. get_sparse_tensors(...) gets sparse tensors from a Prensor.
2. get_ragged_tensors(...) gets ragged tensors from a Prensor.

"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

from struct2tensor import calculate_options
from struct2tensor import path
from struct2tensor import prensor
from struct2tensor.ops import struct2tensor_ops
import tensorflow as tf
from typing import Mapping, Sequence, Tuple, Union


def get_positional_index(node):
  if isinstance(node, (prensor.LeafNodeTensor, prensor.ChildNodeTensor)):
    return struct2tensor_ops.run_length_before(node.parent_index)
  # RootNodeTensor
  return tf.range(node.size)


class _LeafNodePath(object):
  """A path ending in a leaf.

  In order to avoid type checks and casting in the heart of different methods
  using the Prensor object to get a ragged or sparse tensor, we first create a
  typed "list" of nodes. A _LeafNodePath always begins with the root and ends
  with a leaf. Notice that we can get a suffix by casting a child node to a
  root node.
  """

  def __init__(self, head,
               middle,
               tail):
    self._head = head
    self._middle = middle
    self._tail = tail

  @property
  def head(self):
    return self._head

  @property
  def middle(self):
    return self._middle

  @property
  def tail(self):
    return self._tail


class _ChildNodePath(object):
  """A _ChildNodePath is a path that ends with a child node.

  It keeps same triple structure as _LeafNodePath.
  We use these in _get_dewey_encoding.
  """

  def __init__(self, head,
               middle,
               tail):
    self._head = head
    self._middle = middle
    self._tail = tail

  @property
  def head(self):
    return self._head

  @property
  def middle(self):
    return self._middle

  @property
  def tail(self):
    return self._tail


def _as_root_node_tensor(node_tensor
                        ):
  if isinstance(node_tensor, prensor.RootNodeTensor):
    return node_tensor
  if isinstance(node_tensor, prensor.ChildNodeTensor):
    return prensor.RootNodeTensor(node_tensor.size)
  raise ValueError("Must be child or root node tensor (found {})".format(
      type(node_tensor)))


def _get_leaf_node_path(p, t):
  """Creates a _LeafNodePath to p."""
  leaf_node = t.get_descendant_or_error(p).node
  if not isinstance(leaf_node, prensor.LeafNodeTensor):
    raise ValueError("Expected Leaf Node at {} in {}".format(str(p), str(t)))
  if not p:
    raise ValueError("Leaf should not be at the root")
  # If there is a leaf at the root, this will return a ValueError.
  root_node = _as_root_node_tensor(t.node)

  # Not the root, not p.
  strict_ancestor_paths = [p.prefix(i) for i in range(1, len(p))]

  child_node_pairs = [(t.get_descendant_or_error(ancestor).node, ancestor)
                      for ancestor in strict_ancestor_paths]
  bad_struct_paths = [
      ancestor for node, ancestor in child_node_pairs
      if not isinstance(node, prensor.ChildNodeTensor)
  ]
  if bad_struct_paths:
    raise ValueError("Expected ChildNodeTensor at {} in {}".format(
        " ".join([str(x) for x in bad_struct_paths]), str(t)))
  # This should select all elements: the isinstance is for type-checking.
  child_nodes = [
      node for node, ancestor in child_node_pairs
      if isinstance(node, prensor.ChildNodeTensor)
  ]
  assert len(child_nodes) == len(child_node_pairs)
  return _LeafNodePath(root_node, child_nodes, leaf_node)


def _get_leaf_node_path_suffix(p):
  """Get the suffix of a LeafNodePath."""
  return _LeafNodePath(_as_root_node_tensor(p.middle[0]), p.middle[1:], p.tail)


def _get_node_path_parent(p
                         ):
  return _ChildNodePath(p.head, p.middle[:-1], p.middle[-1])


def _get_leaf_node_paths(t
                        ):
  """Gets a map of paths to leaf nodes in the expression."""
  return {
      k: _get_leaf_node_path(k, t)
      for k, v in t.get_descendants().items()
      if isinstance(v.node, prensor.LeafNodeTensor)
  }


#################### Code for get_sparse_tensors(...) ##########################


def _get_dewey_encoding(p
                       ):
  """Gets a pair of the indices and shape of these protos.

  See http://db.ucsd.edu/static/cse232B-s05/papers/tatarinov02.pdf

  Args:
    p: the path to encode.

  Returns:
    A pair of an indices matrix and a dense_shape
  """
  parent = p.middle[-1] if p.middle else p.head
  parent_size = tf.reshape(parent.size, [1])
  positional_index = get_positional_index(p.tail)
  # tf.reduce_max([]) == -kmaxint64 but we need it to be 0.
  current_size = tf.maximum(
      tf.reshape(tf.reduce_max(positional_index) + 1, [1]), [0])
  if not p.middle:
    if p.tail.is_repeated:
      return (tf.stack([p.tail.parent_index, positional_index],
                       axis=1), tf.concat([parent_size, current_size], 0))
    else:
      return tf.expand_dims(p.tail.parent_index, -1), parent_size
  else:
    parent_dewey_encoding, parent_size = _get_dewey_encoding(
        _get_node_path_parent(p))
    if p.tail.is_repeated:
      positional_index_as_matrix = tf.expand_dims(
          get_positional_index(p.tail), -1)
      indices = tf.concat([
          tf.gather(parent_dewey_encoding, p.tail.parent_index),
          positional_index_as_matrix
      ], 1)
      size = tf.concat([parent_size, current_size], 0)
      return (indices, size)
    else:
      return tf.gather(parent_dewey_encoding, p.tail.parent_index), parent_size


def _get_sparse_tensor(p):
  indices, dense_shape = _get_dewey_encoding(p)
  return tf.SparseTensor(
      indices=indices, values=p.tail.values, dense_shape=dense_shape)


def get_sparse_tensors(t, options
                      ):
  """Gets sparse tensors for all the leaves of the prensor expression.

  Args:
    t: The expression to extract tensors from.
    options: Currently unused.

  Returns:
    A map from paths to sparse tensors.
  """

  del options
  return {p: _get_sparse_tensor(v) for p, v in _get_leaf_node_paths(t).items()}


#################### Code for get_ragged_tensors(...) ##########################


def from_value_rowids_bridge(values,
                             value_rowids=None,
                             nrows=None,
                             validate=True):
  """validate option is only available internally for tf 0.13.1."""
  return tf.RaggedTensor.from_value_rowids(
      values,
      value_rowids=value_rowids,
      nrows=nrows
  )


def _get_ragged_tensor_from_leaf_node_path(nodes,
                                           options
                                          ):
  """Gets a ragged tensor from a leaf node path."""
  if not nodes.middle:
    return from_value_rowids_bridge(
        nodes.tail.values,
        value_rowids=nodes.tail.parent_index,
        nrows=nodes.head.size,
        validate=options.ragged_checks)
  deeper_ragged = _get_ragged_tensor_from_leaf_node_path(
      _get_leaf_node_path_suffix(nodes), options)
  first_child_node = nodes.middle[0]
  return from_value_rowids_bridge(
      deeper_ragged,
      value_rowids=first_child_node.parent_index,
      nrows=nodes.head.size,
      validate=options.ragged_checks)


def get_ragged_tensors(t, options
                      ):
  """Gets ragged tensors for all the leaves of the prensor expression.

  Args:
    t: The expression to extract tensors from.
    options: used to pass options for calculating ragged tensors.

  Returns:
    A map from paths to ragged tensors.
  """
  return {
      p: _get_ragged_tensor_from_leaf_node_path(v, options)
      for p, v in _get_leaf_node_paths(t).items()
  }
