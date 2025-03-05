"""
Computation Graph Building
==========================

This module allows to build an abstract computation graph using TensorFlow-like
computation primitives and concepts. These primitives and concepts are also similar
to NumPy primitives, with minor naming differences.

This module provides:

1. Basic node types for constants and placeholders
2. Arithmetic operations (add, subtract, multiply, divide)
3. Comparison operations (equal, not_equal, less, less_equal, greater, greater_equal)
4. Logical operations (and, or, xor, not)
5. Mathematical operations (exp, power, log)
6. Shape manipulation operations (expand_dims, squeeze)
7. Reduction operations (sum, mean)

The nodes form a directed acyclic graph (DAG) that represents computations
to be performed. Each node implements a specific operation and stores its
inputs as attributes. The graph can then be evaluated by traversing the nodes
and performing their operations using NumPy, TensorFlow, etc.

We anticipate using NumPy/TensorFlow to perform computation based on matrices
of diverse shapes ("tensors"), therefore, we have included operations for shape
manipulation including expanding the dimensions and projecting over the axes. For
example, expand_dims allows to add new axes of size 1 to a tensor's shape, while
project_using_sum allows to compute the sum of tensor elements along specified
axes, thus projecting the tensor onto a lower-dimensional space.

To allow for uniform manipulation, we define the following operation groups:

1. BinaryOp: Operations that take two graph nodes as input
2. UnaryOp: Operations that take one graph node as input
3. AxisOp: Operations that take a graph node and an axis specification as input
and either expand to a higher-dimensional space or reduce to a lower-dimensional
space by projecting over one or more axes using a specific reduction operation.

Here's an example of what you can do with this module:

    >>> from dt_model.engine.frontend import graph
    >>>
    >>> a = graph.placeholder("a", 1.0)
    >>> b = graph.constant(2.0)
    >>> c = graph.add(a, b)
    >>> d = grap.multiply(c, c)
    >>>
    >>> # Expand to a higher-dimensional space
    >>> e = graph.expand_dims(d, axis=(1,2))
    >>>
    >>> # Project to a lower-dimensional space by summing over axis 0
    >>> f = graph.project_using_sum(e, axis=0)

Like TensorFlow, we support placeholders. That is, variables with a given
name that can be filled in at execution time with concrete values. We also
support constants, which must be bool, float, or int scalars.

Because our goal is to *capture* the arguments provided to function invocations
for later evaluation, we are using classes instead of functions. (We could
alternatively have used closures, but it would have been more clumsy.) To keep
the invoked entities names as close as possible to TensorFlow, we named the
classes using snake_case rather than CamelCase. This is a pragmatic and conscious
choice: violating PEP8 to produce code that reads like TensorFlow.

The main type in this module is the `Node`, representing a node in the
computation graph. Each operation (e.g., `add`) is a subclass of the `Node`
capturing the arguments it has been provided on construction.

Design Decisions
----------------

1. Class-based vs Function-based:
   - Classes capture operation arguments naturally
   - Enable visitor pattern for transformations
   - Allow future addition of operation-specific attributes

2. Snake Case Operation Names:
   - Match NumPy/TensorFlow conventions
   - Improve readability in mathematical context

3. Node Identity:
   - Nodes are identified by their instance identity
   - Enables graph traversal and transformation
"""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Sequence

from .. import atomic


Axis = int | tuple[int, ...]
"""Type alias for axis specifications in shape operations."""

Scalar = bool | float | int
"""Type alias for supported scalar value types."""


NODE_FLAG_TRACE = 1 << 0
"""Inserts a tracepoint at the corresponding graph node."""

NODE_FLAG_BREAK = 1 << 1
"""Inserts a breakpoint at the corresponding graph node."""


_id_generator = atomic.Int()
"""Atomic integer generator for unique node IDs."""


class Node:
    """
    Base class for all computation graph nodes.

    Design Notes
    ------------

    1. Identity Semantics:
        - Nodes use identity-based hashing and equality
        - This allows graph traversal algorithms to work correctly
        - Enables use of nodes as dictionary and sets keys

    2. Debug Support:
        - Nodes carry flags for debugging (trace/break)
        - Names for better error reporting
        - Extensible flag system for future debug features
    """

    def __init__(self, name: str = "") -> None:
        self.name = name
        self.flags = 0
        self.id = _id_generator.add(1)

    def __hash__(self) -> int:
        # Note: introducing hashing by identity in the class inheritance
        # chain to ensure that overriding the equality operator type signature
        # in derived classes does not break assigning to dicts.
        return id(self)


class constant(Node):
    """A constant scalar value in the computation graph.

    Args:
        value: The scalar value to store in this node.
    """

    def __init__(self, value: Scalar, name: str = "") -> None:
        super().__init__(name)
        self.value = value


class placeholder(Node):
    """Named placeholder for a value to be provided during evaluation.

    Args:
        default_value: Optional default scalar value to use for the
        placeholder if no type is provided at evaluation time.
    """

    def __init__(self, name: str, default_value: Scalar | None = None) -> None:
        super().__init__(name)
        self.default_value = default_value


class BinaryOp(Node):
    """Base class for binary operations.

    Args:
        left: First input node
        right: Second input node
    """

    def __init__(self, left: Node, right: Node) -> None:
        super().__init__()
        self.left = left
        self.right = right


# Arithmetic operations


class add(BinaryOp):
    """Element-wise addition of two tensors."""


class subtract(BinaryOp):
    """Element-wise subtraction of two tensors."""


class multiply(BinaryOp):
    """Element-wise multiplication of two tensors."""


class divide(BinaryOp):
    """Element-wise division of two tensors."""


# Comparison operations


class equal(BinaryOp):
    """Element-wise equality comparison of two tensors."""


class not_equal(BinaryOp):
    """Element-wise inequality comparison of two tensors."""


class less(BinaryOp):
    """Element-wise less-than comparison of two tensors."""


class less_equal(BinaryOp):
    """Element-wise less-than-or-equal comparison of two tensors."""


class greater(BinaryOp):
    """Element-wise greater-than comparison of two tensors."""


class greater_equal(BinaryOp):
    """Element-wise greater-than-or-equal comparison of two tensors."""


# Logical operations


class logical_and(BinaryOp):
    """Element-wise logical AND of two boolean tensors."""


class logical_or(BinaryOp):
    """Element-wise logical OR of two boolean tensors."""


class logical_xor(BinaryOp):
    """Element-wise logical XOR of two boolean tensors."""


class UnaryOp(Node):
    """Base class for unary operations.

    Args:
        node: Input node
    """

    def __init__(self, node: Node) -> None:
        super().__init__()
        self.node = node


class logical_not(UnaryOp):
    """Element-wise logical NOT of a boolean tensor."""


# Math operations


class exp(UnaryOp):
    """Element-wise exponential of a tensor."""


class power(BinaryOp):
    """Element-wise power operation (first tensor raised to power of second)."""


pow = power
"""Name alias for power, for compatibility with NumPy naming."""


class log(UnaryOp):
    """Element-wise natural logarithm of a tensor."""


class maximum(BinaryOp):
    """Element-wise maximum of two tensors."""


# Conditional operations


class where(Node):
    """Selects elements from tensors based on a condition.

    Args:
        condition: Boolean tensor
        then: Values to use where condition is True
        otherwise: Values to use where condition is False
    """

    def __init__(self, condition: Node, then: Node, otherwise: Node) -> None:
        super().__init__()
        self.condition = condition
        self.then = then
        self.otherwise = otherwise


class multi_clause_where(Node):
    """Selects elements from tensors based on multiple conditions.

    Args:
        clauses: List of (condition, value) pairs
        default_value: Value to use when no condition is met
    """

    def __init__(
        self, clauses: Sequence[tuple[Node, Node]], default_value: Node
    ) -> None:
        super().__init__()
        self.clauses = clauses
        self.default_value = default_value


# Shape-changing operations


class AxisOp(Node):
    """Base class for axis manipulation operations.

    We use these operations to expand a tensor to a higher-dimensional
    space or to reduce its dimensionality by projecting over one or more
    axes using a specific reduction operation.

    Args:
        node: Input tensor
        axis: Axis specification
    """

    def __init__(self, node: Node, axis: Axis) -> None:
        super().__init__()
        self.node = node
        self.axis = axis


class expand_dims(AxisOp):
    """Adds new axes of size 1 to a tensor's shape, thus expanding the
    tensor to a higher-dimensional space."""


class squeeze(AxisOp):
    """Removes axes of size 1 from a tensor's shape."""


class project_using_sum(AxisOp):
    """Computes sum of tensor elements along specified axes, thus
    projecting the tensor onto a lower-dimensional space."""


reduce_sum = project_using_sum
"""Name alias for project_using_sum, for compatibility with yakof, which still
uses this name. We will remove this symbol once the merge of yakof into
the dt-model is complete."""


class project_using_mean(AxisOp):
    """Computes mean of tensor elements along specified axes, thus
    projecting the tensor onto a lower-dimensional space."""


reduce_mean = project_using_mean
"""Name alias for project_using_mean, for compatibility with yakof, which
still uses this name. We will remove this symbol once the merge of
yakof into the dt-model is complete."""


# Debug operations


def tracepoint(node: Node) -> Node:
    """
    Marks the node as a tracepoint and returns it. The tracepoint
    will take effect while evaluating the node. We will print information
    before evaluating the node, evaluate it, then print the result.

    This function acts like the unit in the category with semantic side
    effects depending on the debug operation that is requested.
    """
    node.flags |= NODE_FLAG_TRACE
    return node


def breakpoint(node: Node) -> Node:
    """
    Marks the node as a breakpoint and returns it. The breakpoint will
    cause the interpreter to stop before evaluating the node.

    This function acts like the unit in the category with semantic side
    effects depending on the debug operation that is requested.
    """
    node.flags |= NODE_FLAG_TRACE | NODE_FLAG_BREAK
    return node
