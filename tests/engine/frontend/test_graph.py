"""Tests for the dt_model.engine.frontend.graph module."""

# SPDX-License-Identifier: Apache-2.0

from dt_model.engine.frontend import graph


def test_basic_node_creation():
    """Test basic node creation and properties."""
    # Test constants
    c1 = graph.constant(1.0, name="c1")
    assert c1.value == 1.0
    assert c1.name == "c1"

    # Test placeholders
    p1 = graph.placeholder("x", 1.0)
    assert p1.name == "x"
    assert p1.default_value == 1.0

    p2 = graph.placeholder("y")
    assert p2.name == "y"
    assert p2.default_value is None


def test_node_identity():
    """Test that nodes maintain proper identity."""
    a = graph.constant(1.0)
    b = graph.constant(1.0)
    assert hash(a) != hash(b)
    assert a is not b

    # Test identity preservation in operations
    c = graph.add(a, b)
    assert c.left is a
    assert c.right is b


def test_debug_flags():
    """Test debug operation flags."""
    a = graph.constant(1.0)

    # Test tracepoint
    traced = graph.tracepoint(a)
    assert traced.flags & graph.NODE_FLAG_TRACE
    assert traced is a  # Should return same node

    # Test breakpoint
    broken = graph.breakpoint(a)
    assert broken.flags & graph.NODE_FLAG_BREAK
    assert broken.flags & graph.NODE_FLAG_TRACE
    assert broken is a  # Should return same node


def test_complex_arithmetic_graph():
    """Test building a complex arithmetic computation graph."""
    # Build ((a + b) * c)^2 + exp(d)
    a = graph.placeholder("a")
    b = graph.placeholder("b")
    c = graph.placeholder("c")
    d = graph.placeholder("d")

    add_ab = graph.add(a, b)
    mult_c = graph.multiply(add_ab, c)
    square = graph.power(mult_c, graph.constant(2.0))
    exp_d = graph.exp(d)
    result = graph.add(square, exp_d)

    # Verify structure
    assert result.left is square
    assert result.right is exp_d
    assert square.left is mult_c
    assert isinstance(square.right, graph.constant)
    assert square.right.value == 2.0
    assert mult_c.left is add_ab
    assert mult_c.right is c
    assert add_ab.left is a
    assert add_ab.right is b


def test_binary_op_copy():
    """Test that binary operations can be correctly copied with new children."""
    # Create some nodes
    a = graph.placeholder("a")
    b = graph.placeholder("b")
    c = graph.placeholder("c")
    d = graph.placeholder("d")

    # Create a binary operation
    add_op = graph.add(a, b)
    add_op.name = "add_operation"
    add_op.flags = graph.NODE_FLAG_TRACE

    # Copy the operation with new children
    new_add = add_op.copy(left=c, right=d)

    # Verify the copy has correct structure
    assert isinstance(new_add, graph.add)
    assert new_add.left is c
    assert new_add.right is d

    # Verify attributes were preserved
    assert new_add.name == "add_operation"
    assert new_add.flags == graph.NODE_FLAG_TRACE

    # Verify it's a new object
    assert new_add is not add_op

    # Test more binary operations
    sub_op = graph.subtract(a, b)
    sub_op.name = "subtract_operation"
    new_sub = sub_op.copy(left=c, right=d)
    assert isinstance(new_sub, graph.subtract)
    assert new_sub.name == "subtract_operation"

    mul_op = graph.multiply(a, b)
    new_mul = mul_op.copy(left=c, right=d)
    assert isinstance(new_mul, graph.multiply)


def test_complex_conditional_graph():
    """Test building a complex conditional computation graph."""
    # Build a multi-clause conditional expression
    x = graph.placeholder("x")
    y = graph.placeholder("y")

    cond1 = graph.less(x, graph.constant(0.0))
    cond2 = graph.greater(x, graph.constant(1.0))

    val1 = graph.multiply(y, graph.constant(-1.0))
    val2 = graph.exp(y)
    default = y

    result = graph.multi_clause_where([(cond1, val1), (cond2, val2)], default)

    # Verify structure
    assert len(result.clauses) == 2
    assert result.clauses[0][0] is cond1
    assert result.clauses[0][1] is val1
    assert result.clauses[1][0] is cond2
    assert result.clauses[1][1] is val2
    assert result.default_value is default


def test_where_op_copy():
    """Test that where operations can be correctly copied with new children."""
    # Create some nodes
    a = graph.placeholder("a")
    b = graph.placeholder("b")
    c = graph.placeholder("c")
    d = graph.placeholder("d")
    e = graph.placeholder("e")
    f = graph.placeholder("f")

    # Create a where operation
    where_op = graph.where(a, b, c)
    where_op.name = "where_operation"
    where_op.flags = graph.NODE_FLAG_TRACE

    # Copy the operation with new children
    new_where = where_op.copy(condition=d, then=e, otherwise=f)

    # Verify the copy has correct structure
    assert isinstance(new_where, graph.where)
    assert new_where.condition is d
    assert new_where.then is e
    assert new_where.otherwise is f

    # Verify attributes were preserved
    assert new_where.name == "where_operation"
    assert new_where.flags == graph.NODE_FLAG_TRACE

    # Verify it's a new object
    assert new_where is not where_op


def test_multi_clause_where_op_copy():
    """Test that multi-clause where operations can be correctly copied with new children."""
    # Create some nodes
    a = graph.placeholder("a")
    b = graph.placeholder("b")
    c = graph.placeholder("c")
    d = graph.placeholder("d")

    # Create a multi-clause where operation
    clauses = [(a, b), (c, d)]
    default = graph.constant(0.0)
    mcw_op = graph.multi_clause_where(clauses, default)
    mcw_op.name = "mcw_operation"
    mcw_op.flags = graph.NODE_FLAG_TRACE

    # Create new nodes for the copy
    e = graph.placeholder("e")
    f = graph.placeholder("f")
    g = graph.placeholder("g")
    h = graph.placeholder("h")
    new_default = graph.constant(1.0)
    new_clauses = [(e, f), (g, h)]

    # Copy the operation with new children
    new_mcw = mcw_op.copy(clauses=new_clauses, default_value=new_default)

    # Verify the copy has correct structure
    assert isinstance(new_mcw, graph.multi_clause_where)
    assert len(new_mcw.clauses) == 2
    assert new_mcw.clauses[0][0] is e
    assert new_mcw.clauses[0][1] is f
    assert new_mcw.clauses[1][0] is g
    assert new_mcw.clauses[1][1] is h
    assert new_mcw.default_value is new_default

    # Verify attributes were preserved
    assert new_mcw.name == "mcw_operation"
    assert new_mcw.flags == graph.NODE_FLAG_TRACE

    # Verify it's a new object
    assert new_mcw is not mcw_op


def test_reduction_graph():
    """Test building a graph with reduction operations."""
    x = graph.placeholder("x")
    y = graph.placeholder("y")

    # Build mean(sum(x * y, axis=0), axis=1)
    prod = graph.multiply(x, y)
    sum_0 = graph.project_using_sum(prod, axis=0)
    result = graph.project_using_mean(sum_0, axis=1)

    # Verify structure
    assert result.node is sum_0
    assert result.axis == 1
    assert sum_0.node is prod
    assert sum_0.axis == 0
    assert prod.left is x
    assert prod.right is y


def test_unary_op_copy():
    """Test that unary operations can be correctly copied with new children."""
    # Create some nodes
    a = graph.placeholder("a")
    b = graph.placeholder("b")

    # Create a unary operation
    exp_op = graph.exp(a)
    exp_op.name = "exp_operation"
    exp_op.flags = graph.NODE_FLAG_BREAK

    # Copy the operation with a new child
    new_exp = exp_op.copy(node=b)

    # Verify the copy has correct structure
    assert isinstance(new_exp, graph.exp)
    assert new_exp.node is b

    # Verify attributes were preserved
    assert new_exp.name == "exp_operation"
    assert new_exp.flags == graph.NODE_FLAG_BREAK

    # Verify it's a new object
    assert new_exp is not exp_op

    # Test logical not
    not_op = graph.logical_not(a)
    not_op.name = "not_operation"
    new_not = not_op.copy(node=b)
    assert isinstance(new_not, graph.logical_not)
    assert new_not.name == "not_operation"
    assert new_not.node is b


def test_axis_op_copy():
    """Test that axis operations can be correctly copied with new children."""
    # Create some nodes
    a = graph.placeholder("a")
    b = graph.placeholder("b")

    # Create an axis operation
    sum_op = graph.reduce_sum(a, axis=0)
    sum_op.name = "sum_operation"
    sum_op.flags = graph.NODE_FLAG_TRACE

    # Copy the operation with a new child
    new_sum = sum_op.copy(node=b)

    # Verify the copy has correct structure
    assert isinstance(new_sum, graph.reduce_sum)
    assert new_sum.node is b
    assert new_sum.axis == 0  # Axis should be preserved

    # Verify attributes were preserved
    assert new_sum.name == "sum_operation"
    assert new_sum.flags == graph.NODE_FLAG_TRACE

    # Verify it's a new object
    assert new_sum is not sum_op

    # Test other axis operations
    mean_op = graph.reduce_mean(a, axis=1)
    new_mean = mean_op.copy(node=b)
    assert isinstance(new_mean, graph.reduce_mean)
    assert new_mean.axis == 1

    expand_op = graph.expand_dims(a, axis=2)
    new_expand = expand_op.copy(node=b)
    assert isinstance(new_expand, graph.expand_dims)
    assert new_expand.axis == 2


def test_name_propagation():
    """Test name handling across operations."""
    # Test explicit naming
    a = graph.placeholder("input_a", 1.0)
    b = graph.constant(2.0, name="const_b")
    c = graph.add(a, b)

    assert a.name == "input_a"
    assert b.name == "const_b"
    assert c.name == ""  # Operations don't get automatic names


def test_name_uniqueness():
    """Test that nodes with same names remain distinct."""
    a1 = graph.placeholder("x", 1.0)
    a2 = graph.placeholder("x", 2.0)

    assert a1.name == a2.name == "x"
    assert a1 is not a2
    assert hash(a1) != hash(a2)


def test_debug_operations_name_preservation():
    """Test that debug operations preserve names."""
    a = graph.placeholder("debug_node", 1.0)

    traced = graph.tracepoint(a)
    assert traced.name == "debug_node"
    assert traced is a

    breakpointed = graph.breakpoint(traced)
    assert breakpointed.name == "debug_node"
    assert breakpointed is a


def test_where_operation():
    """Test the where operation."""
    condition = graph.placeholder("condition")
    then = graph.constant(1.0)
    otherwise = graph.constant(0.0)

    result = graph.where(condition, then, otherwise)

    assert result.condition is condition
    assert result.then is then
    assert result.otherwise is otherwise


def test_node_id_generation():
    """Test that nodes receive unique, incrementing IDs."""
    # Create multiple nodes of different types
    nodes = [
        graph.Node(),
        graph.constant(1.0),
        graph.placeholder("x"),
        graph.Node(),
    ]

    # Check that IDs exist and are unique
    ids = [node.id for node in nodes]
    assert len(set(ids)) == len(ids)  # All IDs should be unique

    # Verify IDs are increasing
    for i in range(1, len(ids)):
        assert ids[i] > ids[i - 1]


def test_node_id_persistence():
    """Test that node IDs remain stable across operations."""
    a = graph.placeholder("a")
    id_a = a.id

    # Perform operations that use the node
    traced_a = graph.tracepoint(a)

    # Verify node identity is preserved and ID remains the same
    assert traced_a is a
    assert traced_a.id == id_a


def test_binary_op_node_ids():
    """Test that binary operation nodes get unique IDs."""
    a = graph.placeholder("a")
    b = graph.placeholder("b")

    op1 = graph.add(a, b)
    op2 = graph.subtract(a, b)

    assert hasattr(op1, "id")
    assert hasattr(op2, "id")
    assert op1.id != op2.id
    assert op1.id != a.id
    assert op1.id != b.id


def test_infix_arithmetic_operations():
    """Test infix arithmetic operations between nodes."""
    a = graph.placeholder("a")
    b = graph.placeholder("b")

    # Test operator overloading
    add1 = a + b
    assert isinstance(add1, graph.add)
    assert add1.left is a
    assert add1.right is b

    # Test scalar operations
    add2 = a + 2.0
    assert isinstance(add2, graph.add)
    assert add2.left is a
    assert isinstance(add2.right, graph.constant)
    assert add2.right.value == 2.0

    # Test reverse operations
    add3 = 2.0 + a
    assert isinstance(add3, graph.add)
    assert isinstance(add3.left, graph.constant)
    assert add3.left.value == 2.0
    assert add3.right is a

    # Test other arithmetic operations
    sub = a - b
    assert isinstance(sub, graph.subtract)
    assert sub.left is a
    assert sub.right is b

    mul = a * b
    assert isinstance(mul, graph.multiply)
    assert mul.left is a
    assert mul.right is b

    div = a / b
    assert isinstance(div, graph.divide)
    assert div.left is a
    assert div.right is b


def test_infix_comparison_operations():
    """Test infix comparison operations between nodes."""
    a = graph.placeholder("a")
    b = graph.placeholder("b")

    # Test all comparison operators
    eq = a == b
    assert isinstance(eq, graph.equal)
    assert eq.left is a
    assert eq.right is b

    ne = a != b
    assert isinstance(ne, graph.not_equal)
    assert ne.left is a
    assert ne.right is b

    lt = a < b
    assert isinstance(lt, graph.less)
    assert lt.left is a
    assert lt.right is b

    le = a <= b
    assert isinstance(le, graph.less_equal)
    assert le.left is a
    assert le.right is b

    gt = a > b
    assert isinstance(gt, graph.greater)
    assert gt.left is a
    assert gt.right is b

    ge = a >= b
    assert isinstance(ge, graph.greater_equal)
    assert ge.left is a
    assert ge.right is b


def test_infix_logical_operations():
    """Test infix logical operations between nodes."""
    a = graph.placeholder("a")
    b = graph.placeholder("b")

    # Test logical operators
    and_op = a & b
    assert isinstance(and_op, graph.logical_and)
    assert and_op.left is a
    assert and_op.right is b

    or_op = a | b
    assert isinstance(or_op, graph.logical_or)
    assert or_op.left is a
    assert or_op.right is b

    xor_op = a ^ b
    assert isinstance(xor_op, graph.logical_xor)
    assert xor_op.left is a
    assert xor_op.right is b

    not_op = ~a
    assert isinstance(not_op, graph.logical_not)
    assert not_op.node is a


def test_infix_reverse_operations():
    """Test reverse infix operations with scalar values."""
    a = graph.placeholder("a")

    # Test reverse operations with scalars
    rsub = 2.0 - a
    assert isinstance(rsub, graph.subtract)
    assert isinstance(rsub.left, graph.constant)
    assert rsub.left.value == 2.0
    assert rsub.right is a

    rmul = 2.0 * a
    assert isinstance(rmul, graph.multiply)
    assert isinstance(rmul.left, graph.constant)
    assert rmul.left.value == 2.0
    assert rmul.right is a

    rdiv = 2.0 / a
    assert isinstance(rdiv, graph.divide)
    assert isinstance(rdiv.left, graph.constant)
    assert rdiv.left.value == 2.0
    assert rdiv.right is a

    # Test reverse logical operations
    rand = True & a
    assert isinstance(rand, graph.logical_and)
    assert isinstance(rand.left, graph.constant)
    assert rand.left.value is True
    assert rand.right is a

    ror = True | a
    assert isinstance(ror, graph.logical_or)
    assert isinstance(ror.left, graph.constant)
    assert ror.left.value is True
    assert ror.right is a

    rxor = True ^ a
    assert isinstance(rxor, graph.logical_xor)
    assert isinstance(rxor.left, graph.constant)
    assert rxor.left.value is True
    assert rxor.right is a


def test_complex_infix_expressions():
    """Test complex expressions using infix operations."""
    a = graph.placeholder("a")
    b = graph.placeholder("b")
    c = graph.placeholder("c")

    # Test complex expression: (a + b) * (c - 2.0)
    expr = (a + b) * (c - 2.0)

    assert isinstance(expr, graph.multiply)
    assert isinstance(expr.left, graph.add)
    assert expr.left.left is a
    assert expr.left.right is b
    assert isinstance(expr.right, graph.subtract)
    assert expr.right.left is c
    assert isinstance(expr.right.right, graph.constant)
    assert expr.right.right.value == 2.0

    # Test complex logical expression: (a & b) | (~c)
    logical_expr = (a & b) | (~c)

    assert isinstance(logical_expr, graph.logical_or)
    assert isinstance(logical_expr.left, graph.logical_and)
    assert logical_expr.left.left is a
    assert logical_expr.left.right is b
    assert isinstance(logical_expr.right, graph.logical_not)
    assert logical_expr.right.node is c


def test_ensure_node_function():
    """Test the ensure_node function that converts scalars to constants."""
    # Test with node input
    node = graph.placeholder("test")
    result = graph.ensure_node(node)
    assert result is node  # Should return the same node

    # Test with scalar inputs
    result_float = graph.ensure_node(3.14)
    assert isinstance(result_float, graph.constant)
    assert result_float.value == 3.14

    result_int = graph.ensure_node(42)
    assert isinstance(result_int, graph.constant)
    assert result_int.value == 42

    result_bool = graph.ensure_node(True)
    assert isinstance(result_bool, graph.constant)
    assert result_bool.value is True
