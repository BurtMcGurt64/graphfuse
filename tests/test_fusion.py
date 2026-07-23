"""
Structural tests for fusion: on top of "the answer is the same" (covered in
test_passes), check that fuse() actually groups things the way we expect.
"""

from graphfuse.graph import Graph
from graphfuse.fuse import fuse


def _count(graph, op):
    return sum(n.op == op for n in graph.nodes)


def test_chain_fuses_to_one_node():
    g = Graph()
    a, b, c = g.input("a"), g.input("b"), g.input("c")
    g.outputs = [g.relu(g.add(g.mul(a, b), c))]
    fg = fuse(g)
    # the whole pointwise chain collapses; only the 3 inputs + 1 fused node remain
    assert _count(fg, "fused") == 1
    assert _count(fg, "input") == 3
    assert len(fg.nodes) == 4


def test_matmul_splits_groups():
    g = Graph()
    a, b, c = g.input("a"), g.input("b"), g.input("c")
    x = g.mul(a, b)
    mm = g.matmul(x, x)
    g.outputs = [g.relu(g.add(mm, c))]
    fg = fuse(g)
    # mul is one group, relu(add(...)) is another, matmul stays on its own
    assert _count(fg, "fused") == 2
    assert _count(fg, "matmul") == 1


def test_shared_node_becomes_its_own_group():
    g = Graph()
    a, b, c, d = g.input("a"), g.input("b"), g.input("c"), g.input("d")
    m = g.mul(a, b)  # two users, so it can't be absorbed - it's its own sink
    g.outputs = [g.relu(g.add(m, c)), g.relu(g.add(m, d))]
    fg = fuse(g)
    # the shared mul plus the two relu(add) chains = 3 fused nodes
    assert _count(fg, "fused") == 3


def test_lone_op_still_fuses():
    g = Graph()
    g.outputs = [g.relu(g.input("a"))]
    fg = fuse(g)
    assert _count(fg, "fused") == 1
