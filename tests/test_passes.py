"""
Differential testing for the optimization passes.

The rule is simple: an optimization is only valid if it doesn't change the answer. So
for a bunch of graphs we compute the reference result with the interpreter, then run the
same inputs through each optimized graph and check they still match.
"""

import pytest
import torch

from graphfuse.graph import Graph
from graphfuse.interpreter import run
from graphfuse.passes import dce, cfold, cse
from graphfuse.fuse import fuse


def _tensors(*names, n=64):
    return {nm: torch.randn(n) for nm in names}


def g_chain():
    # one straight pointwise chain
    g = Graph()
    a, b, c = g.input("a"), g.input("b"), g.input("c")
    g.outputs = [g.relu(g.add(g.mul(a, b), c))]
    return g, _tensors("a", "b", "c")


def g_shared():
    # mul(a,b) feeds two branches (fan-out) and there are two outputs
    g = Graph()
    a, b, c, d = g.input("a"), g.input("b"), g.input("c"), g.input("d")
    m = g.mul(a, b)
    g.outputs = [g.relu(g.add(m, c)), g.relu(g.add(m, d))]
    return g, _tensors("a", "b", "c", "d")


def g_repeated():
    # mul(a,b) written twice - CSE should collapse it
    g = Graph()
    a, b = g.input("a"), g.input("b")
    g.outputs = [g.add(g.mul(a, b), g.mul(a, b))]
    return g, _tensors("a", "b")


def g_const():
    # a constant subexpression for constant folding to eat
    g = Graph()
    a = g.input("a")
    g.outputs = [g.add(g.mul(g.const(2.0), g.const(3.0)), a)]
    return g, _tensors("a")


def g_matmul():
    # a non-pointwise op in the middle, which splits the pointwise chains around it
    g = Graph()
    a, b, c = g.input("a"), g.input("b"), g.input("c")
    x = g.mul(a, b)
    mm = g.matmul(x, x)
    g.outputs = [g.relu(g.add(mm, c))]
    return g, {nm: torch.randn(8, 8) for nm in ("a", "b", "c")}


BUILDERS = [g_chain, g_shared, g_repeated, g_const, g_matmul]


def _run_all(inputs, graph):
    return [run(inputs, out) for out in graph.outputs]


def _same(a, b):
    assert len(a) == len(b)
    for got, want in zip(a, b):
        assert torch.allclose(got, want, atol=1e-5)


@pytest.mark.parametrize("builder", BUILDERS, ids=lambda b: b.__name__)
@pytest.mark.parametrize("pass_fn", [dce, cfold, cse, fuse], ids=lambda p: p.__name__)
def test_pass_preserves_semantics(builder, pass_fn):
    g, inputs = builder()
    ref = _run_all(inputs, g)
    _same(_run_all(inputs, pass_fn(g)), ref)


@pytest.mark.parametrize("builder", BUILDERS, ids=lambda b: b.__name__)
def test_full_pipeline_preserves_semantics(builder):
    # fuse goes last: the graph-level passes clean up the fine-grained graph, then we
    # collapse what's left into fused kernels
    g, inputs = builder()
    ref = _run_all(inputs, g)
    optimized = fuse(dce(cfold(cse(g))))
    _same(_run_all(inputs, optimized), ref)
