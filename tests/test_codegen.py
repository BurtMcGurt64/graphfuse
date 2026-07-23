"""
Codegen tests that don't need a GPU: the kernel is generated as a string, so we can
check its structure (and that it's valid python) without triton installed. Also covers
the memory-traffic model the benchmark reports.
"""

from graphfuse.graph import Graph
from graphfuse.fuse import fuse
from graphfuse.codegen import emit_kernel
from graphfuse.bench import memory_model


def _relu_ab_c():
    g = Graph()
    a, b, c = g.input("a"), g.input("b"), g.input("c")
    g.outputs = [g.relu(g.add(g.mul(a, b), c))]
    return fuse(g).outputs[0]


def test_emit_kernel_structure():
    src, name = emit_kernel(_relu_ab_c())

    assert name == "fused_kernel"
    # three external inputs -> three pointer args + three loads
    assert src.count("tl.load") == 3
    for ptr in ("in0_ptr", "in1_ptr", "in2_ptr"):
        assert ptr in src
    # the ops themselves, in order
    assert "x0 * x1" in src
    assert "+ x2" in src
    assert "tl.maximum(" in src
    # exactly one write-back
    assert src.count("tl.store") == 1


def test_emit_kernel_is_valid_python():
    src, _ = emit_kernel(_relu_ab_c())
    # compile() parses without executing, so this passes even without triton installed
    compile(src, "<generated>", "exec")


def test_memory_model_counts():
    m = memory_model(_relu_ab_c(), n=1, dtype_size=4)
    assert m["fused_bytes"] == 4 * 4   # load a, b, c + store one result
    assert m["eager_bytes"] == 8 * 4   # mul(3) + add(3) + relu(2)
    assert m["fused_launches"] == 1
    assert m["eager_launches"] == 3
