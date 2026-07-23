"""
End-to-end GPU tests: generate a triton kernel, run it, and check it matches PyTorch.

These need triton + a cuda gpu, so they auto-skip everywhere else (e.g. the mac we edit
from). On the server `uv run pytest` picks them up.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("needs a cuda gpu", allow_module_level=True)

from graphfuse.graph import Graph
from graphfuse.fuse import fuse
from graphfuse.codegen import compile_fused


def _run_fused(build, n=4096):
    """build the expr, fuse it, run the kernel, and hand back (kernel result, tensors)."""
    g = Graph()
    build(g)
    fnode = fuse(g).outputs[0]
    call = compile_fused(fnode)

    names = {node.name for node in fnode.attrs["inputs"]}
    tensors = {nm: torch.randn(n, device="cuda") for nm in names}
    args = [tensors[node.name] for node in fnode.attrs["inputs"]]  # kernel's input order
    return call(*args), tensors


def test_kernel_matches_torch_short():
    def build(g):
        a, b, c = g.input("a"), g.input("b"), g.input("c")
        g.outputs = [g.relu(g.add(g.mul(a, b), c))]

    got, t = _run_fused(build)
    ref = torch.relu(t["a"] * t["b"] + t["c"])
    assert torch.allclose(got, ref, atol=1e-5)


def test_kernel_matches_torch_long():
    def build(g):
        a, b, c = g.input("a"), g.input("b"), g.input("c")
        g.outputs = [g.relu(g.add(g.mul(g.add(g.mul(a, b), c), a), b))]

    got, t = _run_fused(build)
    ref = torch.relu((t["a"] * t["b"] + t["c"]) * t["a"] + t["b"])
    assert torch.allclose(got, ref, atol=1e-5)
