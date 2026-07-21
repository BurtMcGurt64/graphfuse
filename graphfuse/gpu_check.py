"""
Run this ON THE GPU SERVER to actually validate the generated triton kernel:

    uv run python -m graphfuse.gpu_check

It fuses relu((a*b)+c), lowers the fused group to a triton kernel, runs it on cuda,
and checks it matches plain torch. (Can't run on the mac - no triton, no gpu.)
"""

import torch

from .graph import Graph
from .fuse import fuse
from .codegen import emit_kernel, compile_fused


def main():
    g = Graph()
    a, b, c = g.input("a"), g.input("b"), g.input("c")
    out = g.relu(g.add(g.mul(a, b), c))
    g.outputs = [out]

    fg = fuse(g)
    fnode = fg.outputs[0]

    print("kernel source:\n")
    print(emit_kernel(fnode)[0])

    assert torch.cuda.is_available(), "no cuda visible - are you on the server?"

    # attrs['inputs'] order is a, b, c, so pass the tensors in that order
    ta = torch.randn(4096, device="cuda")
    tb = torch.randn(4096, device="cuda")
    tc = torch.randn(4096, device="cuda")

    call = compile_fused(fnode)
    got = call(ta, tb, tc)
    ref = torch.relu(ta * tb + tc)

    diff = (got - ref).abs().max().item()
    print("max abs diff vs torch:", diff)
    assert torch.allclose(got, ref, atol=1e-5), "triton output does not match torch!"
    print("triton kernel matches torch ✓")


if __name__ == "__main__":
    main()
