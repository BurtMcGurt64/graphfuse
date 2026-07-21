"""
a quick end-to-end walk through what graphfuse can currently do:
build a graph, run it, and watch the passes rewrite it.

run with:  uv run python -m graphfuse.demo
"""

import torch

from .graph import Graph
from .interpreter import run
from .passes import dce, cfold, cse
from .fuse import fuse
from .codegen import emit_kernel


def demo_interpreter():
    # relu((a * b) + c) on real tensors
    g = Graph()
    a, b, c = g.input("a"), g.input("b"), g.input("c")
    out = g.relu(g.add(g.mul(a, b), c))
    g.outputs = [out]

    inputs = {
        "a": torch.tensor([1.0, -2.0, 3.0]),
        "b": torch.tensor([4.0, 5.0, -6.0]),
        "c": torch.tensor([0.0, 1.0, 2.0]),
    }
    print("interpreter: relu((a*b)+c) =", run(inputs, out))


def demo_cfold():
    # add(2, 3) is all-const, so cfold should collapse it to a single const node
    g = Graph()
    out = g.add(g.const(2), g.const(3))
    g.outputs = [out]

    folded = dce(cfold(g))
    print("cfold: add(2,3) ->", [n.op for n in folded.nodes], "value =", folded.outputs[0].attrs.get("val"))


def demo_cse():
    # mul(a,b) shows up twice, cse should dedup it down to one
    g = Graph()
    a, b = g.input("a"), g.input("b")
    out = g.add(g.mul(a, b), g.mul(a, b))
    g.outputs = [out]

    before = len(g.nodes)
    deduped = cse(g)
    print(f"cse: mul(a,b) twice -> nodes {before} down to {len(deduped.nodes)}")


def demo_fusion():
    # relu((a*b)+c) is one pointwise chain, so fuse() should collapse mul/add/relu
    # into a single fused node with a, b, c as its external inputs
    g = Graph()
    a, b, c = g.input("a"), g.input("b"), g.input("c")
    out = g.relu(g.add(g.mul(a, b), c))
    g.outputs = [out]

    fg = fuse(g)

    inputs = {
        "a": torch.tensor([1.0, -2.0, 3.0]),
        "b": torch.tensor([4.0, 5.0, -6.0]),
        "c": torch.tensor([0.0, 1.0, 2.0]),
    }
    # the whole point: fusing must not change the answer. check the fused graph against
    # the original by running both through the interpreter
    original = run(inputs, g.outputs[0])
    fused = run(inputs, fg.outputs[0])
    print("fusion: nodes", [n.op for n in g.nodes], "->", [n.op for n in fg.nodes])
    print("fusion: original == fused ?", bool(torch.allclose(original, fused)), "|", fused)

    # and here's the triton kernel we'd ship to the gpu for that fused node. this is
    # pure string generation so it's safe to look at on the mac (running it needs cuda)
    src, _ = emit_kernel(fg.outputs[0])
    print("\ngenerated triton kernel:\n")
    print(src)


def main():
    demo_interpreter()
    demo_cfold()
    demo_cse()
    demo_fusion()


if __name__ == "__main__":
    main()
