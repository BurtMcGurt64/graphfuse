"""
Benchmark fused triton kernels against eager torch: wall-clock timing plus an
analytical model of DRAM traffic and kernel launches saved by fusion.
A pointwise op is memory-bound - it does a
tiny bit of arithmetic per element and spends all its time reading/writing DRAM. Eager
runs each op as its own kernel and materializes every intermediate back to memory, so an
n-op chain pays for n round trips. Fusing the chain into one kernel keeps the
intermediates in registers, so we only pay to load the inputs once and store the result
once, no matter how long the chain is.
"""

import torch

from .graph import Graph
from .fuse import fuse
from .codegen import compile_fused


def memory_model(fnode, n, dtype_size):
    """
    Analytical DRAM traffic (bytes) and kernel launches for fused vs eager.

    fused: load each external input once, store the sink once. intermediates live in
    registers so they never touch DRAM.

    eager: every op reads its operands from DRAM and writes its result back, so we sum
    (num_inputs + 1) over every op in the group.
    """
    members = fnode.attrs["nodes"]
    n_inputs = len(fnode.attrs["inputs"])

    fused_bytes = (n_inputs + 1) * n * dtype_size
    eager_bytes = sum(len(m.inputs) + 1 for m in members) * n * dtype_size

    return {
        "fused_bytes": fused_bytes,
        "eager_bytes": eager_bytes,
        "fused_launches": 1,
        "eager_launches": len(members),
    }


def _time_ms(fn):
    """median runtime in ms, with warmup. uses triton's do_bench if available."""
    try:
        from triton.testing import do_bench
        return do_bench(fn)
    except ImportError:
        # manual fallback: warmup, then time with cuda events
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(100):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / 100


def run_case(name, build, eager_fn, sizes):
    """
    build(g)     -> sets g.outputs to a single fused-able expression
    eager_fn(d)  -> the same expression in plain torch, given a name->tensor dict
    """
    g = Graph()
    build(g)
    fnode = fuse(g).outputs[0]
    assert fnode.op == "fused", f"{name}: expected the whole expr to fuse into one node"
    call = compile_fused(fnode)
    order = [node.name for node in fnode.attrs["inputs"]]  # tensor order the kernel wants

    print(f"\n=== {name} ===")
    print(f"fusing {len(fnode.attrs['nodes'])} ops over inputs {order}")
    print(f"{'N':>12} {'eager ms':>9} {'fused ms':>9} {'speedup':>8} "
          f"{'eager GB':>9} {'fused GB':>9} {'traffic':>8} {'launches':>9}")

    for n in sizes:
        tensors = {nm: torch.randn(n, device="cuda") for nm in set(order)}
        args = [tensors[nm] for nm in order]

        got = call(*args)
        ref = eager_fn(tensors)
        assert torch.allclose(got, ref, atol=1e-5), f"{name}: mismatch at N={n}"

        t_eager = _time_ms(lambda: eager_fn(tensors))
        t_fused = _time_ms(lambda: call(*args))
        m = memory_model(fnode, n, args[0].element_size())

        print(f"{n:>12} {t_eager:>9.4f} {t_fused:>9.4f} {t_eager / t_fused:>7.2f}x "
              f"{m['eager_bytes'] / 1e9:>9.3f} {m['fused_bytes'] / 1e9:>9.3f} "
              f"{m['eager_bytes'] / m['fused_bytes']:>7.2f}x "
              f"{m['eager_launches']}->{m['fused_launches']:>4}")


def main():
    assert torch.cuda.is_available(), "no cuda visible - are you on the server?"
    sizes = [1 << 16, 1 << 20, 1 << 22, 1 << 24, 1 << 26]

    # short chain: 3 ops, 3 inputs -> 8N eager vs 4N fused = 2x
    def build_short(g):
        a, b, c = g.input("a"), g.input("b"), g.input("c")
        g.outputs = [g.relu(g.add(g.mul(a, b), c))]

    run_case(
        "relu((a*b)+c)",
        build_short,
        lambda d: torch.relu(d["a"] * d["b"] + d["c"]),
        sizes,
    )

    # longer chain: 5 ops, 3 inputs -> 14N eager vs 4N fused = 3.5x
    def build_long(g):
        a, b, c = g.input("a"), g.input("b"), g.input("c")
        g.outputs = [g.relu(g.add(g.mul(g.add(g.mul(a, b), c), a), b))]

    run_case(
        "relu((a*b+c)*a + b)",
        build_long,
        lambda d: torch.relu((d["a"] * d["b"] + d["c"]) * d["a"] + d["b"]),
        sizes,
    )


if __name__ == "__main__":
    main()
