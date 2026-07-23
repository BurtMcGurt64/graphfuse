"""
Regenerate assets/benchmark.png from the numbers graphfuse.bench printed.

The speedups below came from one run on the gpu server (fp32). If you re-run the
benchmark and get different numbers, paste them in here and re-run:

    uv run --with matplotlib python scripts/plot_bench.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# tensor size (elements) -> measured speedup vs eager pytorch
N = [65536, 1048576, 4194304, 16777216, 67108864]
short = [1.69, 1.54, 1.22, 1.41, 1.94]   # relu((a*b)+c)       - memory model says 2.0x
long_ = [2.34, 2.09, 1.69, 2.38, 3.38]   # relu((a*b+c)*a+b)   - memory model says 3.5x

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(N, short, "o-", label="relu((a*b)+c)  ·  3 ops fused")
ax.plot(N, long_, "s-", label="relu((a*b+c)*a+b)  ·  5 ops fused")

# the analytical memory-traffic ceilings each curve should approach as it becomes
# bandwidth-bound
ax.axhline(2.0, ls="--", color="gray", lw=1)
ax.axhline(3.5, ls="--", color="gray", lw=1)
ax.text(N[0], 2.06, "memory model ceiling · 2.0x", fontsize=8, color="gray")
ax.text(N[0], 3.56, "memory model ceiling · 3.5x", fontsize=8, color="gray")

ax.set_xscale("log")
ax.set_xlabel("tensor size (elements)")
ax.set_ylabel("speedup vs eager pytorch")
ax.set_title("fused triton kernel vs eager pytorch (fp32)")
ax.set_ylim(0, 4)
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)

os.makedirs("assets", exist_ok=True)
fig.tight_layout()
fig.savefig("assets/benchmark.png", dpi=130)
print("wrote assets/benchmark.png")
