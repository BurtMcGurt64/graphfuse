"""
Triton codegen for fused pointwise groups

We only lower the 'fused' nodes that fuse() produces, and those are pointwise by
construction (add / mul / relu over same-shape tensors). Anything else still runs
through torch
"""

import torch


# how each pointwise op looks as a triton expression, given the variable names of its
# already-computed inputs
_TRITON_OP = {
    "add": lambda a, b: f"{a} + {b}",
    "mul": lambda a, b: f"{a} * {b}",
    "relu": lambda x: f"tl.maximum({x}, 0.0)",
}


def emit_kernel(fnode, name="fused_kernel"):
    """
    Build triton source for one fused pointwise node. Returns (source, name).

    The idea: every external operand becomes a pointer arg that we load once, every
    internal op becomes one line of arithmetic over names we've already defined, and
    the sink gets stored back out.
    """
    assert fnode.op == "fused", "codegen only lowers fused nodes"

    ext_inputs = fnode.attrs["inputs"]   # external operands (old nodes), in call order
    members = fnode.attrs["nodes"]       # internal ops, topo order
    output = fnode.attrs["output"]       # the sink

    var = {}          # old node -> variable name inside the kernel
    ptr_args = []
    load_lines = []
    for j, ext in enumerate(ext_inputs):
        ptr = f"in{j}_ptr"
        v = f"x{j}"
        var[ext] = v
        ptr_args.append(ptr)
        load_lines.append(f"    {v} = tl.load({ptr} + offsets, mask=mask)")

    body_lines = []
    for k, m in enumerate(members):
        args = [var[i] for i in m.inputs]
        v = f"t{k}"
        var[m] = v
        body_lines.append(f"    {v} = {_TRITON_OP[m.op](*args)}")

    header = f"def {name}({', '.join(ptr_args)}, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):"
    src = "\n".join(
        [
            "import triton",
            "import triton.language as tl",
            "",
            "@triton.jit",
            header,
            "    pid = tl.program_id(0)",
            "    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)",
            "    mask = offsets < n_elements",
            *load_lines,
            *body_lines,
            f"    tl.store(out_ptr + offsets, {var[output]}, mask=mask)",
            "",
        ]
    )
    return src, name


def compile_fused(fnode, block_size=1024):
    """
    Turn a fused node into a python callable that runs its triton kernel.

    The returned function takes the external operands (in attrs['inputs'] order) as cuda
    tensors and returns the result. NOTE: needs triton + a gpu, so this only works on
    the server.
    """
    src, name = emit_kernel(fnode)

    ns = {}
    exec(src, ns)  # defines the @triton.jit kernel
    kernel = ns[name]

    import triton

    def call(*tensors):
        assert len(tensors) == len(fnode.attrs["inputs"]), "wrong number of operands"
        out = torch.empty_like(tensors[0])
        n = out.numel()
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
        kernel[grid](*tensors, out, n, BLOCK_SIZE=block_size)
        return out

    return call
