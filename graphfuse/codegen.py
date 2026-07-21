"""
Triton codegen for fused pointwise groups

We only lower the 'fused' nodes that fuse() produces, and those are pointwise by
construction (add / mul / relu over same-shape tensors). Anything else still runs
through torch
"""

import hashlib
import importlib.util
import os
import sys
import tempfile

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


def _load_kernel(src, name):
    """
    Import the generated kernel from a real .py file and hand back the function.

    We can't just exec() the source: triton's @jit reads the kernel's source via
    inspect.getsourcelines() (for compilation + caching), and a function built from an
    exec'd string has no file on disk, so that blows up with "should be defined in a
    Python file". So we write the source out and import it like a normal module.

    The filename is a hash of the source, so identical kernels land in the same file and
    triton's source-keyed cache stays happy.
    """
    digest = hashlib.sha1(src.encode()).hexdigest()[:12]
    mod_name = f"graphfuse_gen_{digest}"
    path = os.path.join(tempfile.gettempdir(), mod_name + ".py")

    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(src)

    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod  # so inspect can find it back
    spec.loader.exec_module(mod)
    return getattr(mod, name)


def compile_fused(fnode, block_size=1024):
    """
    Turn a fused node into a python callable that runs its triton kernel.

    The returned function takes the external operands (in attrs['inputs'] order) as cuda
    tensors and returns the result. NOTE: needs triton + a gpu, so this only works on
    the server.
    """
    src, name = emit_kernel(fnode)
    kernel = _load_kernel(src, name)

    import triton

    def call(*tensors):
        assert len(tensors) == len(fnode.attrs["inputs"]), "wrong number of operands"
        out = torch.empty_like(tensors[0])
        n = out.numel()
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
        kernel[grid](*tensors, out, n, BLOCK_SIZE=block_size)
        return out

    return call
