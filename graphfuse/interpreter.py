"""
Walks the graph in topological order and executes each node by calling the
corresponding torch operation
"""

import torch
from .graph import topo_sort


def _eval(node, resolve):
    """
    Compute a single op given `resolve`, a function old_input_node -> tensor.

    Pulled out of run() so the fused-node path can reuse the exact same op logic on
    its internal subgraph.
    """
    op = node.op

    if op == "add":
        a, b = node.inputs
        return resolve(a) + resolve(b)

    elif op == "mul":
        a, b = node.inputs
        return resolve(a) * resolve(b)

    elif op == "relu":
        (x,) = node.inputs
        return torch.relu(resolve(x))

    elif op == "matmul":
        a, b = node.inputs
        return resolve(a) @ resolve(b)

    elif op == "sum":
        (x,) = node.inputs
        return torch.sum(resolve(x), dim=node.attrs["dim"])

    raise ValueError(f"don't know how to eval op {op!r}")


def _run_fused(node, env):
    """
    Execute a fused node by just running its internal subgraph.

    env already holds the fused node's external operands (its node.inputs are the
    nodes that produce them), so we seed a local env with those and walk the internal
    ops in topo order.
    """
    local = {}
    for ext_old, producer in zip(node.attrs["inputs"], node.inputs):
        local[ext_old] = env[producer]

    for member in node.attrs["nodes"]:
        local[member] = _eval(member, local.__getitem__)

    return local[node.attrs["output"]]


def run(inputs, output):
    """
    Execute the graph that produces output.

    Inputs: dictionary mapping tensor name -> tensor object
    """
    walk = topo_sort(output)

    env = {} # node -> tensor

    for node in walk:
        if node.op == "input":
            env[node] = inputs[node.name]

        elif node.op == "const":
            env[node] = node.attrs["val"]

        elif node.op == "fused":
            env[node] = _run_fused(node, env)

        else:
            env[node] = _eval(node, env.__getitem__)

    return env[output]
