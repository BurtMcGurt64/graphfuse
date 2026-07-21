"""
Walks the graph in topological order and executes each node by calling the
corresponding torch operation
"""

import torch
from .graph import topo_sort

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

        elif node.op == "add":
            a, b = node.inputs
            env[node] = env[a] + env[b]
 
        elif node.op == "mul":
            a, b = node.inputs
            env[node] = env[a] * env[b]
 
        elif node.op == "relu":
            (x,) = node.inputs
            env[node] = torch.relu(env[x])
 
        elif node.op == "matmul":
            a, b = node.inputs
            env[node] = env[a] @ env[b]
 
        elif node.op == "sum":
            (x,) = node.inputs
            env[node] = torch.sum(env[x], dim=node.attrs["dim"])
        
    return env[output]