from graph import topo_sort, Node, Graph
from interpreter import run
from passes import dce, cfold


import torch


g = Graph()
out = g.add(g.const(2), g.const(3))
g.outputs = [out]

result = dce(cfold(g))
print([n.op for n in result.nodes])
print([n.op for n in result.outputs])
print(result.outputs[0].attrs.get("val"))