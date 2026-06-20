"""
graphfuse IR
"""

from __future__ import annotations


class Node:
    """One operation in the graph.
      op     : str          the operation: 'input', 'add', 'mul', 'relu', 'matmul', 'sum'
      inputs : list[Node]   the nodes whose results this op consumes (its operands)
      attrs  : dict         op-specific extras, e.g. {'dim': 1} for a reduction
      name   : str | None   used only by 'input' nodes, to match a runtime tensor later
    """

    def __init__(self, op, inputs=(), attrs=None, name=None):
        self.op = op
        self.inputs = list(inputs)
        self.attrs = attrs or {}
        self.name = name

    @property
    def is_input(self):
        return self.op == "input"
    
    @property
    def is_const(self):
        return self.op == "const"

    def __repr__(self):
        # make nicer later
        return self.name if self.is_input else self.op


class Graph:
    """Builder API. Each method CREATES a node, registers it, and RETURNS it, so that
    expressions compose the way the math reads:

        out = g.relu(g.add(g.mul(a, b), c))
    """

    def __init__(self, outputs=()):
        self.nodes = []  # every node created, in construction order
        self.outputs = list(outputs) # the output nodes of the program

    def _add(self, node):
        """Register a freshly-made node and hand it back. Call this from every builder
        method so `self.nodes` always holds the complete set."""
        self.nodes.append(node)
        return node

    def input(self, name):
        """Placeholder for a runtime tensor. Value is unknown until run() binds it
        by name from the caller's dict. The same graph can be run with different
        values each time"""
        return self._add(Node("input", name=name))

    def const(self, val):
        """A literal value baked into the graph at construction time. Already known
        so no binding required. Used during constant folding"""
        return self._add(Node("const", attrs={"val": val}))

    def add(self, a, b):
        return self._add(Node("add", inputs=[a, b]))

    def mul(self, a, b):
        return self._add(Node("mul", inputs=[a, b]))

    def relu(self, x):
        return self._add(Node("relu", inputs=[x]))

    def matmul(self, a, b):
        return self._add(Node("matmul", inputs=[a, b]))

    def sum(self, x, dim=None):
        return self._add(Node("sum", inputs=[x], attrs={"dim": dim}))


def topo_sort(output):
    """Return every node that `output` depends on, ordered so that each node appears
    AFTER all of its inputs. This is the order the interpreter (and, later, codegen)
    must follow
    """
    visited = set()
    result = []

    def dfs(node):
        visited.add(node)
        

        for nbr in node.inputs:
            if nbr not in visited:
                dfs(nbr)

        result.append(node) # add the node only after its dependencies have been added


    dfs(output)
    return result