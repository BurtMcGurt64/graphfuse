# Design Decisions


## Overview
This is a small compiler for tensor graphs. Given an expression in PyTorch, we dynamically build a directed acyclic graph, run passes over it, and lower the optimized graph into Triton kernels.

The high-level workflow is:

PyTorch Expression (ex: `relu((a * b) + c)`) --> Directed Acyclic Graph --> Optimization (Dead Code Elimination, Fusing Operations) --> Triton Kernels

## Design Specification
### Data Representation
A tensor graph is represented by the class `Graph` in `graph.py`, a collection of `Node` objects. Each `Node` object corresponds to a single operation or input, with fields describing the operation type, what inputs it requires, and some operation-specific arguments such as `dim` for a reduction. 

In other words, we can think of an expression such as `relu((a * b) + c)` as described by a directed acyclic graph, as shown below:

input(a) ─┐
           ├─→ mul ─┐
input(b) ─┘         ├─→ add ─→ relu
input(c) ───────────┘

We execute the instruction by traversing the corresponding graph.

### IR & Execution Separation
The graph we created is the compiler's intermediate representation (IR). We build the graph and optimize over it *before* execution. This separation allows us to optimize the graph, as the kernels that will eventually execute this graph don't exist yet while the graph is being built and rewritten. Another advantage of this separation is that rewriting the graph does not require any GPU compute/runtime, as the graph is simply represented by a Python class.

### Graph Traversal
To execute the graph we need to run nodes in an order where every node comes after the nodes it depends on. We do this by DFS postorder starting from the output node. Also, because a node can feed into multiple operations downstream (we are working on a DAG, not a tree), the traversal can reach it more than once. We solve this by maintaining a `visited` set to make sure that each node is emitted exactly once.


### Optimization Passes (`passes.py`)
Each optimization pass generates a new `Graph` object instead of mutating the original.

**Dead Code Elimination** - Since the tensor graph already encodes information about node dependency, we run `topo_sort()` over all output nodes in the graph, and construct a new graph from those nodes. This eliminates any nodes that are not a dependency of an output node.

**Constant Folding** - We constant fold only over nodes whose inputs are `const`. Since `const` nodes must be leaves (a `const` node is generated with no inputs), this is a safe operation. A `node_map` is maintained between the original and constant folded graph. All other nodes are cloned, with their inputs changed based on the new folded graph defined by `node_map`.

**Common Subexpression Elimination** - We maintain a hashmap to track seen nodes and their signature, where signature(node) = (operation, remapped inputs, attributes, node name). Then for each node in the walk, if it has been seen, we replace it with the already seen node. For example, 

```
mul_1(a, b)
mul_2(a, b)
add(mul_1, mul_2)
```

is mapped to:

```
mul_1(a, b)
add(mul_1, mul_1)
```

### Project Layout
Everything lives under the `graphfuse/` package. The modules import each other with package-relative imports (`from .graph import ...`) instead of flat ones, so the project runs the same from anywhere, not just from inside the folder. `__init__.py` only re-exports the pure-python IR (`Node`, `Graph`, `topo_sort`) on purpose - we don't want a plain `import graphfuse` to drag in torch/triton until you actually reach for the interpreter or codegen.

To see the current pipeline end to end (build a graph, run it, watch the passes rewrite it):

```
uv run python -m graphfuse.demo
```


