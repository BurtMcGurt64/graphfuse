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

### Fusion (`fuse.py`)
The reason fusion matters is memory bandwidth. Running `relu((a*b)+c)` as three separate kernels means writing `a*b` back to GPU memory, reading it in again for the add, writing that out, reading it in again for the relu. Every intermediate makes a round trip through global memory. If we fuse the whole chain into one kernel, the intermediates stay in registers and we only touch memory once at the start (loading a, b, c) and once at the end (storing the result).

So we want to group pointwise ops that chain into each other. We already have `find_sink` / `build_groups` to figure out the grouping - a group is every node that flows into the same sink. `fuse()` turns each group into a single `fused` node.

A node is only pulled into its user's group if it's pointwise, has exactly one user, and that user is pointwise (this is the `find_sink` rule). The nice consequence is that a non-sink member of a group is never read from outside the group - if it had another reader it would have had a second user and become its own sink. So the only value a group exposes to the rest of the graph is its sink, which is exactly what the `fused` node produces.

Non-pointwise nodes (inputs, consts, matmul, sum) can't be fused, so they just get cloned through unchanged. That means a matmul or reduction in the middle naturally splits the pointwise chains around it into separate groups.

A `fused` node keeps its insides in `attrs` so nothing downstream loses information:
- `attrs['nodes']` - the internal pointwise ops, in topo order
- `attrs['output']` - which internal node is the group's result (the sink)
- `attrs['inputs']` - the external operands, lined up with `node.inputs`

The interpreter knows how to run a `fused` node: it seeds a local env with the external operands and walks the internal ops. This lets us check that fusion didn't change the answer by running the fused graph against the original.

### Triton Codegen (`codegen.py`)
This is where the graph finally becomes real kernels. For now we only lower `fused` nodes, and those are pointwise by construction, so the codegen is simple: each external operand is a pointer we `tl.load` once, each internal op is one line of arithmetic over values we've already named, and the sink gets `tl.store`d back out. Everything in between lives in registers, which is the whole point of fusing.

`emit_kernel` is pure string generation - it just walks the fused node's internal ops in topo order and prints a triton expression for each one. `compile_fused` is what actually jits and launches it, so it imports triton lazily (triton only exists on the gpu server, not on the mac we edit from).

