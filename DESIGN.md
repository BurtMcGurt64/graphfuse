# Design Decisions


## Overview
This is a small compiler for tensor graphs. Given an expression in PyTorch, we dynamically build a directed acyclic graph, run passes over it, and lower the optimized graph into Triton kernels.

The high-level workflow is:

PyTorch Expression (ex: relu((a * b) + c)) --> Directed Acyclic Graph --> Optimization (Dead Code Elimination, Fusing Operations) --> Triton Kernels

## Design Specification
### Data Representation
A tensor graph is represented by the class `Graph` in `graph.py`, a collection of `Node` objects. Each `Node` object corresponds to a single operation or input, with fields describing the operation type, what inputs it requires, and some operation-specific arguments such as `dim` for a reduction. 

In other words, we can think of an expression such as `relu((a * b) + c)` as described by a directed acyclic graph, as shown below:

input(a) ─┐
           ├─→ mul ─┐
input(b) ─┘         ├─→ add ─→ relu
input(c) ───────────┘

### IR & Execution Separation
The graph we created is the compiler's intermediate representation (IR). We build the graph and optimize over it *before* execution. This separation allows us to optimize the graph, as the kernels that will eventually execute this graph don't exist yet while the graph is being built and rewritten. Another advantage of this separation is that rewriting the graph does not require any GPU compute/runtime, as the graph is simply represented by a Python class.

### Graph Traversal
To execute the graph we need to run nodes in an order where every node comes after the nodes it depends on. We do this by DFS postorder starting from the output node. Also, because a node can feed into multiple operations downstream (we are working on a DAG, not a tree), the traversal can reach it more than once. We solve this by maintaining a `visited` set to make sure that each node is emitted exactly once.


## Optimization Passes
