"""
Fuse nodes
"""

from collections import defaultdict
from .graph import Node, Graph, topo_sort

def build_users(nodes):
    """
    Graph stores node.inputs, but for fusion we need the reverse direction

    Here, we store the nodes that use the current node.
    """

    users = {node: [] for node in nodes}

    for node in nodes:
        for inp in node.inputs:
            if inp in users:
                users[inp].append(node)
    return users

def find_sink(node, users):
    """
    We define a sink as the node whose output is written back to GPU memory. 

    A sink must: 
        - be a pointwise operation AND
        - have exactly 1 user (if it has multiple, and its output is not written to GPU memory, then we cannot pass that to the rest of the users) AND
        - that user must also be pointwise
    """

    if node.is_pointwise and len(users[node]) == 1 and users[node][0].is_pointwise:
        return find_sink(users[node][0], users) # recurse onto the next
    
    # otherwise, the sink is the node itself
    return node

def build_groups(nodes):
    users = build_users(nodes)
    groups = defaultdict(list)

    for node in nodes:
        sink = find_sink(node, users)
        groups[sink].append(node)

    return groups


def fuse(graph):
    """
    Rewrite the graph so each pointwise chain collapses into a single 'fused' node.

    We use build_groups() to figure out which nodes belong together (everything that
    flows into the same sink), then emit one fused node per group. Non-pointwise nodes
    (inputs, consts, matmul, sum) aren't fusable, so they just get cloned through as-is.

    Because find_sink only absorbs a pointwise node when it has exactly one pointwise
    user, a non-sink member of a group is never read from outside the group. So the only
    value a group exposes is its sink - that's what the fused node produces.

    A fused node keeps its insides in attrs so the interpreter (and, later, codegen) can
    still see what it's made of:
        attrs['nodes']  : the internal pointwise ops, in topo order
        attrs['output'] : which internal node is the group's result (the sink)
        attrs['inputs'] : the external operands, lined up with node.inputs
    """
    # all live nodes, in topo order, across every output
    nodes = []
    seen = set()
    for out in graph.outputs:
        for node in topo_sort(out):
            if node not in seen:
                seen.add(node)
                nodes.append(node)

    groups = build_groups(nodes)
    node_to_sink = {}
    members_of = {}
    for sink, members in groups.items():
        members_of[sink] = set(members)
        for m in members:
            node_to_sink[m] = sink

    fused = Graph()
    produced = {}  # old node -> new node that produces its value

    for node in nodes:
        if node.is_pointwise:
            # a group only becomes a fused node once, when we reach its sink
            if node_to_sink[node] is not node:
                continue

            members = groups[node]
            inside = members_of[node]

            # collect the external operands in a stable, deduped order
            ext_inputs = []
            ext_seen = set()
            for m in members:
                for inp in m.inputs:
                    if inp not in inside and inp not in ext_seen:
                        ext_seen.add(inp)
                        ext_inputs.append(inp)

            fnode = Node(
                "fused",
                inputs=[produced[e] for e in ext_inputs],
                attrs={"nodes": members, "output": node, "inputs": ext_inputs},
            )
            fused._add(fnode)
            produced[node] = fnode

        else:
            clone = Node(node.op, inputs=[produced[i] for i in node.inputs], attrs=node.attrs, name=node.name)
            fused._add(clone)
            produced[node] = clone

    fused.outputs = [produced[out] for out in graph.outputs]
    return fused
