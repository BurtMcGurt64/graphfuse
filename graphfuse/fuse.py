"""
Fuse nodes
"""

from collections import defaultdict
from graph import Graph, topo_sort

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
