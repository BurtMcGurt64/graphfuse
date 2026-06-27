"""
Optimization passes over the tensor graph
    - DCE, cfold, CSE

We return graphs for each optimization so they are composable:
    - dce(cfold(cse(graph)))

"""


from graph import Node, Graph, topo_sort
from interpreter import run

def dce(graph):
    """
    Dead code elimination over tensor graph - eliminate unused operations.

    The observation is that topo_sort(output) gives us all operations and inputs required
    for `output`, which is exactly the live nodes.

    However, a program might have multiple outputs, such as the program "x = relu(a), y = relu(a*b)". 
    So we need topo_sort on all outputs to get all live nodes.
    """

    live = []
    seen = set()
    for out in graph.outputs:
        for node in topo_sort(out):
            if node not in seen:
                seen.add(node)
                live.append(node)
    dce_graph = Graph(outputs=graph.outputs)
    dce_graph.nodes = live
    return dce_graph



def cfold(graph):
    """
    Constant folding over tensor graph. If an operation only is made up of constant inputs,
    then we compute while constructing the graph. 
    """
    visited = set()
    folded = Graph()
    node_map = {} # old node -> new node to be constructed in folded graph

    for out in graph.outputs:
        # run topo_sort on each output node to get its dependencies
        for node in topo_sort(out):
            if node not in visited:
                visited.add(node)

                if not node.is_const and not node.is_input and all(node_map[inp].is_const for inp in node.inputs):
                    # this is currently O(n^2) - we don't need to run every iteration
                    value = run({}, node)
                    node_map[node] = folded.const(value)
        
                else:
                    clone = Node(node.op, inputs=[node_map[inp] for inp in node.inputs], attrs=node.attrs, name=node.name)
                    folded._add(clone)
                    node_map[node] = clone


    folded.outputs = [node_map[out] for out in graph.outputs]
    return folded


def cse(graph):
    """
    Common subexpression elimination over tensor graph. If the current node has already been seen, 
    we substitute the expression it corresponds to. 
    """

    def _signature(node, node_map):
        # allows us to compare nodes
        # sig(node) = (operation, inputs, attributes, name)
        inputs = tuple(node_map[i] for i in node.inputs)
        return (node.op, inputs, tuple(sorted(node.attrs.items())), node.name)

    seen = {} # signature -> node
    visited = set()
    cse_graph = Graph()
    node_map = {}

    for out in graph.outputs:
        for node in topo_sort(out):
            if node not in visited:
                visited.add(node)
                sig = _signature(node, node_map)

                if sig in seen:
                    node_map[node] = seen[sig]
                
                else:
                    clone = Node(node.op, inputs=[node_map[inp] for inp in node.inputs], attrs=node.attrs, name=node.name)
                    seen[sig] = clone
                    node_map[node] = clone
                    cse_graph._add(clone)
    
    cse_graph.outputs = [node_map[out] for out in graph.outputs]

    return cse_graph