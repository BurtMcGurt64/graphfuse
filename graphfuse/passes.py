"""
Optimization passes over the tensor graph
    - DCE, cfold, CSE

We return graphs for each optimization so they are composable:
    - dce(cfold(cse(graph)))
"""


from graph import Graph, topo_sort
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

    node_map = {} # old node -> new node to be constructed in folded graph

    for out in graph.outputs:
        # run topo_sort on each output node to get its dependencies
        for node in topo_sort(out):
            if node not in visited:
                visited.add(node)

                if not node.is_const and not node.is_input and all(node_map[inp].is_const for inp in node.inputs):

                    node_map[node] = graph.const(value)



    for node in graph.nodes:
        if not node.is_const and not node.is_input and all(inp.is_const for inp in node.inputs): # we need ALL inputs to be constants
            value = run({}, node)

            const_node = graph.const(value)
            
            # replace all places where `node` appears as an input to other nodes
            for _node in graph.nodes:
                if node in _node.inputs:
                    _node.inputs = [const_node if inp is node else inp for inp in _node.inputs]
            
            graph.outputs = [const_node if out is node else out for out in graph.outputs]
                    
    return graph


