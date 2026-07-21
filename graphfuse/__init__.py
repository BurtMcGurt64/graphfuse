"""
graphfuse - a small compiler for tensor graphs
"""

from .graph import Node, Graph, topo_sort

__all__ = ["Node", "Graph", "topo_sort"]
