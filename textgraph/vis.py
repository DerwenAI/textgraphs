#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright ©2023 Derwen, Inc. All rights reserved.

"""
Visualization methods based on `PyVis`
"""

from dataclasses import dataclass
import typing

from icecream import ic  # pylint: disable=E0401
import networkx as nx  # pylint: disable=E0401
import pyvis  # pylint: disable=E0401

from .elem import Node, Edge


@dataclass(order=False, frozen=True)
class NodeStyle:  # pylint: disable=R0902
    """
Dataclass used for styling PyVis nodes.
    """
    color: str
    shape: str

DIM_NODE: NodeStyle = NodeStyle(
    color = "hsla(72, 19%, 90%, 0.4)",
    shape = "star",
)


class RenderPyVis:  # pylint: disable=R0903
    """
Render the _lemma graph_ as a `PyVis` network.
    """

    def __init__ (
        self,
        nodes: typing.Dict[ str, Node ],
        edges: typing.Dict[ str, Edge ],
        lemma_graph: nx.MultiDiGraph,
        ) -> None:
        """
Constructor.
        """
        self.nodes: typing.Dict[ str, Node ] = nodes
        self.edges: typing.Dict[ str, Edge ] = edges
        self.lemma_graph: nx.MultiDiGraph = lemma_graph


    def build_lemma_graph (
        self,
        *,
        debug: bool = True,
        ) -> pyvis.network.Network:
        """
Prepare the structure of the `NetworkX` graph to use for building
and returning a `PyVis` network to render.
        """
        for node in self.nodes.values():
            neighbors: int = 0

            try:
                neighbors = len(list(nx.neighbors(self.lemma_graph, node.node_id)))
            except Exception:  # pylint: disable=W0718
                pass

            nx_node = self.lemma_graph.nodes[node.node_id]
            nx_node["value"] = node.weight
            nx_node["size"] = node.count
            nx_node["neighbors"] = neighbors

            if node.count < 1:
                nx_node["kind"] = 0
                nx_node["shape"] = "star"
                nx_node["label"] = ""
                nx_node["title"] = node.text
                nx_node["color"] = "hsla(72, 19%, 90%, 0.4)"
            elif node.kind is not None:
                nx_node["kind"] = 1
                nx_node["shape"] = "circle"
                nx_node["label"] = node.text
                nx_node["color"] = "#d2d493"
            else:
                nx_node["kind"] = 2
                nx_node["shape"] = "square"
                nx_node["label"] = node.text
                nx_node["color"] = "#c083bb"

            if debug:
                ic(node.count, node, nx_node)

        # prepare the edge labels
        edge_labels: dict = {}

        for edge in self.edges.values():
            edge_labels[(edge.src_node, edge.dst_node,)] = ( edge.kind.value, edge.rel, )

        # build the network
        vis_graph: pyvis.network.Network = pyvis.network.Network()
        vis_graph.from_nx(self.lemma_graph)

        for pv_edge in vis_graph.get_edges():
            edge_key = ( pv_edge["from"], pv_edge["to"], )
            edge_info = edge_labels.get(edge_key)

            if edge_info[0] == 0:  # type: ignore
                pv_edge["color"] = "ltgray"  # type: ignore
                pv_edge["width"] = 0  # type: ignore
                pv_edge["label"] = ""  # type: ignore
            else:
                pv_edge["label"] = edge_info[1]  # type: ignore

        return vis_graph
