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

from .elem import Edge, Node, NodeKind, RelEnum


@dataclass(order=False, frozen=True)
class NodeStyle:  # pylint: disable=R0902
    """
Dataclass used for styling PyVis nodes.
    """
    kind: NodeKind
    shape: str
    color: str

NODE_STYLES: typing.List[ NodeStyle ] = [
    NodeStyle(
        kind = NodeKind.DEP,
        shape = "star",
        color = "hsla(72, 19%, 90%, 0.4)",
    ),
    NodeStyle(
        kind = NodeKind.LEM,
        shape = "square",
        color = "hsl(306, 45%, 57%)",
    ),
    NodeStyle(
        kind = NodeKind.ENT,
        shape = "circle",
        color = "hsl(65, 46%, 58%)",
    ),
]


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
        for node_key, node in self.nodes.items():
            nx_node = self.lemma_graph.nodes[node.node_id]

            if node.weight == 0.0:
                node_kind: NodeKind = NodeKind.DEP
                nx_node["label"] = ""

            elif node.kind is None:
                node_kind = NodeKind.LEM
                nx_node["label"] = node.text

            else:
                node_kind = NodeKind.ENT
                nx_node["label"] = node.text

            nx_node["kind"] = node_kind
            nx_node["title"] = node_key
            nx_node["value"] = node.weight
            nx_node["size"] = node.count
            nx_node["shape"] = NODE_STYLES[node_kind].shape
            nx_node["color"] = NODE_STYLES[node_kind].color

            if debug:
                ic(node.count, node, nx_node)

        # prepare the edge labels
        edge_labels: dict = {}

        for edge in self.edges.values():
            edge_labels[(edge.src_node, edge.dst_node,)] = ( edge.kind, edge.rel, )

        # build the network
        pv_graph: pyvis.network.Network = pyvis.network.Network()
        pv_graph.from_nx(self.lemma_graph)

        for pv_edge in pv_graph.get_edges():
            edge_key = ( pv_edge["from"], pv_edge["to"], )
            edge_info = edge_labels.get(edge_key)
            pv_edge["title"] = edge_info[1]  # type: ignore

            if edge_info[0] == RelEnum.DEP:  # type: ignore
                pv_edge["arrows"] = "to" # type: ignore
                pv_edge["color"] = "ltgray"  # type: ignore
                pv_edge["width"] = 0  # type: ignore

        return pv_graph
