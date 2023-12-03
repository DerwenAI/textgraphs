#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright ©2023 Derwen, Inc. All rights reserved.

"""
Visualization methods based on `PyVis`
"""

from dataclasses import dataclass
import typing

from icecream import ic  # pylint: disable=E0401
import matplotlib.colors as mcolors  # pylint: disable=E0401
import networkx as nx  # pylint: disable=E0401
import pyvis  # pylint: disable=E0401
import wordcloud  # pylint: disable=E0401

from .elem import Edge, Node, NodeEnum, RelEnum


@dataclass(order=False, frozen=True)
class NodeStyle:  # pylint: disable=R0902
    """
Dataclass used for styling PyVis nodes.
    """
    label: NodeEnum
    shape: str
    color: str

NODE_STYLES: typing.List[ NodeStyle ] = [
    NodeStyle(
        label = NodeEnum.DEP,
        shape = "star",
        color = "hsla(72, 19%, 90%, 0.4)",
    ),
    NodeStyle(
        label = NodeEnum.LEM,
        shape = "square",
        color = "hsl(306, 45%, 57%)",
    ),
    NodeStyle(
        label = NodeEnum.ENT,
        shape = "circle",
        color = "hsl(65, 46%, 58%)",
    ),
    NodeStyle(
        label = NodeEnum.CHU,
        shape = "triangle",
        color = "hsla(72, 19%, 90%, 0.9)",
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
        for node in self.nodes.values():
            nx_node = self.lemma_graph.nodes[node.node_id]
            nx_node["shape"] = NODE_STYLES[node.kind].shape
            nx_node["color"] = NODE_STYLES[node.kind].color

            if node.kind == NodeEnum.DEP:
                nx_node["label"] = ""

            else:
                nx_node["label"] = node.text

            if node.kind == NodeEnum.CHU:
                nx_node["value"] = 0.0

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


    def draw_communities (
        self,
        *,
        spring_distance: float = 1.4,
        debug: bool = False,
        ) -> typing.Dict[ int, int ]:
        """
Cluster the communities in the _lemma graph_, then draw a
`NetworkX` graph of the notes with a specific color for each
community.
        """
        # cluster the communities, using girvan-newman
        comm_iter: typing.Generator = nx.community.girvan_newman(
            self.lemma_graph,
        )

        _ = next(comm_iter)
        next_level = next(comm_iter)
        communities: list = sorted(map(sorted, next_level))

        if debug:
            ic(communities)

        comm_map: typing.Dict[ int, int ] = {
            node_id: i
            for i, comm in enumerate(communities)
            for node_id in comm
        }

        # map from community => color
        xkcd_colors: typing.List[ str ] = list(mcolors.XKCD_COLORS.values())

        colors: typing.List[ str ] = [
            xkcd_colors[comm_map[n]]
            for n in list(self.lemma_graph.nodes())
        ]

        # prep the labels
        labels: typing.Dict[ int, str ] = {
            node.node_id: node.text
            for node in self.nodes.values()
        }

        # ¡dibuja, hombre!
        nx.draw_networkx(
            self.lemma_graph,
            pos = nx.spring_layout(
                self.lemma_graph,
                k = spring_distance / len(communities),
            ),
            labels = labels,
            node_color = colors,
            edge_color = "#bbb",
            with_labels = True,
            font_size = 9,
        )

        return comm_map


    def generate_wordcloud (
        self,
        *,
        background: str = "black",
        ) -> wordcloud.WordCloud:
        """
Generate a tag cloud from the given phrases.
        """
        terms: dict = {}
        max_weight: float = 0.0

        for node in self.nodes.values():
            if node.weight > 0.0:
                phrase: str = node.text.replace(" ", "_")
                max_weight = max(max_weight, node.weight)
                terms[phrase] = node.weight

        freq: dict = {
            phrase: round(weight / max_weight * 1000.0)
            for phrase, weight in terms.items()
        }

        cloud: wordcloud.WordCloud = wordcloud.WordCloud(
            background_color = background,
        )

        return cloud.generate_from_frequencies(freq)
