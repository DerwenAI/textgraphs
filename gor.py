#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Experiment with deserializing a node-link graph,
then transform it into a _graph of relations_
"""

from dataclasses import dataclass
import pathlib
import json
import typing

from icecream import ic  # pylint: disable=E0401
import networkx as nx  # pylint: disable=E0401

import textgraphs


@dataclass(order=False, frozen=False)
class DeserNode:  # pylint: disable=R0902
    """
A data class representing one deserialized Node.
    """
    node_id: int
    base: str
    label: str


node_list: typing.List[ DeserNode ] = []


def gor_transform (
    lemma_graph: nx.MultiDiGraph,
    ) -> None:
    """
Transform a `MultiDiGraph` into a _graph of relations_
    """
    for node in lemma_graph.nodes:
        n_attr: dict = lemma_graph.nodes[node]
        kind: str = n_attr["kind"]

        if kind in [ "iri" ]:
            base: str = kind
            label: str = n_attr["iri"]

            node_list.append(DeserNode(
                node,
                base,
                label,
            ))

        elif kind in [ "ent" ]:
            base = n_attr["iri"]
            label = n_attr["title"]

            node_list.append(DeserNode(
                node,
                base,
                label,
            ))

        elif kind in [ "chu", "dep", "lem" ]:
            base = n_attr["pos"]
            label = n_attr["title"]

            node_list.append(DeserNode(
                node,
                base,
                label,
            ))

        else:
            ic(kind, n_attr)


def load_lemma () -> None:
    """
Load a serialized lemma graph from JSON.
    """
    lemma_path: pathlib.Path = pathlib.Path("lemma.json")

    with open (lemma_path, "r", encoding = "utf-8") as fp:  # pylint: disable=C0103
        dat: dict = json.load(fp)
        lemma_graph: nx.MultiDiGraph = nx.node_link_graph(dat)
        gor_transform(lemma_graph)


    for edge in lemma_graph.edges:
        ic(lemma_graph.edges[edge])


if __name__ == "__main__":
    graph: textgraphs.SimpleGraph = textgraphs.SimpleGraph()
