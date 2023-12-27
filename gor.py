#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Experiment with deserializing a node-link graph,
then transform it into a _graph of relations_
"""

from collections import defaultdict
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


NODE_LIST: typing.List[ DeserNode ] = []


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

            NODE_LIST.append(DeserNode(
                node,
                base,
                label,
            ))

        elif kind in [ "ent" ]:
            base = n_attr["iri"]
            label = n_attr["title"]

            NODE_LIST.append(DeserNode(
                node,
                base,
                label,
            ))

        elif kind in [ "chu", "dep", "lem" ]:
            base = n_attr["pos"]
            label = n_attr["title"]

            NODE_LIST.append(DeserNode(
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

    with open (lemma_path, "r", encoding = "utf-8") as fp:  # pylint: disable=C0103,W0621
        dat: dict = json.load(fp)
        lemma_graph: nx.MultiDiGraph = nx.node_link_graph(dat)
        gor_transform(lemma_graph)

    for edge in lemma_graph.edges:  # pylint: disable=W0621
        ic(lemma_graph.edges[edge])


if __name__ == "__main__":
    graph: textgraphs.SimpleGraph = textgraphs.SimpleGraph()
    rel_list: typing.List[ str ] = []
    head_links: typing.Dict[ int, set ] = defaultdict(set)
    tail_links: typing.Dict[ int, set ] = defaultdict(set)

    with open("gor.json", "r", encoding = "utf-8") as fp:  # pylint: disable=W0621
        triples: dict = json.load(fp)

        for src_name, links in triples.items():
            src_node: textgraphs.Node = graph.make_node(
                [],
                src_name,
                None,
                textgraphs.NodeEnum.ENT,
                0,
                0,
                0,
            )

            for rel_name, dst_name in links:
                dst_node: textgraphs.Node = graph.make_node(
                    [],
                    dst_name,
                    None,
                    textgraphs.NodeEnum.ENT,
                    0,
                    0,
                    0,
                )

                edge: textgraphs.Edge = graph.make_edge(  # type: ignore  # pylint: disable=W0621
                    src_node,
                    dst_node,
                    textgraphs.RelEnum.SYN,
                    rel_name,
                    1.0,
                )

                if rel_name not in rel_list:
                    rel_list.append(rel_name)

                #ic(src_node.text, edge.rel, dst_node.text)

                rel_id: int = rel_list.index(rel_name)
                head_links[rel_id].add( edge.dst_node )
                tail_links[rel_id].add( edge.src_node )

    #print(graph.nodes)
    #print(graph.edges)
    #print(rel_list)
    print(head_links)
    print(tail_links)
