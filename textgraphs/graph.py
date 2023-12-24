#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This class implements a generic, in-memory graph data structure used
to represent the _lemma graph_.

see copyright/license https://huggingface.co/spaces/DerwenAI/textgraphs/blob/main/README.md
"""

from collections import OrderedDict
import json
import typing

from icecream import ic  # pylint: disable=E0401
import networkx as nx  # pylint: disable=E0401
import spacy  # pylint: disable=E0401

from .elem import Edge, Node, NodeEnum, RelEnum


######################################################################
## class definitions

class SimpleGraph:
    """
An in-memory graph used to build a `MultiDiGraph` in NetworkX.
    """

    def __init__ (
        self
        ) -> None:
        """
Constructor.
        """
        self.nodes: typing.Dict[ str, Node ] = OrderedDict()
        self.edges: typing.Dict[ str, Edge ] = {}
        self.lemma_graph: nx.MultiDiGraph = nx.MultiDiGraph()


    def reset (
        self
        ) -> None:
        """
Re-initialize the data structures, resetting all but the configuration.
        """
        self.nodes = OrderedDict()
        self.edges = {}
        self.lemma_graph = nx.MultiDiGraph()


    def make_node (  # pylint: disable=R0913
        self,
        tokens: typing.List[ Node ],
        key: str,
        span: spacy.tokens.token.Token,
        kind: NodeEnum,
        text_id: int,
        para_id: int,
        sent_id: int,
        *,
        label: typing.Optional[ str ] = None,
        length: int = 1,
        linked: bool = True,
        ) -> Node:
        """
Lookup and return a `Node` object:

    * default: link matching keys into the same node
    * instantiate a new node if it does not exist already
        """
        location: typing.List[ int ] = [  # type: ignore
            text_id,
            para_id,
            sent_id,
            span.i,
        ]

        if not linked:
            # construct a placeholder node (stopwords)
            self.nodes[key] = Node(
                len(self.nodes),
                key,
                span,
                span.text,
                span.pos_,
                kind,
                loc = [ location ],
                length = length,
            )

        elif key in self.nodes:
            # link to previously constructed entity node
            self.nodes[key].loc.append(location)
            self.nodes[key].count += 1

        # construct a new node for entity or lemma
        else:
            self.nodes[key] = Node(
                len(self.nodes),
                key,
                span,
                span.text,
                span.pos_,
                kind,
                loc = [ location ],
                label = label,
                length = length,
                count = 1,
            )

        node: Node = self.nodes.get(key)  # type: ignore

        if kind not in [ NodeEnum.CHU, NodeEnum.IRI ]:
            tokens.append(node)

        return node  # type: ignore


    def make_edge (  # pylint: disable=R0913
        self,
        src_node: Node,
        dst_node: Node,
        kind: RelEnum,
        rel: str,
        prob: float,
        *,
        debug: bool = False,
        ) -> typing.Optional[ Edge ]:
        """
Lookup an edge, creating a new one if it does not exist already,
and increment the count if it does.
        """
        key: str = ".".join([
            str(src_node.node_id),
            str(dst_node.node_id),
            rel.replace(" ", "_"),
            str(kind.value),
        ])

        if debug:
            ic(key)

        if key in self.edges:
            self.edges[key].count += 1

        elif src_node.node_id != dst_node.node_id:
            # preclude cycles in the graph
            self.edges[key] = Edge(
                src_node.node_id,
                dst_node.node_id,
                kind,
                rel,
                prob,
            )

        if debug:
            ic(self.edges.get(key))

        return self.edges.get(key)


    def construct_lemma_graph (
        self,
        *,
        debug: bool = False,
        ) -> None:
        """
Construct the base level of the _lemma graph_ from the collected
elements. This gets represented in `NetworkX` as a directed graph
with parallel edges.
        """
        # add the nodes
        self.lemma_graph.add_nodes_from([
            node.node_id
            for node in self.nodes.values()
        ])

        # populate the minimum required node properties
        for node_key, node in self.nodes.items():
            nx_node = self.lemma_graph.nodes[node.node_id]
            nx_node["title"] = node_key
            nx_node["size"] = node.count
            nx_node["value"] = node.weight

            if debug:
                ic(nx_node)

        # add the edges and their properties
        self.lemma_graph.add_edges_from([
            (
                edge.src_node,
                edge.dst_node,
                {
                    "kind": str(edge.kind),
                    "title": edge.rel,
                    "weight": float(edge.count),
                    "prob": edge.prob,
                    "count": edge.count,
                },
            )
            for edge_key, edge in self.edges.items()
        ])


    def dump_lemma_graph (
        self,
        ) -> str:
        """
Dump the _lemma graph_ as a JSON string in _node-link_ format,
suitable for serialization and subsequent use in JavaScript,
Neo4j, Graphistry, etc.

Make sure to call beforehand:

  * `TextGraphs.calc_phrase_ranks()`
        """
        # populate the optional node properties
        for node in self.nodes.values():
            nx_node = self.lemma_graph.nodes[node.node_id]
            nx_node["name"] = node.text
            nx_node["kind"] = str(node.kind)
            nx_node["iri"] = node.label
            nx_node["subobj"] = node.sub_obj
            nx_node["pos"] = node.pos
            nx_node["loc"] = str(node.loc)

        return json.dumps(
            nx.node_link_data(self.lemma_graph),
            sort_keys = True,
            indent =  2,
            separators = ( ",", ":" ),
        )
