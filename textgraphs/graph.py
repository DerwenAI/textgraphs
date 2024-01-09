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


    def make_node (  # pylint: disable=R0913,R0914
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
Lookup and return a `Node` object.
By default, link matching keys into the same node.
Otherwise instantiate a new node if it does not exist already.

    tokens:
list of parsed tokens

    key:
lemma key (invariant)

    span:
token span for the parsed entity

    kind:
the kind of this `Node` object

    text_id:
text (top-level document) identifier

    para_id:
paragraph identitifer

    sent_id:
sentence identifier

    label:
node label (for a new object)

    length:
length of token span

    linked:
flag for whether this links to an entity

    returns:
the constructed `Node` object
        """
        token_id: int = 0
        token_text: str = key
        token_pos: str = "PROPN"

        if span is not None:
            token_id = span.i
            token_text = span.text
            token_pos = span.pos_

        location: typing.List[ int ] = [  # type: ignore
            text_id,
            para_id,
            sent_id,
            token_id,
        ]

        if not linked:
            # construct a placeholder node (stopwords)
            self.nodes[key] = Node(
                len(self.nodes),
                key,
                span.text,
                span.pos_,
                kind,
                span = span,
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
                token_text,
                token_pos,
                kind,
                span = span,
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

    src_node:
source node in the triple

    dst_node:
destination node in the triple

    kind:
the kind of this `Edge` object

    rel:
relation label

    prob:
probability of this `Edge` within the graph

    debug:
debugging flag

    returns:
the constructed `Edge` object; this may be `None` if the input parameters indicate skipping the edge
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


    def dump_lemma_graph (
        self
        ) -> str:
        """
Dump the _lemma graph_ as a JSON string in _node-link_ format,
suitable for serialization and subsequent use in JavaScript,
Neo4j, Graphistry, etc.

Make sure to call beforehand: `TextGraphs.calc_phrase_ranks()`

    returns:
a JSON representation of the exported _lemma graph_ in
[_node-link_](https://networkx.org/documentation/stable/reference/readwrite/json_graph.html)
format
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
            nx_node["length"] = node.length
            nx_node["annotated"] = node.annotated

        # emulate a node-link format serialization, since the call to
        # `NetworkX.node_link_data()` drop some properties, such as `key`
        edge_list: typing.List[ dict ] = []

        for src, dst, props in self.lemma_graph.edges.data():
            props["source"] = src
            props["target"] = dst
            edge_list.append(props)

        node_link: dict = {
            "directed": True,
            "multigraph": True,
            "nodes": [
                props
                for node_id, props in self.lemma_graph.nodes.data()
            ],
            "edges": edge_list,
            "graph": {}
        }

        return json.dumps(
            node_link,
            sort_keys = True,
            indent = 2,
            separators = ( ",", ":" ),
        )
