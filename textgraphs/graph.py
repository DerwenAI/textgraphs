#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=R0801

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

from .elem import Edge, LinkedEntity, Node, NodeEnum, RelEnum


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
            # NB: omit locations
            self.nodes[key] = Node(
                len(self.nodes),
                key,
                span.text,
                span.pos_,
                kind,
                span = span,
                length = length,
            )

        elif key in self.nodes:
            # link to previously constructed entity node
            self.nodes[key].count += 1
            self.nodes[key].loc.append(location)

            # reset the span, if this node was loaded from a
            # previous pipeline or from bootstrap definitions
            if self.nodes[key].span is None:
                self.nodes[key].span = span

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
        key: typing.Optional[ str ] = None,
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

    key:
lemma key (invariant); generate a key if this is not provided

    debug:
debugging flag

    returns:
the constructed `Edge` object; this may be `None` if the input parameters indicate skipping the edge
        """
        if key is None:
            key = ".".join([
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
            nx_node["subobj"] = node.sub_obj
            nx_node["pos"] = node.pos
            nx_node["loc"] = str(node.loc)
            nx_node["length"] = node.length
            nx_node["hood"] = node.neighbors
            nx_node["anno"] = node.annotated

            # juggle the serialized IRIs
            if node.kind in [ NodeEnum.IRI ]:
                nx_node["iri"] = node.key
            elif node.label is not None and node.label.startswith("http"):
                nx_node["iri"] = node.label
            else:
                nx_node["iri"] = None

        # emulate a node-link format serialization, using the
        # default `NetworkX.node_link_data()` property names
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
            "links": edge_list,
            "graph": {}
        }

        return json.dumps(
            node_link,
            sort_keys = True,
            indent = 2,
            separators = ( ",", ":" ),
        )


    def load_lemma_graph (  # pylint: disable=R0914
        self,
        json_str: str,
        *,
        debug: bool = False,
        ) -> None:
        """
Load from a JSON string in
a JSON representation of the exported _lemma graph_ in
[_node-link_](https://networkx.org/documentation/stable/reference/readwrite/json_graph.html)
format

    debug:
debugging flag
        """
        dat: dict = json.loads(json_str)
        tokens: typing.List[ Node ] = []
        to_link: typing.Dict[ str, str ] = {}

        # deserialize the nodes
        for nx_node in dat.get("nodes"):  # type: ignore
            if debug:
                ic(nx_node)

            kind: NodeEnum = NodeEnum.decode(nx_node["kind"])  # type: ignore
            label: typing.Optional[ str ] = nx_node["label"]

            if kind in [ NodeEnum.ENT ] and nx_node["iri"] is not None:
                label = nx_node["iri"]

            node: Node = self.make_node(
                tokens,
                nx_node["lemma"],
                None,
                kind,
                0,
                0,
                0,
                label = label,
                length = nx_node["length"],
            )

            node.text = nx_node["name"]
            node.pos = nx_node["pos"]
            node.loc = eval(nx_node["loc"])  # pylint: disable=W0123
            node.count = int(nx_node["count"])
            node.neighbors = int(nx_node["hood"])
            node.annotated = nx_node["anno"]

            # note which `Node` objects need to have entities linked
            if kind == NodeEnum.ENT and nx_node["iri"] is not None:
                to_link[node.key] = nx_node["iri"]

            if debug:
                ic(node)

        # re-link the entities
        for src_key, cls_key in to_link.items():
            src_node: Node = self.nodes.get(src_key)  # type: ignore
            cls_node: Node = self.nodes.get(cls_key)  # type: ignore

            src_node.entity.append(
                LinkedEntity(
                    cls_node.span,
                    cls_node.label,  # type: ignore
                    cls_node.length,
                    cls_node.pos,
                    cls_node.weight,
                    0,
                    None,
                )
            )

        # deserialize the edges
        node_list: typing.List[ Node ] = list(self.nodes.values())

        for nx_edge in dat.get("links"):  # type: ignore
            if debug:
                ic(nx_edge)

            edge: Edge = self.make_edge(  # type: ignore
                node_list[nx_edge["source"]],
                node_list[nx_edge["target"]],
                RelEnum.decode(nx_edge["kind"]),  # type: ignore
                nx_edge["title"],
                float(nx_edge["prob"]),
                key = nx_edge["lemma"],
            )

            edge.count = int(nx_edge["count"])

            if debug:
                ic(edge)
