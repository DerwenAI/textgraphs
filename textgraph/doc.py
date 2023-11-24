#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright ©2023 Derwen, Inc. All rights reserved.

"""
Implementation of an LLM-augmented `textgraph` algorithm for
constructing a _knowledge graph_ from raw, unstructured text source.

This integrate examples from:

  * <https://derwen.ai/s/mqqm>
  * <https://github.com/tomaarsen/SpanMarkerNER>
  * <https://github.com/thunlp/OpenNRE/>
  * <https://medium.com/@groxli/create-a-spacy-visualizer-with-streamlit-8b9b41b36745>
  * <https://youtu.be/C9p7suS-NGk?si=7Ohq3BV654ia2Im4>
  * <https://towardsdatascience.com/how-to-convert-any-text-into-a-graph-of-concepts-110844f22a1a>
"""

from collections import OrderedDict
import itertools
import typing

from icecream import ic  # pylint: disable=E0401
import networkx as nx  # pylint: disable=E0401
import opennre  # pylint: disable=E0401
import spacy  # pylint: disable=E0401

from .graph import Node, Edge, RelEnum


class TextGraph:
    """
Construct a _lemma graph_ from the unstructured text source, then
extract ranked phrases using a `textgraph` algorithm.
    """
    MAX_SKIP: int = 11
    NER_MODEL: str = "tomaarsen/span-marker-roberta-large-ontonotes5"
    NRE_MODEL: str = "wiki80_cnn_softmax"
    PR_ALPHA: float = 0.6
    SPACY_MODEL: str = "en_core_web_sm"


    def __init__ (
        self,
        ) -> None:
        """
Constructor.
        """
        self.nodes: typing.Dict[ str, Node ] = OrderedDict()
        self.edges: typing.Dict[ str, Edge ] = {}
        self.lemma_graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self.nre: opennre.model.softmax_nn.SoftmaxNN = opennre.get_model(self.NRE_MODEL)


    def build_doc (
        self,
        text_input: str,
        *,
        use_llm: bool = False,
        ) -> spacy.tokens.doc.Doc:
        """
Instantiate a `spaCy` pipeline and return a document which parses
the given text input.
        """
        exclude: typing.List[ str ] = []

        if use_llm:
            exclude.append("ner")

        nlp = spacy.load(
            self.SPACY_MODEL,
            exclude = exclude,
        )

        nlp.add_pipe("merge_entities")

        if use_llm:
            nlp.add_pipe(
                "span_marker",
                config = {
                    "model": self.NER_MODEL,
                },
            )

        return nlp(text_input)


    def make_node (
        self,
        key: str,
        span: spacy.tokens.token.Token,
        *,
        kind: typing.Optional[ str ] = None,
        condense: bool = True,
        ) -> Node:
        """
Lookup and return a `Node` object:

    * default: condense matching keys into the same node
    * instantiate a new node if it does not exist already
        """
        if not condense:
            # placeholder node (stopwords)
            self.nodes[key] = Node(
                len(self.nodes),
                span,
                span.text,
                span.pos_,
            )

        elif key in self.nodes:
            # link to previously constructed entity node
            self.nodes[key].count += 1

        # construct a new entity node
        else:
            self.nodes[key] = Node(
                len(self.nodes),
                span,
                span.text,
                span.pos_,
                kind = kind,
                count = 1,
            )

        return self.nodes.get(key)  # type: ignore


    def extract_phrases (
        self,
        sent,
        *,
        debug: bool = False,
        ) -> typing.Iterator[ Node ]:
        """
Extract phrases from the parsed document to build nodes in the
_lemma graph_, while giving priority to:

  * NER entities+labels
  * lemmatized nouns and verbs
        """
        # extract entities using NER
        ent_seq: typing.List[ spacy.tokens.span.Span ] = list(sent.ents)

        if debug:
            ic(ent_seq)

        for token in sent:
            head = ( token.head, token.head.i, )

            if debug:
                ic(
                    token,
                    token.i,
                    token.dep_,
                    head,
                )

            if len(ent_seq) > 0 and ent_seq[0].start == token.i:
                # link a named entity
                ent = ent_seq.pop(0)
                lemma_key: str = ".".join([ token.lemma_.strip().lower(), token.pos_ ])
                yield self.make_node(lemma_key, token, kind = ent.label_)

            elif token.pos_ in [ "NOUN", "PROPN", "VERB" ]:
                # link a lemmatized entity
                lemma_key = ".".join([ token.lemma_.strip().lower(), token.pos_ ])
                yield self.make_node(lemma_key, token)

            else:
                # fall-through case: use token as a placeholder in the lemma graph
                lemma_key = ".".join([ str(token.i), token.lower_, token.pos_ ])
                yield self.make_node(lemma_key, token, condense = False)


    def make_edge (  # pylint: disable=R0913
        self,
        src_node: Node,
        dst_node: Node,
        kind: RelEnum,
        rel: str,
        prob: float,
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

        if key in self.edges:
            self.edges[key].count += 1

        elif src_node.node_id != dst_node.node_id:
            # no loopback
            self.edges[key] = Edge(
                src_node.node_id,
                dst_node.node_id,
                kind,
                rel,
                prob,
            )

        return self.edges.get(key)


    def build_graph_embeddings (
        self,
        doc: spacy.tokens.doc.Doc,
        *,
        debug: bool = True,
        ) -> None:
        """
Construct a _lemma graph_ from the results of running the `textgraph`
algorithm, as a directed graph in `NetworkX`.

In other words, this represents "reverse embeddings" from the parsed
document.
        """
        for sent in doc.sents:
            if debug:
                ic(sent)

            sent_nodes: typing.List[ Node ] = list(self.extract_phrases(sent))

            for node in sent_nodes:
                self.make_edge(
                    node,
                    sent_nodes[node.span.head.i],
                    RelEnum.DEP,
                    node.span.dep_,
                    1.0,
                )

        self.lemma_graph.add_nodes_from([
            node.node_id
            for node in self.nodes.values()
        ])

        # add edges from the parse dependencies
        self.lemma_graph.add_edges_from([
            (
                edge.src_node,
                edge.dst_node,
                {
                    "weight": float(edge.count),
                },
            )
            for edge in self.edges.values()
        ])


    def infer_relations (
        self,
        text_input: str,
        *,
        debug: bool = True,
        ) -> None:
        """
Run NRE to infer relations between pairs of co-occurring entities.
        """
        inferred_edges: typing.List[ Edge ] = []

        ent_list: typing.List[ Node ] = [
            node
            for node in self.nodes.values()
            if node.kind is not None
        ]

        for pair in itertools.product(ent_list, repeat = 2):
            if pair[0] != pair[1]:
                src: Node = pair[0]
                dst: Node = pair[1]

                try:
                    path: typing.List[ int ] = nx.shortest_path(
                        self.lemma_graph.to_undirected(as_view = True),
                        source = src.node_id,
                        target = dst.node_id,
                        weight = "weight",
                        method = "dijkstra",
                    )

                    if debug:
                        ic(src.node_id, dst.node_id, path)

                    if len(path) <= self.MAX_SKIP:
                        rel, prob = self.nre.infer({
                            "text": text_input,
                            "h": { "pos": src.get_pos() },
                            "t": { "pos": dst.get_pos() },
                        })

                        if debug:
                            ic(rel, prob)

                        edge: Edge = self.make_edge(  # type: ignore
                            src,
                            dst,
                            RelEnum.INFER,
                            rel,
                            prob,
                        )

                        inferred_edges.append(edge)
                except Exception as ex:  # pylint: disable=W0718
                    ic(ex)
                    ic(src, dst)

        # add edges from the inferred relations
        self.lemma_graph.add_edges_from([
            (
                edge.src_node,
                edge.dst_node,
                {
                    "weight": edge.prob,
                },
            )
            for edge in inferred_edges
        ])


    def calc_phrase_ranks (
        self,
        ) -> None:
        """
Calculate the weights for each node in the _lemma graph_, which now
represent ranked phrases.

Note that the phrase ranks should be normalized to sum to 1.0
        """
        n_list: typing.List[ Node ] = list(self.nodes.values())

        for node_id, rank in nx.pagerank(self.lemma_graph, alpha = self.PR_ALPHA).items():
            node: Node = n_list[node_id]

            if node.count > 0:
                node.weight = rank
