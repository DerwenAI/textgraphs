#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright ©2023 Derwen, Inc. All rights reserved.

"""
Reproduce examples from:

  * <https://derwen.ai/s/mqqm
  * <https://github.com/tomaarsen/SpanMarkerNER>
  * <https://github.com/thunlp/OpenNRE/>
  * <https://medium.com/@groxli/create-a-spacy-visualizer-with-streamlit-8b9b41b36745>
  * <https://youtu.be/C9p7suS-NGk?si=7Ohq3BV654ia2Im4>
"""

from collections import OrderedDict
import typing

from icecream import ic  # pylint: disable=E0401
import networkx as nx  # pylint: disable=E0401
import spacy  # pylint: disable=E0401

from textgraph import Node, Edge


class TextGraph:
    """
Construct a *lemma graph* from the unstructured text source, then
extract ranked phrases using a `textgraph` algorithm.
    """
    MAX_SKIP: int = 3
    NER_MODEL: str = "tomaarsen/span-marker-roberta-large-ontonotes5"
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
        self.lemma_graph: nx.DiGraph = nx.DiGraph()


    def build_doc (
        self,
        text_input: str,
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
        span: typing.Union[ spacy.tokens.span.Span, spacy.tokens.token.Token ],
        kind: typing.Optional[ str ] = None,
        ) -> Node:
        """
Lookup and return a `Node` object, instantiating a new one if it
does not exist already.
        """
        if key in self.nodes:
            self.nodes[key].count += 1
            return self.nodes[key]

        self.nodes[key] = Node(
            len(self.nodes),
            key,
            span,
            span.text,
            kind,
        )

        return self.nodes[key]


    def make_edge (
        self,
        src_node: Node,
        dst_node: Node,
        ) -> None:
        """
Lookup an edge, creating a new one if it does not exist already,
and increment the count if it does.
        """
        key: str = ".".join([ str(src_node.node_id), str(dst_node.node_id) ])

        if key in self.edges:
            self.edges[key].count += 1
        else:
            self.edges[key] = Edge(
                src_node.node_id,
                dst_node.node_id,
            )


    def tokenize_phrases (
        self,
        sent,
        *,
        debug: bool = False,
        ) -> typing.Iterator[ Node ]:
        """
Collect the phrases from the parsed document, giving priority to:

  * NER entities+labels
  * noun chunks
  * lemmas
        """
        # extract entities using NER
        ent_seq: typing.List[ spacy.tokens.span.Span ] = list(sent.ents)

        ent_idx: typing.List[ int ] = [
            ent.start for ent in ent_seq
        ]

        # extract noun phrases
        nph_seq: typing.List[ spacy.tokens.span.Span ] = [
            nph for nph in sent.noun_chunks
            if nph.start not in ent_idx
        ]

        if debug:
            ic(ent_seq)
            ic(ent_idx)
            ic(nph_seq)

        i: int = sent.start

        while i < sent.end:
            tok: spacy.tokens.token.Token = sent[i]

            if debug:
                ic(i, tok)

            if len(ent_seq) > 0 and ent_seq[0].start == tok.i:
                ent = ent_seq.pop(0)
                yield self.make_node(ent.text.lower(), ent, kind = ent.label_)
                i = ent.end

            elif len(nph_seq) > 0 and nph_seq[0].start == tok.i:
                nph = nph_seq.pop(0)
                yield self.make_node(nph.text.lower(), nph)
                i = nph.end

            else:
                # fall-through: build lemmatized entities
                if tok.pos_ in [ "ADJ", "NOUN", "PROPN" ]:
                    lemma_key: str = ".".join([ tok.lemma_.strip().lower(), tok.pos_ ])
                    yield self.make_node(lemma_key, tok)

                i += 1


    def build_graph_embeddings (
        self,
        doc: spacy.tokens.doc.Doc,
        *,
        debug: bool = True,
        ) -> None:
        """
Construct a *lemma graph* for use with the `textgraph` algorithm,
which in other words represent "reverse embeddings" from the parsed
document.
        """
        for sent in doc.sents:
            if debug:
                ic(sent)

            sent_nodes: typing.List[ Node ] = list(self.tokenize_phrases(sent))

            for i, src_node in enumerate(sent_nodes):
                for j in range(i + 1, len(sent_nodes)):
                    if (j - i) <= self.MAX_SKIP:
                        self.make_edge(src_node, sent_nodes[j])


    def calc_phrase_ranks (
        self,
        ) -> None:
        """
Build a directed graph in `NetworkX` from the *lemma graph* results of
the `textgraph` algorithm, then calculate the weights for each node as
ranked phrases.

Note that the phrase ranks should be normalized to sum to 1.0
        """
        self.lemma_graph.add_nodes_from([
            node.node_id
            for node in self.nodes.values()
        ])

        self.lemma_graph.add_edges_from([
            ( edge.src_node, edge.dst_node, { "weight": float(edge.count) }, )
            for edge in self.edges.values()
        ])

        n_list: list = list(self.nodes.values())

        for node_id, rank in nx.pagerank(self.lemma_graph, alpha = self.PR_ALPHA).items():
            n_list[node_id].weight = rank


if __name__ == "__main__":
    SRC_TEXT: str = """
Werner Herzog is a remarkable filmmaker and intellectual originally from Germany, the son of Dietrich Herzog.
    """
    tg: TextGraph = TextGraph()

    sample_doc: spacy.tokens.doc.Doc = tg.build_doc(
        SRC_TEXT.strip(),
        use_llm = False,
    )

    tg.build_graph_embeddings(sample_doc)
    tg.calc_phrase_ranks()

    ic(tg.edges)
    ic(tg.nodes)
