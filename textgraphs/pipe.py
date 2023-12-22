#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Leveraging a factory pattern for NLP pipelines.

This class handles processing for one "chunk" of raw text input to
analyze, which is typically a paragraph. In other words, objects in
this class are expected to get recycled when processing moves on to
the next paragraph, to ease memory requirements.

see copyright/license https://huggingface.co/spaces/DerwenAI/textgraphs/blob/main/README.md
"""

import abc
import asyncio
import functools
import itertools
import operator
import traceback
import typing

from icecream import ic  # pylint: disable=E0401,W0611
import networkx as nx  # pylint: disable=E0401
import spacy  # pylint: disable=E0401

from .defaults import NER_MODEL, SPACY_MODEL
from .elem import Node, NodeEnum, NounChunk
from .graph import SimpleGraph


######################################################################
## class definitions

class KnowledgeGraph:
    """
Abstract base class for a _knowledge graph_ interface.
    """

    def remap_ner (
        self,
        label: typing.Optional[ str ],
        ) -> typing.Optional[ str ]:
        """
Remap the OntoTypes4 values from NER output to more general-purpose IRIs.
        """
        return label


    def normalize_prefix (
        self,
        iri: str,
        *,
        debug: bool = False,  # pylint: disable=W0613
        ) -> str:
        """
Normalize the given IRI to use standard namespace prefixes.
        """
        return iri


    def perform_entity_linking (
        self,
        graph: SimpleGraph,
        pipe: "Pipeline",
        *,
        debug: bool = False,
        ) -> None:
        """
Perform _entity linking_ based on "spotlight" and other services.
        """
        pass  # pylint: disable=W0107


    def resolve_rel_iri (
        self,
        rel: str,
        *,
        lang: str = "en",  # pylint: disable=W0613
        debug: bool = False,  # pylint: disable=W0613
        ) -> typing.Optional[ str ]:
        """
Resolve a `rel` string from a _relation extraction_ model which has
been trained on this knowledge graph.
        """
        return rel


class InferRel (abc.ABC):  # pylint: disable=R0903
    """
Abstract base class for a _relation extraction_ model wrapper.
    """

    @abc.abstractmethod
    def gen_triples (
        self,
        pipe: "Pipeline",
        *,
        debug: bool = False,
        ) -> typing.Iterator[typing.Tuple[ Node, str, Node ]]:
        """
Infer relations as triples through a generator _iteratively_.
        """
        raise NotImplementedError


    async def gen_triples_async (
        self,
        pipe: "Pipeline",
        queue: asyncio.Queue,
        *,
        debug: bool = False,
        ) -> None:
        """
Infer relations as triples produced to a queue _concurrently_.
        """
        for src, iri, dst in self.gen_triples(pipe, debug = debug):
            await queue.put(( src, iri, dst, ))


class Pipeline:  # pylint: disable=R0902,R0903
    """
Manage parsing of a document, which is assumed to be paragraph-sized.
    """

    def __init__ (  # pylint: disable=R0913
        self,
        text_input: str,
        lemma_graph: nx.MultiDiGraph,
        tok_pipe: spacy.Language,
        spl_pipe: spacy.Language,
        ner_pipe: spacy.Language,
        kg: KnowledgeGraph,  # pylint: disable=C0103
        infer_rels: typing.List[ InferRel ],
        ) -> None:
        """
Constructor.
        """
        self.text: str = text_input
        self.lemma_graph: nx.MultiDiGraph = lemma_graph

        # `tok_doc` provides a stream of individual tokens
        self.tok_doc: spacy.tokens.Doc = tok_pipe(self.text)

        # `spl_doc` provides span indexing for Spotlight entity linking
        self.spl_doc: spacy.tokens.Doc = spl_pipe(self.text)

        # `ner_doc` provides the merged-entity spans from NER
        self.ner_doc: spacy.tokens.Doc = ner_pipe(self.text)

        self.kg: KnowledgeGraph = kg  # pylint: disable=C0103
        self.infer_rels: typing.List[ InferRel ] = infer_rels

        # list of Node objects for each parsed token, in sequence
        self.tokens: typing.List[ Node ] = []


    @classmethod
    def get_lemma_key (
        cls,
        span: typing.Union[ spacy.tokens.span.Span, spacy.tokens.token.Token ],
        *,
        placeholder: bool = False,
        ) -> str:
        """
Compose a unique, invariant lemma key for the given span.
        """
        if isinstance(span, spacy.tokens.token.Token):
            terms: typing.List[ str ] = [
                span.lemma_.strip().lower(),
                span.pos_,
            ]

            if placeholder:
                terms.insert(0, str(span.i))

        else:
            terms = functools.reduce(
                operator.iconcat,
                [
                    [ token.lemma_.strip().lower(), token.pos_, ]
                    for token in span
                ],
                [],
            )

        return ".".join(terms)


    def get_ent_lemma_keys (
        self,
        ) -> typing.Iterator[ typing.Tuple[ str, int ]]:
        """
Iterate through the fully qualified lemma keys for an extracted entity.
        """
        for ent in self.tok_doc.ents:
            yield self.get_lemma_key(ent), len(ent)


    def link_noun_chunks (
        self,
        nodes: dict,
        *,
        debug: bool = False,
        ) -> typing.List[ NounChunk ]:
        """
Link any noun chunks which are not already subsumed by named entities.
        """
        chunks: typing.List[ NounChunk ] = []

        # first pass: note the available noun chunks
        for sent_id, sent in enumerate(self.tok_doc.sents):
            for span in sent.noun_chunks:
                lemma_key: str = self.get_lemma_key(span)

                chunks.append(
                    NounChunk(
                        span,
                        span.text,
                        len(span),
                        lemma_key,
                        lemma_key not in nodes,
                        sent_id,
                    )
                )

        # second pass: remap span indices to the merged entities pipeline
        for i, span in enumerate(self.ner_doc.noun_chunks):
            if span.text == self.tokens[span.start].text:
                chunks[i].unseen = False
            elif chunks[i].unseen:
                chunks[i].start = span.start

                if debug:
                    ic(chunks[i])

        return chunks


    ######################################################################
    ## relation extraction

    def iter_entity_pairs (
        self,
        max_skip: int,
        *,
        debug: bool = True,
        ) -> typing.Iterator[ typing.Tuple[ Node, Node ]]:
        """
Iterator for entity pairs for which the algorithm infers relations.
        """
        ent_list: typing.List[ Node ] = [
            node
            for node in self.tokens
            if node.kind in [ NodeEnum.ENT ]
        ]

        lemma_graph_view: nx.MultiGraph = self.lemma_graph.to_undirected(
            as_view = True,
        )

        for pair in itertools.product(ent_list, repeat = 2):
            if pair[0] != pair[1]:
                src: Node = pair[0]
                dst: Node = pair[1]

                try:
                    path: typing.List[ int ] = nx.shortest_path(
                        lemma_graph_view,
                        source = src.node_id,
                        target = dst.node_id,
                        weight = "weight",
                        method = "dijkstra",
                    )

                    if debug:
                        ic(src.node_id, dst.node_id, path)

                    if len(path) <= max_skip:
                        yield ( src, dst, )
                except nx.NetworkXNoPath:
                    pass
                except Exception as ex:  # pylint: disable=W0718
                    ic(ex)
                    ic("ERROR", src, dst)
                    traceback.print_exc()


class PipelineFactory:  # pylint: disable=R0903
    """
Factory pattern for building a pipeline, which is one of the more
expensive operations with `spaCy`
    """

    def __init__ (  # pylint: disable=W0102
        self,
        *,
        spacy_model: str = SPACY_MODEL,
        ner_model: typing.Optional[ str ] = NER_MODEL,
        kg: KnowledgeGraph = KnowledgeGraph(),  # pylint: disable=C0103
        infer_rels: typing.List[ InferRel ] = []
        ) -> None:
        """
Constructor which instantiates the `spaCy` pipelines:

  * `tok_pipe` -- regular generator for parsed tokens
  * `spl_pipe` -- spotlight entity linking
  * `ner_pipe` -- with entities merged
        """
        self.kg: KnowledgeGraph = kg  # pylint: disable=C0103
        self.infer_rels: typing.List[ InferRel ] = infer_rels

        # determine the NER model to be used
        exclude: typing.List[ str ] = []

        if ner_model is not None:
            exclude.append("ner")

        # build the pipelines
        # NB: `spaCy` team doesn't quite get the PEP 621 restrictions which PyPa mangled:
        # https://github.com/explosion/spaCy/issues/3536
        # https://github.com/explosion/spaCy/issues/4592#issuecomment-704373657
        if not spacy.util.is_package(spacy_model):
            spacy.cli.download(spacy_model)

        self.tok_pipe = spacy.load(
            spacy_model,
            exclude = exclude,
        )

        self.spl_pipe = spacy.load(
            spacy_model,
            exclude = exclude,
        )

        self.ner_pipe = spacy.load(
            spacy_model,
            exclude = exclude,
        )

        # add NER
        if ner_model is not None:
            # REFACTOR
            self.tok_pipe.add_pipe(
                "span_marker",
                config = {
                    "model": ner_model,
                },
            )

            # REFACTOR
            self.spl_pipe.add_pipe(
                "span_marker",
                config = {
                    "model": ner_model,
                },
            )

            # REFACTOR
            self.ner_pipe.add_pipe(
                "span_marker",
                config = {
                    "model": ner_model,
                },
            )

        # REFACTOR
        # `spl_pipe` only: KG entity linking
        self.spl_pipe.add_pipe(
            "dbpedia_spotlight",
            config = {
                "dbpedia_rest_endpoint": kg.spotlight_api,  # type: ignore
            },
        )

        # `ner_pipe` only: merge entities
        self.ner_pipe.add_pipe(
            "merge_entities",
        )


    def create_pipeline (
        self,
        text_input: str,
        lemma_graph: nx.MultiDiGraph,
        ) -> Pipeline:
        """
Instantiate the document pipelines needed to parse the input text.
        """
        pipe: Pipeline = Pipeline(
            text_input,
            lemma_graph,
            self.tok_pipe,
            self.spl_pipe,
            self.ner_pipe,
            self.kg,
            self.infer_rels,
        )

        return pipe
