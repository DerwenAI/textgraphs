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

import functools
import operator
import typing

from icecream import ic  # pylint: disable=E0401,W0611
import opennre  # pylint: disable=E0401
import spacy  # pylint: disable=E0401

from .defaults import DBPEDIA_SPOTLIGHT_API, NER_MODEL, NRE_MODEL, SPACY_MODEL
from .elem import LinkedEntity, NodeEnum, NounChunk, WikiEntity
from .wiki import WikiDatum


class Pipeline:  # pylint: disable=R0903
    """
Manage parsing of a document, which is assumed to be paragraph-sized.
    """

    def __init__ (
        self,
        text_input: str,
        tok_pipe: spacy.Language,
        dbp_pipe: spacy.Language,
        ent_pipe: spacy.Language,
        ) -> None:
        """
Constructor.
        """
        self.text: str = text_input
        self.tok_doc: spacy.tokens.Doc = tok_pipe(self.text)
        self.dbp_doc: spacy.tokens.Doc = dbp_pipe(self.text)
        self.ent_doc: spacy.tokens.Doc = ent_pipe(self.text)
        self.wiki: WikiDatum = WikiDatum()


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
        tokens: list,
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
        for i, span in enumerate(self.ent_doc.noun_chunks):
            if span.text == tokens[span.start].text:
                chunks[i].unseen = False
            elif chunks[i].unseen:
                chunks[i].start = span.start

                if debug:
                    ic(chunks[i])

        return chunks


    def link_dbpedia_spotlight_entities (  # pylint: disable=R0914
        self,
        tokens: list,
        dbpedia_search_api: str,
        min_alias: float,
        min_similarity: float,
        *,
        debug: bool = False,
        ) -> typing.Iterator[ LinkedEntity ]:
        """
Iterator for the results of using DBPedia Spotlight for entity linking.
        """
        ents: typing.List[ spacy.tokens.span.Span ] = list(self.dbp_doc.ents)

        if debug:
            ic(ents)

        ent_idx: int = 0
        tok_idx: int = 0

        for i, tok in enumerate(tokens):  # pylint: disable=R1702
            if debug:
                print()
                ic(tok_idx, tok.text, tok.pos)
                ic(ent_idx, len(ents))

            if ent_idx < len(ents):
                ent = ents[ent_idx]

                if debug:
                    ic(ent.start, tok_idx)

                if ent.start == tok_idx:
                    if debug:
                        ic(ent.text, ent.start, len(ent))
                        ic(ent.kb_id_, ent._.dbpedia_raw_result["@similarityScore"])
                        ic(ent._.dbpedia_raw_result)

                    prob: float = float(ent._.dbpedia_raw_result["@similarityScore"])
                    count: int = int(ent._.dbpedia_raw_result["@support"])

                    if tok.pos == "PROPN" and prob >= min_similarity:
                        wiki_ent: typing.Optional[ WikiEntity ] = self.wiki.dbpedia_search_entity(
                            ent.text,
                            dbpedia_search_api,
                            debug = debug,
                        )

                        if debug:
                            ic(wiki_ent)

                        if wiki_ent is not None and wiki_ent.prob > min_alias:  # type: ignore
                            iri: str = ent.kb_id_

                            dbp_link: LinkedEntity = LinkedEntity(
                                ent,
                                iri,
                                len(ent),
                                "dbpedia",
                                prob,
                                i,
                                wiki_ent,  # type: ignore
                                count = count,
                            )

                            if debug:
                                ic("found", dbp_link)

                            yield dbp_link

                    ent_idx += 1

            tok_idx += tok.length


    def link_dbpedia_search_entities (
        self,
        nodes: list,
        dbpedia_search_api: str,
        min_alias: float,
        *,
        debug: bool = False,
        ) -> typing.Iterator[ LinkedEntity ]:
        """
Iterator for the results of using DBPedia Search directly for entity linking.
        """
        for i, node in enumerate(nodes):
            if node.kind in [ NodeEnum.ENT ] and len(node.entity) < 1:
                wiki_ent: typing.Optional[ WikiEntity ] = self.wiki.dbpedia_search_entity(
                    node.text,
                    dbpedia_search_api,
                    debug = debug,
                )

                if wiki_ent.prob > min_alias:  # type: ignore
                    dbp_link: LinkedEntity = LinkedEntity(
                        node.span,
                        wiki_ent.iri,  # type: ignore
                        node.length,
                        "dbpedia",
                        wiki_ent.prob,  # type: ignore
                        i,
                        wiki_ent,  # type: ignore
                    )

                    if debug:
                        ic("found", dbp_link)

                    yield dbp_link


class PipelineFactory:  # pylint: disable=R0903
    """
Factory pattern for building a pipeline, which is one of the more
expensive operations with `spaCy`
    """

    def __init__ (
        self,
        *,
        spacy_model: str = SPACY_MODEL,
        ner_model: typing.Optional[ str ] = NER_MODEL,
        nre_model: typing.Optional[ str ] = NRE_MODEL,
        dbpedia_spotlight_api: str = DBPEDIA_SPOTLIGHT_API,
        ) -> None:
        """
Constructor which instantiates the `spaCy` pipelines:

  * `tok_pipe` -- regular generator for parsed tokens
  * `dbp_pipe` -- DBPedia entity linking
  * `ent_pipe` -- with entities merged
        """
        # add NRE model, if used
        self.nre: typing.Optional[ opennre.model.softmax_nn.SoftmaxNN ] = None

        if nre_model is not None:
            self.nre = opennre.get_model(NRE_MODEL)

        # determine the NER model to be used
        exclude: typing.List[ str ] = []

        if ner_model is not None:
            exclude.append("ner")

        # build the pipelines
        self.tok_pipe = spacy.load(
            spacy_model,
            exclude = exclude,
        )

        self.dbp_pipe = spacy.load(
            spacy_model,
            exclude = exclude,
        )

        self.ent_pipe = spacy.load(
            spacy_model,
            exclude = exclude,
        )

        # add NER
        if ner_model is not None:
            self.tok_pipe.add_pipe(
                "span_marker",
                config = {
                    "model": ner_model,
                },
            )

            self.dbp_pipe.add_pipe(
                "span_marker",
                config = {
                    "model": ner_model,
                },
            )

            self.ent_pipe.add_pipe(
                "span_marker",
                config = {
                    "model": ner_model,
                },
            )

        # `dbp_pipe` only: DBPedia entity linking
        self.dbp_pipe.add_pipe(
            "dbpedia_spotlight",
            config = {
                "dbpedia_rest_endpoint": dbpedia_spotlight_api,
            },
        )

        # `ent_pipe` only: merge entities
        self.ent_pipe.add_pipe(
            "merge_entities",
        )


    def create_pipeline (
        self,
        text_input: str,
        ) -> Pipeline:
        """
return document pipelines to parse the given text input.
        """
        return Pipeline(
            text_input,
            self.tok_pipe,
            self.dbp_pipe,
            self.ent_pipe,
        )
