#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright ©2023 Derwen, Inc. All rights reserved.

"""
NLP pipeline factory builder pattern.
"""

from dataclasses import dataclass
import functools
import operator
import typing

from icecream import ic  # pylint: disable=E0401,W0611
import spacy  # pylint: disable=E0401


@dataclass(order=False, frozen=False)
class NounChunk:  # pylint: disable=R0902
    """
A data class representing one noun chunk, i.e., a candidate as an extracted phrase.
    """
    span: spacy.tokens.span.Span
    text: str
    length: int
    lemma_key: str
    unseen: bool
    sent_id: int
    start: int = 0


class Pipeline:  # pylint: disable=R0903
    """
 Manage parsing the two documents, which are assumed to be paragraph-sized.
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
        ) -> typing.Iterator[ str ]:
        """
Iterate through the fully qualified lemma keys for an extracted entity.
        """
        for ent in self.tok_doc.ents:
            yield self.get_lemma_key(ent)


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


class PipelineFactory:  # pylint: disable=R0903
    """
Factory pattern for building a pipeline, which is one of the more
expensive operations with `spaCy`
    """
    NER_MODEL: str = "tomaarsen/span-marker-roberta-large-ontonotes5"
    SPACY_MODEL: str = "en_core_web_sm"


    def __init__ (
        self,
        *,
        spacy_model: str = SPACY_MODEL,
        ner_model: typing.Optional[ str ] = NER_MODEL,
        ) -> None:
        """
Constructor which instantiates two `spaCy` pipeline:

  * `tok_pipe` -- regular generator for parsed tokens
  * `dbp_pipe` -- DBPedia entity linking
  * `ent_pipe` -- with entities merged
        """
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
        self.dbp_pipe.add_pipe("dbpedia_spotlight")

        # `ent_pipe` only: merge entities
        self.ent_pipe.add_pipe("merge_entities")


    def build_pipeline (
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
