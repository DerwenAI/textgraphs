#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright ©2023 Derwen, Inc. All rights reserved.

"""
NLP pipeline factory builder pattern.
"""

import functools
import operator
import typing

from icecream import ic  # pylint: disable=E0401,W0611
import spacy  # pylint: disable=E0401


class Pipeline:  # pylint: disable=R0903
    """
 Manage parsing the two documents, which are assumed to be paragraph-sized.
    """

    def __init__ (
        self,
        text_input: str,
        tok_pipe: spacy.Language,
        ent_pipe: spacy.Language,
        ) -> None:
        """
Constructor.
        """
        self.text: str = text_input
        self.tok_doc: spacy.tokens.Doc = tok_pipe(self.text)
        self.ent_doc: spacy.tokens.Doc = ent_pipe(self.text)


    @classmethod
    def get_lemma_key (
        cls,
        token: spacy.tokens.token.Token,
        *,
        placeholder: bool = False,
        ) -> str:
        """
Compose a unique, invariant lemma key for the given span.
        """
        terms: typing.List[ str ] = [
            token.lemma_.strip().lower(),
            token.pos_,
        ]

        if placeholder:
            terms.insert(0, str(token.i))

        return ".".join(terms)


    def get_ent_lemma_keys (
        self,
        ) -> typing.Iterator[ str ]:
        """
Iterate through the fully qualified lemma keys for an extracted entity.
        """
        for ent in self.tok_doc.ents:
            yield ".".join(
                functools.reduce(
                    operator.iconcat,
                    [
                        [ tok.lemma_.strip().lower(), tok.pos_, ]
                        for tok in ent
                    ],
                    [],
                )
            )


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
  * `ent_pipe` -- with entities merged
        """
        # determine the NER model to be used
        exclude: typing.List[ str ] = []

        if ner_model is not None:
            exclude.append("ner")

        # build both pipelines
        self.tok_pipe = spacy.load(
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

            self.ent_pipe.add_pipe(
                "span_marker",
                config = {
                    "model": ner_model,
                },
            )

        # merge entities on one pipe only
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
            self.ent_pipe,
        )
