#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Classes for encapsulating NER models.

see copyright/license https://huggingface.co/spaces/DerwenAI/textgraphs/blob/main/README.md
"""

from .defaults import NER_MODEL
from .pipe import Component, PipelineFactory


######################################################################
## class definitions

class NERSpanMarker (Component):  # pylint: disable=R0903
    """
Configures a `spaCy` pipeline component for `SpanMarkerNER`
    """

    def __init__ (
        self,
        *,
        ner_model: str = NER_MODEL,
        ) -> None:
        """
Constructor.
        """
        self.ner_model: str = ner_model


    def augment_pipe (
        self,
        factory: PipelineFactory,
        ) -> None:
        """
Encapsulate a `spaCy` call to `add_pipe()` configuration.
        """
        factory.tok_pipe.add_pipe(
            "span_marker",
            config = {
                "model": self.ner_model,
            },
        )

        factory.ner_pipe.add_pipe(
            "span_marker",
            config = {
                "model": self.ner_model,
            },
        )

        factory.aux_pipe.add_pipe(
            "span_marker",
            config = {
                "model": self.ner_model,
            },
        )
