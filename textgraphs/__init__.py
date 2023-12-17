#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Package definitions for the `TextGraphs` library.

see copyright/license https://huggingface.co/spaces/DerwenAI/textgraphs/blob/main/README.md
"""

from .defaults import DBPEDIA_MIN_ALIAS, DBPEDIA_MIN_SIM, \
    DBPEDIA_SEARCH_API, DBPEDIA_SPARQL_API, DBPEDIA_SPOTLIGHT_API, \
    FISHING_API, MAX_SKIP, MREBEL_MODEL, \
    NER_MAP, NER_MODEL, OPENNRE_MIN_PROB, OPENNRE_MODEL, \
    PAGERANK_ALPHA, SPACY_MODEL, WIKIDATA_API

from .doc import TextGraphs

from .elem import Edge, LinkedEntity, Node, NodeEnum, NounChunk, RelEnum, WikiEntity

from .kg import WikiDatum

from .pipe import InferRel, Pipeline, PipelineFactory

from .rel import InferRel_OpenNRE, InferRel_Rebel

from .util import calc_quantile_bins, root_mean_square, stripe_column

from .vis import RenderPyVis


__title__ = "TextGraphs: raw texts, LLMs, and KGs, oh my!"

__description__ = "TextGraphs + LLM + graph ML for entity extraction, linking, ranking, and constructing a lemma graph"  # pylint: disable=C0301

__copyright__ = "2023, Derwen, Inc."

__author__ = """\n""".join([
    "derwen.ai <info@derwen.ai>"
])

__version__ = "0.1.2"
__release__ = "0.1.2"