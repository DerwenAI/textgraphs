#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Default settings for the `TextGraphs` library.

see copyright/license https://huggingface.co/spaces/DerwenAI/textgraphs/blob/main/README.md
"""

import spacy_dbpedia_spotlight  # pylint: disable=E0401


DBPEDIA_MIN_ALIAS: float = 0.8
DBPEDIA_MIN_SIM: float = 0.9

DBPEDIA_SEARCH_API: str = "https://lookup.dbpedia.org/api/search"
DBPEDIA_SPARQL_API: str = "https://dbpedia.org/sparql"
DBPEDIA_SPOTLIGHT_API: str = f"{spacy_dbpedia_spotlight.EntityLinker.base_url}/en"

FISHING_API: str = "https://cloud.science-miner.com/nerd/service"

MAX_SKIP: int = 11

MREBEL_MODEL: str = "Babelscape/mrebel-large"

NER_MODEL: str = "tomaarsen/span-marker-roberta-large-ontonotes5"

OPENNRE_MIN_PROB: float = 0.9
OPENNRE_MODEL: str = "wiki80_cnn_softmax"

PAGERANK_ALPHA: float = 0.85

SPACY_MODEL: str = "en_core_web_sm"

WIKIDATA_API: str = "https://www.wikidata.org/w/api.php"
