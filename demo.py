#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sample application to demo the `TextGraphs` library.

see copyright/license https://huggingface.co/spaces/DerwenAI/textgraphs/blob/main/README.md
"""

import sys  # pylint: disable=W0611
import time

from icecream import ic  # pylint: disable=E0401
from pyinstrument import Profiler  # pylint: disable=E0401
import pandas as pd  # pylint: disable=E0401

import textgraphs


if __name__ == "__main__":
    SRC_TEXT: str = """
Werner Herzog is a remarkable filmmaker and an intellectual originally from Germany, the son of Dietrich Herzog.
After the war, Werner fled to America to become famous.
"""

    ## set up
    profiler: Profiler = Profiler()
    profiler.start()

    start_time: float = time.time()

    tg: textgraphs.TextGraphs = textgraphs.TextGraphs(
        factory = textgraphs.PipelineFactory(
            spacy_model = textgraphs.SPACY_MODEL,
            ner_model = None,
            nre_model = textgraphs.NRE_MODEL,
            dbpedia_spotlight_api = textgraphs.DBPEDIA_SPOTLIGHT_API,
        ),
    )

    duration: float = round(time.time() - start_time, 3)
    print(f"{duration:7.3f} sec: set up")


    ## NLP parse
    start_time = time.time()

    pipe: textgraphs.Pipeline = tg.create_pipeline(
        SRC_TEXT.strip(),
    )

    duration = round(time.time() - start_time, 3)
    print(f"{duration:7.3f} sec: parse text")


    ## collect graph elements from the parse
    start_time = time.time()

    tg.collect_graph_elements(
        pipe,
        debug = False,
    )

    duration = round(time.time() - start_time, 3)
    print(f"{duration:7.3f} sec: collect elements")


    ## perform entity linking
    start_time = time.time()

    tg.perform_entity_linking(
        pipe,
        dbpedia_search_api = textgraphs.DBPEDIA_SEARCH_API,
        min_alias = textgraphs.DBPEDIA_MIN_ALIAS,
        min_similarity = textgraphs.DBPEDIA_MIN_SIM,
        debug = False,
    )

    duration = round(time.time() - start_time, 3)
    print(f"{duration:7.3f} sec: entity linking")


    ## construct the _lemma graph_
    start_time = time.time()

    tg.construct_lemma_graph(
        debug = False,
    )

    duration = round(time.time() - start_time, 3)
    print(f"{duration:7.3f} sec: construct graph")


    ## perform relation extraction
    start_time = time.time()

    inferred_edges: list = tg.infer_relations(
        pipe,
        wikidata_api = textgraphs.WIKIDATA_API,
        max_skip = textgraphs.MAX_SKIP,
        opennre_min_prob = textgraphs.OPENNRE_MIN_PROB,
        debug = False,
    )

    duration = round(time.time() - start_time, 3)
    print(f"{duration:7.3f} sec: relation extraction, {len(inferred_edges)} edges")

    n_list: list = list(tg.nodes.values())

    df_rel: pd.DataFrame = pd.DataFrame.from_dict([
        {
            "src": n_list[edge.src_node].text,
            "dst": n_list[edge.dst_node].text,
            "rel": textgraphs.WikiDatum.normalize_prefix(edge.rel),
            "weight": edge.prob,
        }
        for edge in inferred_edges
    ])

    ic(df_rel)

    ## rank phrases
    start_time = time.time()

    tg.calc_phrase_ranks(
        pr_alpha = textgraphs.PAGERANK_ALPHA,
        debug = False,
    )

    duration = round(time.time() - start_time, 3)
    print(f"{duration:7.3f} sec: rank phrases")

    # show the results
    ic(tg.get_phrases_as_df(pipe))

    sys.exit(0)
    ic(tg.edges)  # pylint: disable=W0101
    ic(tg.nodes)


    # EXPERIMENT
    #sys.exit(0)

    ## stack profiler report
    profiler.stop()
    profiler.print()

    #print(tg.dump_lemma_graph())
