#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sample application to demo the `textgraph` library.

see copyright/license https://huggingface.co/spaces/DerwenAI/textgraph/blob/main/README.md
"""

import sys  # pylint: disable=W0611
import time

from icecream import ic  # pylint: disable=E0401
from pyinstrument import Profiler  # pylint: disable=E0401

from textgraph import Pipeline, PipelineFactory, TextGraph


if __name__ == "__main__":
    SRC_TEXT: str = """
Werner Herzog is a remarkable filmmaker and an intellectual originally from Germany, the son of Dietrich Herzog.
After the war, Werner fled to America to become famous.
"""

    ## set up
    profiler: Profiler = Profiler()
    profiler.start()

    start_time: float = time.time()
    tg: TextGraph = TextGraph()

    fabrica: PipelineFactory = PipelineFactory(
        dbpedia_api = PipelineFactory.DBPEDIA_API,
        ner_model = None,
    )

    duration: float = round(time.time() - start_time, 3)
    print(f"set up: {round(duration, 3)} sec")


    ## NLP parse
    start_time = time.time()

    pipe: Pipeline = fabrica.build_pipeline(
        SRC_TEXT.strip(),
    )

    duration = round(time.time() - start_time, 3)
    print(f"parse text: {round(duration, 3)} sec")


    ## collect graph elements from the parse
    start_time = time.time()

    tg.collect_graph_elements(
        pipe,
        debug = False,
    )

    duration = round(time.time() - start_time, 3)
    print(f"collect elements: {round(duration, 3)} sec")


    ## perform entity linking
    start_time = time.time()

    tg.perform_entity_linking(
        pipe,
        debug = False,
    )

    duration = round(time.time() - start_time, 3)
    print(f"entity linking: {round(duration, 3)} sec")


    ## construct the _lemma graph_
    start_time = time.time()

    tg.construct_lemma_graph(
        debug = False,
    )

    duration = round(time.time() - start_time, 3)
    print(f"construct graph: {round(duration, 3)} sec")


    ## perform relation extraction
    start_time = time.time()

    inferred_edges: list = tg.infer_relations(
        pipe,
        max_skip = TextGraph.MAX_SKIP,
        opennre_min_prob = TextGraph.OPENNRE_MIN_PROB,
        debug = False,
    )

    duration = round(time.time() - start_time, 3)
    print(f"relation extraction: {round(duration, 3)} sec, {len(inferred_edges)} edges")


    ## rank phrases
    start_time = time.time()

    tg.calc_phrase_ranks(
        debug = False,
    )

    duration = round(time.time() - start_time, 3)
    print(f"rank phrases: {round(duration, 3)} sec")

    # show the results
    ic(tg.get_phrases_as_df())


    #sys.exit(0)

    ic(tg.edges)  # pylint: disable=W0101
    ic(tg.nodes)


    # EXPERIMENT
    #sys.exit(0)

    ## stack profiler report
    profiler.stop()
    profiler.print()

    #print(tg.dump_lemma_graph())
