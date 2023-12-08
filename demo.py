#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sample application to demo the `textgraph` library.

see copyright/license https://huggingface.co/spaces/DerwenAI/textgraph/blob/main/README.md
"""

import sys  # pylint: disable=W0611
import time

from icecream import ic  # pylint: disable=E0401

from textgraph import Pipeline, PipelineFactory, TextGraph


if __name__ == "__main__":
    SRC_TEXT: str = """
Werner Herzog is a remarkable filmmaker and an intellectual originally from Germany, the son of Dietrich Herzog.
After the war, Werner fled to America to become famous.
"""

    # set up
    start_time: float = time.time()

    tg: TextGraph = TextGraph()

    fabrica: PipelineFactory = PipelineFactory(
        dbpedia_api = PipelineFactory.DBPEDIA_API,
        ner_model = None,
    )

    duration: float = round(time.time() - start_time, 3)
    print(f"set up: {round(duration, 3)} sec")


    # NLP parse
    start_time = time.time()

    pipe: Pipeline = fabrica.build_pipeline(
        SRC_TEXT.strip(),
    )

    duration = round(time.time() - start_time, 3)
    print(f"parse: {round(duration, 3)} sec")


    # build lemma graph
    start_time = time.time()

    tg.build_graph_embeddings(
        pipe,
        debug = False,
    )

    duration = round(time.time() - start_time, 3)
    print(f"build graph: {round(duration, 3)} sec")


    # rank phrases
    start_time = time.time()

    tg.calc_phrase_ranks(
        debug = False,
    )

    duration = round(time.time() - start_time, 3)
    print(f"rank phrases: {round(duration, 3)} sec")

    # print the resulting entities extracted from the document
    ic(tg.get_phrases_as_df())

    sys.exit(0)

    ic(tg.edges)  # pylint: disable=W0101
    ic(tg.nodes)


    # infer relations
    start_time = time.time()

    tg.infer_relations(
        pipe,
    )

    duration = round(time.time() - start_time, 3)
    print(f"infer rel: {round(duration, 3)} sec")


    #print(tg.dump_lemma_graph())
