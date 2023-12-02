#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright ©2023 Derwen, Inc. All rights reserved.

"""
Sample application to demo the `textgraph` library.
"""

import sys  # pylint: disable=W0611
import time

from icecream import ic  # pylint: disable=E0401

from textgraph import Pipeline, PipelineFactory, TextGraph


if __name__ == "__main__":
    SRC_TEXT: str = """
Werner Herzog is a remarkable filmmaker and intellectual originally from Germany, the son of Dietrich Herzog.
    """

    start_time: float = time.time()

    tg: TextGraph = TextGraph()

    fabrica: PipelineFactory = PipelineFactory(
        ner_model = None,
    )

    duration: float = round(time.time() - start_time, 3)
    print(f"set up: {round(duration, 3)} sec")


    start_time = time.time()

    pipe: Pipeline = fabrica.build_pipeline(
        SRC_TEXT.strip(),
    )

    duration = round(time.time() - start_time, 3)
    print(f"parse: {round(duration, 3)} sec")


    start_time = time.time()

    tg.build_graph_embeddings(
        pipe,
    )

    duration = round(time.time() - start_time, 3)
    print(f"build graph: {round(duration, 3)} sec")


    start_time = time.time()

    tg.infer_relations(
        pipe,
    )

    duration = round(time.time() - start_time, 3)
    print(f"infer rel: {round(duration, 3)} sec")


    start_time = time.time()

    tg.calc_phrase_ranks()

    duration = round(time.time() - start_time, 3)
    print(f"rank phrases: {round(duration, 3)} sec")


    ic(tg.edges)
    ic(tg.nodes)

    # print the resulting entities extracted from the document
    for node in tg.get_phrases():
        ic(node)
