#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright ©2023 Derwen, Inc. All rights reserved.

"""
Sample application to demo the `textgraph` library.
"""

import sys  # pylint: disable=W0611
import time

from icecream import ic  # pylint: disable=E0401
import spacy  # pylint: disable=E0401

from textgraph import TextGraph


if __name__ == "__main__":
    SRC_TEXT: str = """
Werner Herzog is a remarkable filmmaker and intellectual originally from Germany, the son of Dietrich Herzog.
    """

    start_time: float = time.time()

    tg: TextGraph = TextGraph()

    duration: float = round(time.time() - start_time, 3)
    print(f"start: {round(duration, 3)} sec")


    start_time = time.time()

    sample_doc: spacy.tokens.doc.Doc = tg.build_doc(
        SRC_TEXT.strip(),
        ner_model = None,
    )

    duration = round(time.time() - start_time, 3)
    print(f"parse: {round(duration, 3)} sec")


    #sys.exit(0)

    start_time = time.time()

    tg.build_graph_embeddings(
        sample_doc,
    )

    duration = round(time.time() - start_time, 3)
    print(f"build graph: {round(duration, 3)} sec")


    start_time = time.time()

    tg.infer_relations(
        SRC_TEXT.strip(),
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
