#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright ©2023 Derwen, Inc. All rights reserved.

"""
Sample application to demo the `textgraph` library.
"""

import sys  # pylint: disable=W0611
import typing

from icecream import ic  # pylint: disable=E0401
import spacy  # pylint: disable=E0401

from textgraph import Node, TextGraph


if __name__ == "__main__":
    SRC_TEXT: str = """
Werner Herzog is a remarkable filmmaker and intellectual originally from Germany, the son of Dietrich Herzog.
    """
    tg: TextGraph = TextGraph()

    sample_doc: spacy.tokens.doc.Doc = tg.build_doc(
        SRC_TEXT.strip(),
        use_llm = False,
    )

    #sys.exit(0)

    tg.build_graph_embeddings(
        sample_doc,
    )

    tg.infer_relations(
        SRC_TEXT.strip(),
    )

    tg.calc_phrase_ranks()

    ic(tg.edges)
    ic(tg.nodes)

    # print the resulting entities extracted from the document
    results: typing.List[ Node ] = sorted(
        [
            node
            for node in tg.nodes.values()
            if node.weight > 0
        ],
        key = lambda n: n.weight,
        reverse = True,
    )

    ic(results)
