#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright ©2023 Derwen, Inc. All rights reserved.

"""
unit tests:

  * extract the top-k entities from a raw text
"""

from os.path import abspath, dirname
import pathlib
import sys

import spacy  # pylint: disable=E0401

sys.path.insert(0, str(pathlib.Path(dirname(dirname(abspath(__file__))))))
from textgraph import TextGraph  # pylint: disable=C0413


def test_extract_herzog ():
    """
Run an extract with the Werner Herzog blurb.
    """
    text: str = """
Werner Herzog is a remarkable filmmaker and intellectual originally from Germany, the son of Dietrich Herzog.
    """
    tg: TextGraph = TextGraph()  # pylint: disable=C0103

    sample_doc: spacy.tokens.doc.Doc = tg.build_doc(
        text.strip(),
        ner_model = None,
    )

    tg.build_graph_embeddings(
        sample_doc,
        debug = False,
    )

    tg.calc_phrase_ranks(
        debug = False,
    )

    results: list = [
        ( row["text"], row["pos"], )
        for _, row in tg.get_phrases_as_df().iterrows()
    ]

    # top-k, k=3
    results = results[:3]

    expects: list = [
        ("Germany", "PROPN"),
        ("Werner Herzog", "PROPN"),
        ("Dietrich Herzog", "PROPN"),
    ]

    for pair in expects:
        assert pair in results


if __name__ == "__main__":
    test_extract_herzog()
