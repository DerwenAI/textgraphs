#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
unit tests:

  * extract the top-k entities from a raw text

see copyright/license https://huggingface.co/spaces/DerwenAI/textgraph/blob/main/README.md
"""

from os.path import abspath, dirname
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(dirname(dirname(abspath(__file__))))))
from textgraph import Pipeline, PipelineFactory, TextGraph  # pylint: disable=C0413


def test_extract_herzog ():
    """
Run an extract with the Werner Herzog blurb.
    """
    text: str = """
Werner Herzog is a remarkable filmmaker and intellectual originally from Germany, the son of Dietrich Herzog.
    """
    tg: TextGraph = TextGraph()  # pylint: disable=C0103

    fabrica: PipelineFactory = PipelineFactory(
        ner_model = None,
    )

    pipe: Pipeline = fabrica.build_pipeline(
        text.strip(),
    )

    tg.build_graph_embeddings(
        pipe,
        debug = False,
    )

    tg.calc_phrase_ranks(
        debug = False,
    )

    results: list = [
        ( row["text"], row["pos"], )
        for _, row in tg.get_phrases_as_df().iterrows()
    ]

    # top-k, k=4
    results = results[:4]

    expects: list = [
        ("Germany", "PROPN"),
        ("Werner Herzog", "PROPN"),
        ("Dietrich Herzog", "PROPN"),
    ]

    for pair in expects:
        assert pair in results


if __name__ == "__main__":
    test_extract_herzog()
