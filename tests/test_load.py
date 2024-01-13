#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
unit tests:

  * serialization and deserialization

see copyright/license https://huggingface.co/spaces/DerwenAI/textgraphs/blob/main/README.md
"""

from os.path import abspath, dirname
import json
import pathlib
import sys

import deepdiff  # pylint: disable=E0401

sys.path.insert(0, str(pathlib.Path(dirname(dirname(abspath(__file__))))))
import textgraphs  # pylint: disable=C0413


def test_load_minimal (
    *,
    debug: bool = False,
    ) -> None:
    """
Construct a _lemma graph_ from a minimal example, then compare
serialized and deserialized data to ensure no fields get corrupted
in the conversions.
    """
    text: str = """
See Spot run.
    """

    tg: textgraphs.TextGraphs = textgraphs.TextGraphs()  # pylint: disable=C0103
    pipe: textgraphs.Pipeline = tg.create_pipeline(text.strip())

    # serialize into node-link format
    tg.collect_graph_elements(pipe)
    tg.construct_lemma_graph()
    tg.calc_phrase_ranks()

    json_str: str = tg.dump_lemma_graph()
    exp_graph = json.loads(json_str)

    # deserialize from node-link format
    tg = textgraphs.TextGraphs()  # pylint: disable=C0103
    tg.load_lemma_graph(json_str)
    tg.construct_lemma_graph()

    obs_graph: dict = json.loads(tg.dump_lemma_graph())

    if debug:
        print(obs_graph)

    # compare
    diff: deepdiff.diff.DeepDiff = deepdiff.DeepDiff(exp_graph, obs_graph)

    if debug:
        print(diff)

    if len(diff) > 0:
        print(json.dumps(json.loads(diff.to_json()), indent = 2))

    assert len(diff) == 0


if __name__ == "__main__":
    test_load_minimal(debug = True)
