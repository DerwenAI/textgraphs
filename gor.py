#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Experiment with deserializing a node-link graph,
then transform it into a _graph of relations_
"""

import pathlib
import typing

import matplotlib.pyplot as plt  # pylint: disable=E0401

import textgraphs


if __name__ == "__main__":
    graph: textgraphs.GraphOfRelations = textgraphs.GraphOfRelations(
        textgraphs.SimpleGraph()
    )

    graph.load_ingram(
        pathlib.Path("examples/ingram.json"),
        debug = False,  # True
    )

    graph.seeds(
        debug = True,  # False
    )

    graph.trace_source_graph()

    graph.construct_gor(
        debug = True,  # False
    )

    _scores: typing.Dict[ tuple, float ] = graph.get_affinity_scores(
        debug = True,  # False
    )

    graph.trace_metrics(_scores)

    graph.render_gor(_scores)
    plt.show()
