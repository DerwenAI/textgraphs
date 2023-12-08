#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Package definitions.

see copyright/license https://huggingface.co/spaces/DerwenAI/textgraph/blob/main/README.md
"""

from .doc import TextGraph

from .elem import Edge, LinkedEntity, Node, NodeEnum, NounChunk, RelEnum

from .pipe import Pipeline, PipelineFactory

from .rebel import Rebel

from .util import calc_quantile_bins, root_mean_square, stripe_column

from .vis import RenderPyVis


__title__ = "textgraph: raw texts, KGs, and LLMs, oh my!"

__description__ = "Using textgraph + LLM + graph ML for entity extraction, linking, and ranking"

__copyright__ = "2023, Derwen, Inc."

__author__ = """\n""".join([
    "derwen.ai <info@derwen.ai>"
])

__version__ = "0.1.0"
__release__ = "0.1.0"
