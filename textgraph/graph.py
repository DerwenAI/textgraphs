#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright ©2023 Derwen, Inc. All rights reserved.

"""
Classes used for graph representation:

  * superset of openCypher, adding support for a probabilitic graph
  * IRI discipline also guarantees export to RDF, e.g., for transitive closure
"""

from dataclasses import dataclass
import typing

import spacy  # pylint: disable=E0401


@dataclass(order=False, frozen=False)
class Node:
    """
A data class representing one node, i.e., an extracted phrase.
    """
    node_id: int
    key: str
    span: typing.Union[ spacy.tokens.span.Span, spacy.tokens.token.Token ]
    text: str
    kind: typing.Optional[ str ] = None
    count: int = 1
    weight: float = 0.0


@dataclass(order=False, frozen=False)
class Edge:
    """
A data class representing an edge between two nodes.
    """
    src_node: int
    dst_node: int
    count: int = 1
