#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright ©2023 Derwen, Inc. All rights reserved.

"""
Classes used for graph representation:

  * superset of openCypher, adding support for a probabilitic graph
  * IRI discipline also guarantees export to RDF, e.g., for transitive closure
"""

from dataclasses import dataclass
import enum
import typing

import spacy  # pylint: disable=E0401


@dataclass(order=False, frozen=False)
class Node:
    """
A data class representing one node, i.e., an extracted phrase.
    """
    node_id: int
    span: spacy.tokens.token.Token
    text: str
    pos: str
    kind: typing.Optional[ str ] = None
    count: int = 0
    weight: float = 0.0

    def get_pos (
        self,
        ) -> typing.Tuple[ int, int ]:
        """
Generate a position span for OpenNRE.
        """
        return (self.span.idx, self.span.idx + len(self.text) - 1)


class RelEnum (enum.IntEnum):
    """
Enumeration for the kinds of edge relations
    """
    DEP = 0
    INFER = 1

    def __str__ (
        self
        ) -> str:
        """
Codec for representing as a string.
        """
        decoder: typing.List[ str ] = [
            "dependency",
            "inferred",
        ]

        return decoder[self.value]


@dataclass(order=False, frozen=False)
class Edge:
    """
A data class representing an edge between two nodes.
    """
    src_node: int
    dst_node: int
    kind: RelEnum
    rel: str
    prob: float
    count: int = 1
