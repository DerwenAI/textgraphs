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


class NodeEnum (enum.IntEnum):
    """
Enumeration for the kinds of node categories
    """
    DEP = 0  # `spaCy` parse dependency
    LEM = 1  # lemmatized token
    ENT = 2  # named entity
    CHU = 3  # noun chunk

    def __str__ (
        self
        ) -> str:
        """
Codec for representing as a string.
        """
        decoder: typing.List[ str ] = [
            "dep",
            "lem",
            "ent",
            "chu",
        ]

        return decoder[self.value]


@dataclass(order=False, frozen=False)
class Node:  # pylint: disable=R0902
    """
A data class representing one node, i.e., an extracted phrase.
    """
    node_id: int
    key: str
    span: spacy.tokens.token.Token
    text: str
    pos: str
    kind: NodeEnum
    loc: typing.List[ typing.List[ int ] ]
    label: typing.Optional[ str ] = None
    sub_obj: bool = False
    count: int = 0
    neighbors: int = 0
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
    DEP = 0  # `spaCy` parse dependency
    CHU = 1  # `spaCy` noun chunk
    INF = 2  # `OpenNRE` inferred relation
    SYN = 3  # `sense2vec` inferred synonym

    def __str__ (
        self
        ) -> str:
        """
Codec for representing as a string.
        """
        decoder: typing.List[ str ] = [
            "dep",
            "inf",
            "syn",
            "chu",
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
