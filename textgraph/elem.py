#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Classes used for graph representation:

  * superset of openCypher, adding support for a probabilitic graph
  * IRI discipline also guarantees export to RDF, e.g., for transitive closure

see copyright/license https://huggingface.co/spaces/DerwenAI/textgraph/blob/main/README.md
"""

from dataclasses import dataclass, field
import enum
import typing

import spacy  # pylint: disable=E0401


@dataclass(order=False, frozen=False)
class LinkedEntity:  # pylint: disable=R0902
    """
A data class representing one noun chunk, i.e., a candidate as an extracted phrase.
    """
    span: spacy.tokens.span.Span
    iri: str
    length: int
    rel: str
    prob: float
    count: int
    token_id: int


@dataclass(order=False, frozen=False)
class NounChunk:  # pylint: disable=R0902
    """
A data class representing one noun chunk, i.e., a candidate as an extracted phrase.
    """
    span: spacy.tokens.span.Span
    text: str
    length: int
    lemma_key: str
    unseen: bool
    sent_id: int
    start: int = 0


class NodeEnum (enum.IntEnum):
    """
Enumeration for the kinds of node categories
    """
    DEP = 0  # `spaCy` parse dependency
    LEM = 1  # lemmatized token
    ENT = 2  # named entity
    CHU = 3  # noun chunk
    IRI = 4  # IRI for linked entity

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
            "iri",
        ]

        return decoder[self.value]


@dataclass(order=False, frozen=False)
class Node:  # pylint: disable=R0902
    """
A data class representing one node, i.e., an extracted phrase.
    """
    node_id: int
    key: str
    span: typing.Union[ spacy.tokens.span.Span, spacy.tokens.token.Token ]
    text: str
    pos: str
    kind: NodeEnum
    loc: typing.List[ typing.List[ int ] ] = field(default_factory = lambda: [])
    label: typing.Optional[ str ] = None
    length: int = 1
    sub_obj: bool = False
    count: int = 0
    neighbors: int = 0
    weight: float = 0.0
    entity: typing.Optional[ LinkedEntity ] = None


    def get_linked_label (
        self,
        ) -> typing.Optional[ str ]:
        """
When this node has a linked entity, return that IRI.
Otherwise return is `label` value.
        """
        if self.entity is not None:
            return self.entity.iri

        return self.label


    def get_stacked_count (
        self,
        ) -> int:
        """
Return a modified count, to redact verbs and linked entities from
the stack-rank partitions.
        """
        if self.pos == "VERB" or self.kind == NodeEnum.IRI:
            return 0

        return self.count


    def get_pos (
        self,
        ) -> typing.Tuple[ int, int ]:
        """
Generate a position span for OpenNRE.
        """
        position: typing.Tuple[ int, int ] = ( self.span.idx, self.span.idx + len(self.text) - 1, )
        return position


class RelEnum (enum.IntEnum):
    """
Enumeration for the kinds of edge relations
    """
    DEP = 0  # `spaCy` parse dependency
    CHU = 1  # `spaCy` noun chunk
    INF = 2  # `OpenNRE` inferred relation
    SYN = 3  # `sense2vec` inferred synonym
    IRI = 4  # `DBPedia` linked entity

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
            "iri",
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
