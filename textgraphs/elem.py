#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
These classes represent graph elements.

Consider this "flavor" of graph representation to be a superset of
`openCypher` _labeled property graphs_ (LPG) with additional support
for probabilistic graphs.

Imposing a discipline of IRIs for node names and edge relations
helps guarantee that a view of the graph can be exported to RDF
for data quality checks, transitive closure, semantic inference,
and so on.

see copyright/license https://huggingface.co/spaces/DerwenAI/textgraphs/blob/main/README.md
"""

from dataclasses import dataclass, field
import enum
import typing

import spacy  # pylint: disable=E0401


######################################################################
## class definitions

@dataclass(order=False, frozen=False)
class KGSearchHit:  # pylint: disable=R0902
    """
A data class representing a hit from a _knowledge graph_ search.
    """
    iri: str
    label: str
    descrip: str
    aliases: typing.List[ str ]
    prob: float


@dataclass(order=False, frozen=False)
class LinkedEntity:  # pylint: disable=R0902
    """
A data class representing one linked entity.
    """
    span: spacy.tokens.span.Span
    iri: str
    length: int
    rel: str
    prob: float
    token_id: int
    kg_ent: KGSearchHit
    count: int = 1


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
    entity: typing.List[ LinkedEntity ] = field(default_factory = lambda: [])
    annotated: bool = False


    def get_linked_label (
        self
        ) -> typing.Optional[ str ]:
        """
When this node has a linked entity, return that IRI.
Otherwise return is `label` value.
        """
        if len(self.entity) > 0:
            return self.entity[0].iri

        return self.label


    def get_name (
        self
        ) -> str:
        """
Return a brief name for the graphical depiction of this Node.
        """
        if self.kind == NodeEnum.IRI:
            return self.label  # type: ignore
        if self.kind == NodeEnum.LEM:
            return self.key

        return self.text


    def get_stacked_count (
        self
        ) -> int:
        """
Return a modified count, to redact verbs and linked entities from
the stack-rank partitions.
        """
        if self.pos == "VERB" or self.kind == NodeEnum.IRI:
            return 0

        return self.count


    def get_pos (
        self
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
    INF = 2  # `REBEL` or `OpenNRE` inferred relation
    SYN = 3  # `sense2vec` inferred synonym
    IRI = 4  # `DBPedia` or `Wikidata` linked entity

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
