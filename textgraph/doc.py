#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright ©2023 Derwen, Inc. All rights reserved.

"""
Implementation of an LLM-augmented `textgraph` algorithm for
constructing a _knowledge graph_ from raw, unstructured text source.

This integrates examples from:

  * <https://derwen.ai/s/mqqm>
  * <https://github.com/tomaarsen/SpanMarkerNER>
  * <https://github.com/thunlp/OpenNRE/>
  * <https://youtu.be/C9p7suS-NGk?si=7Ohq3BV654ia2Im4>
  * <https://towardsdatascience.com/how-to-convert-any-text-into-a-graph-of-concepts-110844f22a1a>
"""

from collections import OrderedDict
import itertools
import json
import pathlib
import typing

from icecream import ic  # pylint: disable=E0401
import networkx as nx  # pylint: disable=E0401
import numpy as np  # pylint: disable=E0401
import opennre  # pylint: disable=E0401
import pandas as pd  # pylint: disable=E0401
import pulp  # pylint: disable=E0401
import spacy  # pylint: disable=E0401

from .elem import Edge, Node, NodeEnum, RelEnum
from .pipe import Pipeline
from .util import calc_quantile_bins, root_mean_square, stripe_column


class TextGraph:
    """
Construct a _lemma graph_ from the unstructured text source, then
extract ranked phrases using a `textgraph` algorithm.
    """
    MAX_SKIP: int = 11
    NER_MAP: str = "dat/ner_map.json"
    NRE_MODEL: str = "wiki80_cnn_softmax"


    def __init__ (
        self,
        ) -> None:
        """
Constructor.
        """
        self.nodes: typing.Dict[ str, Node ] = OrderedDict()
        self.edges: typing.Dict[ str, Edge ] = {}
        self.tokens: typing.List[ Node ] = []
        self.lemma_graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self.nre: opennre.model.softmax_nn.SoftmaxNN = opennre.get_model(self.NRE_MODEL)


    def _make_node (  # pylint: disable=R0913
        self,
        key: str,
        span: spacy.tokens.token.Token,
        kind: NodeEnum,
        text_id: int,
        para_id: int,
        sent_id: int,
        *,
        label: typing.Optional[ str ] = None,
        linked: bool = True,
        ) -> Node:
        """
Lookup and return a `Node` object:

    * default: link matching keys into the same node
    * instantiate a new node if it does not exist already
        """
        location: typing.List[ int ] = [  # type: ignore
            text_id,
            para_id,
            sent_id,
            span.i,
        ]

        if not linked:
            # construct a placeholder node (stopwords)
            self.nodes[key] = Node(
                len(self.nodes),
                key,
                span,
                span.text,
                span.pos_,
                kind,
                [ location ],
            )

        elif key in self.nodes:
            # link to previously constructed entity node
            self.nodes[key].loc.append(location)
            self.nodes[key].count += 1

        # construct a new node for entity or lemma
        else:
            self.nodes[key] = Node(
                len(self.nodes),
                key,
                span,
                span.text,
                span.pos_,
                kind,
                [ location ],
                label = label,
                count = 1,
            )

        node: Node = self.nodes.get(key)  # type: ignore

        if kind not in [ NodeEnum.CHU ]:
            self.tokens.append(node)

        return node  # type: ignore


    def _make_edge (  # pylint: disable=R0913
        self,
        src_node: Node,
        dst_node: Node,
        kind: RelEnum,
        rel: str,
        prob: float,
        ) -> typing.Optional[ Edge ]:
        """
Lookup an edge, creating a new one if it does not exist already,
and increment the count if it does.
        """
        key: str = ".".join([
            str(src_node.node_id),
            str(dst_node.node_id),
            rel.replace(" ", "_"),
            str(kind.value),
        ])

        if key in self.edges:
            self.edges[key].count += 1

        elif src_node.node_id != dst_node.node_id:
            # no loopback
            self.edges[key] = Edge(
                src_node.node_id,
                dst_node.node_id,
                kind,
                rel,
                prob,
            )

        return self.edges.get(key)


    def _extract_phrases (  # pylint: disable=R0913
        self,
        sent_id: int,
        sent: spacy.tokens.span.Span,
        text_id: int,
        para_id: int,
        lemma_iter: typing.Iterator[ str ],
        *,
        debug: bool = False,
        ) -> typing.Iterator[ Node ]:
        """
Extract phrases from the parsed document to build nodes in the
_lemma graph_, while giving priority to:

  1. NER entities+labels
  2. lemmatized nouns and verbs
  3. noun chunks intersecting with entities
        """
        # extract entities using NER
        ent_seq: typing.List[ spacy.tokens.span.Span ] = list(sent.ents)

        if debug:
            ic(ent_seq)

        for token in sent:
            head = ( token.head, token.head.i, )

            if debug:
                ic(
                    token,
                    token.i,
                    token.dep_,
                    head,
                )

            if len(ent_seq) > 0 and ent_seq[0].start == token.i:
                # link a named entity
                ent = ent_seq.pop(0)

                yield self._make_node(
                    next(lemma_iter),  # pylint: disable=R1708
                    token,
                    NodeEnum.ENT,
                    text_id,
                    para_id,
                    sent_id,
                    label = ent.label_,
                )

            elif token.pos_ in [ "NOUN", "PROPN", "VERB" ]:
                # link a lemmatized entity
                yield self._make_node(
                    Pipeline.get_lemma_key(token),
                    token,
                    NodeEnum.LEM,
                    text_id,
                    para_id,
                    sent_id,
                )

            else:
                # fall-through case: use token as a placeholder in the lemma graph
                yield self._make_node(
                    Pipeline.get_lemma_key(token, placeholder = True),
                    token,
                    NodeEnum.DEP,
                    text_id,
                    para_id,
                    sent_id,
                    linked = False,
                )


    def _overlay_noun_chunks (
        self,
        pipe: Pipeline,
        *,
        text_id: int = 0,
        para_id: int = 0,
        debug: bool = False,
        ) -> None:
        """
Identify the unique noun chunks, i.e., those which differ from the
entities and lemmas that have already been linked in the lemma graph.
        """
        # scan the noun chunks for uniqueness
        for chunk in pipe.link_noun_chunks(self.nodes, self.tokens):
            if chunk.unseen:
                location: typing.List[ int ] = [
                    text_id,
                    para_id,
                    chunk.sent_id,
                    chunk.start,
                ]

                if chunk.lemma_key in self.nodes:
                    node = self.nodes.get(chunk.lemma_key)
                    node.loc.append(location)  # type: ignore
                    node.count += 1  # type: ignore
                else:
                    node = Node(
                        len(self.nodes),
                        chunk.lemma_key,
                        chunk.span,
                        chunk.text,
                        "noun_chunk",
                        NodeEnum.CHU,
                        [ location ],
                        count = 1,
                    )

                    self.nodes[chunk.lemma_key] = node

                # add the related edges, which do not necessarily
                # correspond 1:1 with the existing nodes
                for token_id in range(chunk.start, chunk.start + chunk.length):
                    if debug:
                        ic(self.tokens[token_id])

                    self._make_edge(
                        node,  # type: ignore
                        self.tokens[token_id],
                        RelEnum.CHU,
                        "noun_chunk",
                        1.0,
                    )


    def _build_lemma_graph (
        self,
        ) -> None:
        """
Construct the _lemma graph_ from the collected elements: parse
dependencies, lemmas, entities, and noun chunks.
        """
        # add the nodes
        self.lemma_graph.add_nodes_from([
            node.node_id
            for node in self.nodes.values()
        ])

        # populate the node properties
        for node_key, node in self.nodes.items():
            nx_node = self.lemma_graph.nodes[node.node_id]
            nx_node["name"] = node.text
            nx_node["kind"] = str(node.kind)
            nx_node["value"] = node.weight
            nx_node["iri"] = node.label
            nx_node["title"] = node_key
            nx_node["size"] = node.count
            nx_node["subobj"] = node.sub_obj
            nx_node["pos"] = node.pos
            nx_node["loc"] = str(node.loc)

        # add the edges and their properties
        self.lemma_graph.add_edges_from([
            (
                edge.src_node,
                edge.dst_node,
                {
                    "kind": str(edge.kind),
                    "title": edge.rel,
                    "weight": float(edge.count),
                    "prob": edge.prob,
                    "count": edge.count,
                },
            )
            for edge_key, edge in self.edges.items()
        ])


    def build_graph_embeddings (
        self,
        pipe: Pipeline,
        *,
        text_id: int = 0,
        para_id: int = 0,
        ner_map_path: pathlib.Path = pathlib.Path(NER_MAP),
        debug: bool = True,
        ) -> None:
        """
Construct a _lemma graph_ from the results of running the `textgraph`
algorithm, represented in `NetworkX` as a directed graph with parallel
edges.

In other words, this represents "reverse embeddings" from the parsed
document.

    ner_map_path: maps from OntoTypes4 to an IRI; defaults to local file `dat/ner_map.json`
        """
        # load the NER map
        ner_map: typing.Dict[ str, dict ] = OrderedDict(
            json.loads(ner_map_path.read_text(encoding = "utf-8"))
        )

        # parse each sentence
        lemma_iter: typing.Iterator[ str ] = pipe.get_ent_lemma_keys()

        for sent_id, sent in enumerate(pipe.ent_doc.sents):
            if debug:
                ic(sent_id, sent, sent.start)

            sent_nodes: typing.List[ Node ] = list(self._extract_phrases(
                sent_id,
                sent,
                text_id,
                para_id,
                lemma_iter,
            ))

            if debug:
                ic(sent_nodes)

            for node in sent_nodes:
                # remap OntoTypes4 values to more general-purpose IRIs
                if node.label is not None:
                    iri: typing.Optional[ dict ] = ner_map.get(node.label)

                    try:
                        if iri is not None:
                            node.label = iri["iri"]
                    except TypeError as ex:
                        ic(ex)
                        print(f"unknown label: {node.label}")

                # link parse elements, based on the token's head
                head_idx: int = node.span.head.i

                if head_idx >= len(sent_nodes):
                    head_idx -= sent.start

                if debug:
                    ic(node, len(sent_nodes), node.span.head.i, node.span.head.text, head_idx)

                self._make_edge(
                    node,
                    sent_nodes[head_idx],
                    RelEnum.DEP,
                    node.span.dep_,
                    1.0,
                )

                # annotate src nodes which are subjects or direct objects
                if node.span.dep_ in [ "nsubj", "pobj" ]:
                    node.sub_obj = True

        # overlay unique noun chunks onto the parsed elements,
        self._overlay_noun_chunks(
            pipe,
            text_id = text_id,
            para_id = para_id,
            debug = debug,
        )

        # build the _lemma graph_ used for analysis
        self._build_lemma_graph()


    def infer_relations (
        self,
        pipe: Pipeline,
        *,
        max_skip: int = MAX_SKIP,
        debug: bool = True,
        ) -> None:
        """
Run NRE to infer relations between pairs of co-occurring entities.
        """
        inferred_edges: typing.List[ Edge ] = []

        ent_list: typing.List[ Node ] = [
            node
            for node in self.nodes.values()
            if node.label is not None
        ]

        for pair in itertools.product(ent_list, repeat = 2):
            if pair[0] != pair[1]:
                src: Node = pair[0]
                dst: Node = pair[1]

                try:
                    path: typing.List[ int ] = nx.shortest_path(
                        self.lemma_graph.to_undirected(as_view = True),
                        source = src.node_id,
                        target = dst.node_id,
                        weight = "weight",
                        method = "dijkstra",
                    )

                    if debug:
                        ic(src.node_id, dst.node_id, path)

                    if len(path) <= max_skip:
                        rel, prob = self.nre.infer({
                            "text": pipe.text,
                            "h": { "pos": src.get_pos() },
                            "t": { "pos": dst.get_pos() },
                        })

                        if debug:
                            ic(rel, prob)

                        edge: Edge = self._make_edge(  # type: ignore
                            src,
                            dst,
                            RelEnum.INF,
                            rel,
                            prob,
                        )

                        inferred_edges.append(edge)
                except Exception as ex:  # pylint: disable=W0718
                    ic(ex)
                    ic(src, dst)

        # add edges from the inferred relations
        self.lemma_graph.add_edges_from([
            (
                edge.src_node,
                edge.dst_node,
                {
                    "weight": edge.prob,
                },
            )
            for edge in inferred_edges
        ])


    @classmethod
    def _solve_restack_coeffs (
        cls,
        sum_e: float,
        sum_l: float,
        min_e: float,
        max_l: float,
        *,
        debug: bool = True,
        ) -> typing.Tuple[ float, float ]:
        """
Solve for the rank coefficients using a `pulp` linear programming model.
        """
        coef0: pulp.LpVariable = pulp.LpVariable("coef0", 0)  # coef for ranked entities
        coef1: pulp.LpVariable = pulp.LpVariable("coef1", 0)  # coef for ranked lemmas
        slack: pulp.LpVariable = pulp.LpVariable("slack", 0)  # "stack gap" slack variable

        prob: pulp.LpProblem = pulp.LpProblem("restack_coeffs", pulp.LpMinimize)
        prob += coef0 * sum_e + coef1 * sum_l + slack == 1.0
        prob += coef0 * min_e - coef1 * max_l - slack == 0.0
        prob += coef0 - coef1 >= 0

        # final expression becomes the objective function
        prob += slack

        status: int = prob.solve(
            pulp.PULP_CBC_CMD(msg = False),
        )

        if debug:
            ic(pulp.LpStatus[status])
            ic(pulp.value(coef0))
            ic(pulp.value(coef1))
            ic(pulp.value(slack))

        return ( pulp.value(coef0), pulp.value(coef1), )  # type: ignore


    def _restack_ranks (  # pylint: disable=R0914
        self,
        ranks: typing.List[ float ],
        *,
        debug: bool = True,  # False
        ) -> typing.List[ float ]:
        """
Stack-rank the nodes so that entities have priority over lemmas.
        """
        # build a dataframe of node ranks and counts
        df1: pd.DataFrame = pd.DataFrame.from_dict([
            {
                "weight": ranks[node.node_id],
                "count": node.count,
                "neighbors": node.neighbors,
                "subobj": int(node.sub_obj),
            }
            for node in self.nodes.values()
        ])

        df1.loc[df1["count"] < 1, "weight"] = 0

        # normalize by column and calculate quantiles
        df2: pd.DataFrame = df1.apply(lambda x: x / x.max(), axis = 0)
        bins: np.ndarray = calc_quantile_bins(len(df2.index))

        # stripe each columns
        df3: pd.DataFrame = pd.DataFrame([
            stripe_column(values, bins)
            for _, values in df2.items()
        ]).T

        # renormalize the ranks
        df1["rank"] = df3.apply(root_mean_square, axis=1)
        df1.loc[df1["count"] < 1, "rank"] = 0

        rank_col: np.ndarray = df1["rank"].to_numpy()
        rank_col /= sum(rank_col)

        # prepare to stack entities atop lemmas
        df1["E"] = df1["rank"]
        df1["L"] = df1["rank"]

        df1["entity"] = [
            node.kind == NodeEnum.ENT
            for node in self.nodes.values()
        ]

        df1.loc[~df1["entity"], "E"] = 0
        df1.loc[df1["entity"], "L"] = 0

        # knock out the verbs
        df1["pos"] = [
            node.pos
            for node in self.nodes.values()
        ]

        df1.loc[df1["pos"] == "VERB", "E"] = 0
        df1.loc[df1["pos"] == "VERB", "L"] = 0

        if debug:
            ic(df1)

        E: typing.List[ float ] = [  # pylint: disable=C0103
            rank
            for rank in df1["E"].to_numpy()
            if rank > 0.0
        ]

        L: typing.List[ float ] = [  # pylint: disable=C0103
            rank
            for rank in df1["L"].to_numpy()
            if rank > 0.0
        ]

        # just use the calculated ranks when either list is empty
        if len(E) < 1 or len(L) < 1:
            return ranks

        # configure a system of linear equations
        coef0, coef1 = self._solve_restack_coeffs(
            sum_e = sum(E),
            sum_l = sum(L),
            min_e = min(E),
            max_l = max(L),
            debug = debug,
        )

        df1["stacked"] = df1["E"] * coef0 + df1["L"] * coef1

        if debug:
            ic(df1)

        return list(df1["stacked"].to_numpy())


    def calc_phrase_ranks (
        self,
        *,
        pr_alpha: float = 0.85,
        debug: bool = False,
        ) -> None:
        """
Calculate the weights for each node in the _lemma graph_, then
stack-rank the nodes so that entities have priority over lemmas.

Phrase ranks are normalized to sum to 1.0 and these now represent
the ranked entities extracted from the document.
        """
        for node in self.nodes.values():
            nx_node = self.lemma_graph.nodes[node.node_id]
            neighbors: int = 0

            try:
                neighbors = len(list(nx.neighbors(self.lemma_graph, node.node_id)))
            except Exception:  # pylint: disable=W0718
                pass
            finally:
                node.neighbors = neighbors
                nx_node["neighbors"] = neighbors

        # restack
        ranks: typing.List[ float ] = self._restack_ranks(
            list(nx.pagerank(
                self.lemma_graph,
                alpha = pr_alpha,
            ).values()),
            debug = debug,
        )

        # update the node weights
        for i, node in enumerate(self.nodes.values()):
            node.weight = ranks[i]


    def get_phrases (
        self,
        ) -> typing.Iterator[ Node ]:
        """
Return the entities extracted from the document.
        """
        for node in sorted(
                [
                    node
                    for node in self.nodes.values()
                    if node.weight > 0
                ],
                key = lambda n: n.weight,
                reverse = True,
        ):
            yield node


    def get_phrases_as_df (
        self,
        ) -> pd.DataFrame:
        """
Return the ranked extracted entities as a `pandas.DataFrame`
        """
        return pd.DataFrame.from_dict([
            {
                "node_id": node.node_id,
                "text": node.text,
                "pos": node.pos,
                "label": node.label,
                "count": node.count,
                "weight": node.weight,
            }
            for node in self.get_phrases()
        ])


    def dump_lemma_graph (
        self,
        ) -> str:
        """
Dump the _lemma graph_ as a JSON string in _node-link_ format,
suitable for serialization and subsequent use in JavaScript,
Neo4j, Graphistry, etc.
        """
        return json.dumps(
            nx.node_link_data(self.lemma_graph),
            sort_keys = True,
            indent =  2,
            separators = ( ",", ":" ),
        )
