#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Experiment with transform graph data into a _graph of relations_.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import enum
import itertools
import pathlib
import json
import sys
import typing

from icecream import ic  # pylint: disable=E0401
import networkx as nx  # pylint: disable=E0401
import pandas as pd  # pylint: disable=E0401
import pyvis  # pylint: disable=E0401

from .elem import Edge, Node, NodeEnum, RelEnum
from .graph import SimpleGraph


class RelDir (enum.IntEnum):
    """
Enumeration for the directions of a relation.
    """
    HEAD = 0  # relation flows into node
    TAIL = 1  # relation flows out of node

    def __str__ (
        self
        ) -> str:
        """
Codec for representing as a string.
        """
        decoder: typing.List[ str ] = [
            "head",
            "tail",
        ]

        return decoder[self.value]


@dataclass(order=False, frozen=False)
class SheafSeed:
    """
A data class representing a node from the source graph plus its
partial edge, based on a _Sheaf Theory_ decomposition of a graph.
    """
    node_id: int
    rel_id: int
    rel_dir: RelDir
    edge: Edge


@dataclass(order=False, frozen=False)
class TransArc:
    """
A data class representing one transformed rel-node-rel triple in
a _graph of relations_.
    """
    pair_key: tuple
    a_rel: int
    b_rel: int
    node_id: int
    a_dir: RelDir
    b_dir: RelDir


@dataclass(order=False, frozen=False)
class Affinity:
    """
A data class representing the affinity scores from one entity
in the transformed _graph of relations_.

NB: there are much more efficient ways to calculate these
_affinity scores_ using sparse tensor algebra; this approach
illustrates the process -- for research and debugging.
    """
    pairs: typing.Dict[ int, Counter ] = field(default_factory = lambda: defaultdict(Counter))
    scores: typing.Dict[ int, float ] = field(default_factory = lambda: {})
    tally: int = 0


class GraphOfRelations:  # pylint: disable=R0902
    """
Attempt to reproduce results published in
"INGRAM: Inductive Knowledge Graph Embedding via Relation Graphs"
<https://arxiv.org/abs/2305.19987>
    """

    def __init__ (
        self,
        source: SimpleGraph
        ) -> None:
        """
Constructor.
        """
        self.source: SimpleGraph = source
        self.rel_list: typing.List[ str ] = []

        self.node_list: typing.List[ Node ] = []
        self.edge_list: typing.List[ Edge ] = []

        self.seed_links: typing.Dict[ int, list ] = defaultdict(list)

        self.head_affin: typing.Dict[ int, Affinity ] = defaultdict(Affinity)
        self.tail_affin: typing.Dict[ int, Affinity ] = defaultdict(Affinity)

        # NB: should load these from the dataset?
        self.pub_score: typing.Dict[ tuple, float ] = {
            (0, 1): .22,
            (0, 2): .50,
            (1, 2): .33,
            (1, 4): .11,
            (2, 4): .11,
            (3, 4): .81,
            (3, 5): .11,
            (4, 5): .36,
        }


    def load_ingram (
        self,
        json_file: pathlib.Path,
        *,
        debug: bool = False,
        ) -> None:
        """
Load data for a source graph, as illustrated in _InGram_
        """
        with open(json_file, "r", encoding = "utf-8") as fp:  # pylint: disable=C0103,W0621
            dat: dict = json.load(fp)

            # JSON file provides an ordered list of relations
            # to simplify tracing/debugging
            self.rel_list = dat["rels"]

            # build the src node of the triple
            for src_name, links in dat["ents"].items():
                src_node: Node = self.source.make_node(
                    [],
                    src_name,
                    None,
                    NodeEnum.ENT,
                    0,
                    0,
                    0,
                )

                for rel_name, dst_name in links:
                    # error-check input
                    if rel_name not in self.rel_list:
                        print("Unknown relation:", rel_name)
                        sys.exit(-1)

                    # build the dst node of the triple
                    dst_node: Node = self.source.make_node(
                        [],
                        dst_name,
                        None,
                        NodeEnum.ENT,
                        0,
                        0,
                        0,
                    )

                    # create an edge between src/dst
                    edge: Edge = self.source.make_edge(  # type: ignore  # pylint: disable=W0612,W0621
                        src_node,
                        dst_node,
                        RelEnum.SYN,
                        rel_name,
                        1.0,
                    )

        if debug:
            print(self.source.nodes)
            print(self.source.edges)
            print(self.rel_list)


    def seeds (
        self,
        *,
        debug: bool = False,
        ) -> None:
        """
Prep data for the topological transform illustrated in _InGram_
        """
        self.node_list = list(self.source.nodes.values())
        self.edge_list = list(self.source.edges.values())

        if debug:
            print("\n--- triples in source graph ---")

        for edge in self.source.edges.values():
            if edge.rel not in self.rel_list:
                self.rel_list.append(edge.rel)

            rel_id: int = self.rel_list.index(edge.rel)

            if debug:
                ic(edge.src_node, rel_id, edge.dst_node)
                print("", self.node_list[edge.src_node].text, edge.rel, self.node_list[edge.dst_node].text)  # pylint: disable=C0301

            # enumerate the partially decoupled links ("seeds")
            # for the topological transform:
            self.seed_links[edge.dst_node].append(SheafSeed(
                edge.dst_node,
                rel_id,
                RelDir.HEAD,
                edge,
            ))

            self.seed_links[edge.src_node].append(SheafSeed(
                edge.src_node,
                rel_id,
                RelDir.TAIL,
                edge,
            ))


    def trace_source_graph (
        self
        ) -> None:
        """
Output a "seed" representation of the source graph.
        """
        print("\n--- nodes in source graph ---")

        for node in self.source.nodes.values():
            # CONFIRMED: correct according to examples in the paper
            print(f"n: {node.node_id:2}, {node.text}")

            head_edges = [
                ( seed.edge.src_node, seed.edge.rel, seed.edge.dst_node, )
                for seed in self.seed_links[node.node_id]
                if seed.rel_dir == RelDir.HEAD
            ]

            print("", "head:", head_edges)

            tail_edges = [
                ( seed.edge.src_node, seed.edge.rel, seed.edge.dst_node, )
                for seed in self.seed_links[node.node_id]
                if seed.rel_dir == RelDir.TAIL
            ]

            print("", "tail:", tail_edges)

        print("\n--- edges in source graph ---")

        for rel_id, rel in enumerate(self.rel_list):
            print(f"e: {rel_id:2}, {rel}")


    def _transformed_triples (
        self,
        *,
        debug: bool = False,
        ) -> typing.Iterator[ TransArc ]:
        """
Generate the transformed triples for a _graph of relations_.
        """
        for node_id, seeds in sorted(self.seed_links.items()):
            if debug:
                ic(node_id, len(seeds))

            for seed_a, seed_b in itertools.combinations(seeds, 2):
                pair_key: tuple = tuple(sorted([ seed_a.rel_id, seed_b.rel_id ]))

                if debug:
                    print(f" {pair_key} {seed_a.edge.rel}.{seed_a.rel_dir} {self.node_list[node_id].text} {seed_b.edge.rel}.{seed_b.rel_dir}")  # pylint: disable=C0301

                trans_arc: TransArc = TransArc(
                    pair_key,
                    seed_a.rel_id,
                    seed_b.rel_id,
                    node_id,
                    seed_a.rel_dir,
                    seed_b.rel_dir,
                )

                yield trans_arc


    def construct_gor (
        self,
        *,
        debug: bool = False,
        ) -> None:
        """
Perform the topological transform described by _InGram_, constructing
a _graph of relations_ (GOR) and calculating _affinity scores_ between
entities in the GOR based on their definitions:

> we measure the affinity between two relations by considering how many
entities are shared between them and how frequently they share the same
entity
        """
        if debug:
            print("\n--- transformed triples ---")

        for trans_arc in self._transformed_triples(debug = debug):
            if debug:
                ic(trans_arc)
                print()

            if trans_arc.a_dir == RelDir.HEAD:
                self.head_affin[trans_arc.a_rel].pairs[trans_arc.b_rel][trans_arc.node_id] += 1
            else:
                self.tail_affin[trans_arc.a_rel].pairs[trans_arc.b_rel][trans_arc.node_id] += 1

            if trans_arc.b_dir == RelDir.HEAD:
                self.head_affin[trans_arc.b_rel].pairs[trans_arc.a_rel][trans_arc.node_id] += 1
            else:
                self.tail_affin[trans_arc.b_rel].pairs[trans_arc.a_rel][trans_arc.node_id] += 1


    @classmethod
    def tally_frequencies (
        cls,
        counter: Counter,
        ) -> int:
        """
Tally the frequency of shared entities.
        """
        sum_freq: int = counter.total()  # type: ignore

        for occur in counter.values():  # pylint: disable=W0612
            sum_freq += 1

        return sum_freq


    def _collect_tallies (
        self,
        *,
        debug: bool = False,
        ) -> None:
        """
Collect tallies, in preparation for calculating the affinity scores.
        """
        if debug:
            print("\n--- collect shared entity tallies ---")

        for rel_a, rel in enumerate(self.rel_list):
            for rel_b, counter in sorted(self.head_affin[rel_a].pairs.items()):
                tally: int = self.tally_frequencies(counter)
                self.head_affin[rel_a].scores[rel_b] = float(tally)
                self.head_affin[rel_a].tally += tally

            for rel_b, counter in sorted(self.tail_affin[rel_a].pairs.items()):
                tally = self.tally_frequencies(counter)
                self.tail_affin[rel_a].scores[rel_b] = float(tally)
                self.tail_affin[rel_a].tally += tally

            if debug:
                print(rel_a, rel)
                print(" h:", self.head_affin[rel_a].tally, self.head_affin[rel_a].scores.items())
                print(" t:", self.tail_affin[rel_a].tally, self.tail_affin[rel_a].scores.items())


    def get_affinity_scores (
        self,
        *,
        debug: bool = False,
        ) -> typing.Dict[ tuple, float ]:
        """
Reproduce metrics based on the example published in _InGram_
        """
        self._collect_tallies(debug = debug)

        scores: typing.Dict[ tuple, float ] = {}
        n_rels: int = len(self.rel_list)

        pairs: typing.Set[ tuple ] = {
            tuple(sorted([ rel_a, rel_b ]))
            for rel_a in range(n_rels)
            for rel_b in range(n_rels)
        }

        for rel_a, rel_b in sorted(list(pairs)):
            pair_affin: float = 0.0

            if rel_b in self.head_affin and rel_a in self.tail_affin:
                rel_a_sum = self.head_affin[rel_a].tally + self.tail_affin[rel_a].tally
                a_contrib = self.tally_frequencies(self.head_affin[rel_b].pairs[rel_a])

                rel_b_sum = self.head_affin[rel_b].tally + self.tail_affin[rel_b].tally
                b_contrib = self.tally_frequencies(self.tail_affin[rel_a].pairs[rel_b])

                pair_affin += (a_contrib / float(rel_a_sum)) + (b_contrib / float(rel_b_sum))

            if rel_b in self.tail_affin and rel_a in self.head_affin:
                rel_a_sum = self.head_affin[rel_a].tally + self.tail_affin[rel_a].tally
                a_contrib = self.tally_frequencies(self.tail_affin[rel_b].pairs[rel_a])

                rel_b_sum = self.head_affin[rel_b].tally + self.tail_affin[rel_b].tally
                b_contrib = self.tally_frequencies(self.head_affin[rel_a].pairs[rel_b])

                pair_affin += (a_contrib / float(rel_a_sum)) + (b_contrib / float(rel_b_sum))

            if pair_affin > 0.0:
                pair_key: tuple = tuple(sorted([ rel_a, rel_b ]))
                scores[pair_key] = pair_affin / 2.0

        return scores


    def trace_metrics (
        self,
        scores: typing.Dict[ tuple, float ],
        ) -> pd.DataFrame:
        """
Compare the calculated affinity scores with results from a published
example.
        """
        df_compare: pd.DataFrame = pd.DataFrame.from_dict([
            {
                "pair": pair_key,
                "rel_a": self.rel_list[pair_key[0]],
                "rel_b": self.rel_list[pair_key[1]],
                "affinity": round(aff, 2),
                "expected": self.pub_score.get(pair_key)
            }
            for pair_key, aff in sorted(scores.items())
        ])

        return df_compare


    def _build_nx_graph (
        self,
        scores: typing.Dict[ tuple, float ],
        ) -> nx.Graph:
        """
Construct a network representation of the _graph of relations_
in `NetworkX`
        """
        vis_graph: nx.Graph = nx.Graph()

        vis_graph.add_nodes_from([
            (
                rel_id,
                {
                    "label": rel,
                },
            )
            for rel_id, rel in enumerate(self.rel_list)
        ])

        vis_graph.add_edges_from([
            (
                rel_a,
                rel_b,
                {
                    "weight": affinity,
                },
            )
            for (rel_a, rel_b), affinity in scores.items()
        ])

        return vis_graph


    def render_gor_plt (
        self,
        scores: typing.Dict[ tuple, float ],
        ) -> None:
        """
Visualize the _graph of relations_ using `matplotlib`
        """
        vis_graph: nx.Graph = self._build_nx_graph(scores)

        node_labels: typing.Dict[ int, str ] = dict(enumerate(self.rel_list))

        edge_labels: typing.Dict[ int, str ] = {
            edge_id: str(round(vis_graph.edges[edge_id]["weight"], 2))
            for edge_id in vis_graph.edges
        }

        pos: dict = nx.spring_layout(
            vis_graph,
            k = 2.0,
        )

        nx.draw_networkx(
            vis_graph,
            pos,
            labels = node_labels,
            with_labels = True,
            node_color = "#eee",
            edge_color = "#bbb",
            font_size = 9,
        )

        nx.draw_networkx_edge_labels(
            vis_graph,
            pos,
            edge_labels = edge_labels,
        )


    def render_gor_pyvis (
        self,
        scores: typing.Dict[ tuple, float ],
        ) -> pyvis.network.Network:
        """
Visualize the _graph of relations_ interactively using `PyVis`
        """
        pv_graph: pyvis.network.Network = pyvis.network.Network()
        pv_graph.from_nx(self._build_nx_graph(scores))

        for pv_edge in pv_graph.get_edges():
            pair_key: tuple = ( pv_edge["from"], pv_edge["to"], )
            aff: typing.Optional[ float ] = scores.get(pair_key)

            if aff is not None:
                pv_edge["title"] = round(aff, 2)
                pv_edge["label"] = round(aff, 2)
                pv_edge["width"] = int(aff * 10.0)

        return pv_graph
