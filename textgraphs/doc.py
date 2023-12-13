#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implementation of an LLM-augmented `textgraph` algorithm for
constructing a _lemma graph_ from raw, unstructured text source.
The results provide elements for semi-automated construction or
augmentation of a _knowledge graph_.

This class maintains the state of a graph. Updates get applied by
running methods on `Pipeline` objects, typically per paragraph.

see copyright/license https://huggingface.co/spaces/DerwenAI/textgraphs/blob/main/README.md
"""

from collections import OrderedDict
import itertools
import json
import pathlib
import sys
import traceback
import typing

from icecream import ic  # pylint: disable=E0401
import networkx as nx  # pylint: disable=E0401
import numpy as np  # pylint: disable=E0401
import pandas as pd  # pylint: disable=E0401
import pulp  # pylint: disable=E0401
import spacy  # pylint: disable=E0401

from .defaults import DBPEDIA_MIN_ALIAS, DBPEDIA_MIN_SIM, DBPEDIA_SEARCH_API, \
    MAX_SKIP, NER_MAP, OPENNRE_MIN_PROB, PAGERANK_ALPHA, WIKIDATA_API
from .elem import Edge, LinkedEntity, Node, NodeEnum, RelEnum
from .pipe import Pipeline, PipelineFactory
from .rebel import Rebel
from .util import calc_quantile_bins, root_mean_square, stripe_column

# determine whether this is loading into a Jupyter notebook,
# to allow for `tqdm` progress bars
if "ipykernel" in sys.modules:
    from tqdm.notebook import tqdm  # pylint: disable=E0401,W0611
else:
    from tqdm import tqdm  # pylint: disable=E0401


class TextGraphs:
    """
Construct a _lemma graph_ from the unstructured text source,
then extract ranked phrases using a `textgraph` algorithm.
    """

    def __init__ (
        self,
        *,
        factory: typing.Optional[ PipelineFactory ] = None,
        ) -> None:
        """
Constructor.
        """
        self.nodes: typing.Dict[ str, Node ] = OrderedDict()
        self.edges: typing.Dict[ str, Edge ] = {}
        self.tokens: typing.List[ Node ] = []
        self.lemma_graph: nx.MultiDiGraph = nx.MultiDiGraph()

        # initialize the pipeline factory
        if factory is not None:
            self.factory = factory
        else:
            self.factory = PipelineFactory()


    def create_pipeline (
        self,
        text_input: str,
        ) -> Pipeline:
        """
Use the pipeline factory to create a pipeline (e.g., `spaCy.Document`)
for each text input, which are typically paragraph-length.
        """
        return self.factory.create_pipeline(
            text_input,
        )


    ######################################################################
    ## graph construction

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
        length: int = 1,
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
                loc = [ location ],
                length = length,
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
                loc = [ location ],
                label = label,
                length = length,
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
            # preclude cycles in the graph
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
        lemma_iter: typing.Iterator[ typing.Tuple[ str, int ]],
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
                lemma_key, span_len = next(lemma_iter)  # pylint: disable=R1708

                yield self._make_node(
                    lemma_key,
                    token,
                    NodeEnum.ENT,
                    text_id,
                    para_id,
                    sent_id,
                    label = ent.label_,
                    length = span_len,
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
                        loc = [ location ],
                        length = chunk.length,
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


    def collect_graph_elements (
        self,
        pipe: Pipeline,
        *,
        text_id: int = 0,
        para_id: int = 0,
        ner_map_path: pathlib.Path = pathlib.Path(NER_MAP),
        debug: bool = False,
        ) -> None:
        """
Collect the elements of a _lemma graph_ from the results of running
the `textgraph` algorithm. These elements include: parse dependencies,
lemmas, entities, and noun chunks.

    ner_map_path: map OntoTypes4 to IRI; defaults to local file `dat/ner_map.json`
        """
        # load the NER map
        ner_map: typing.Dict[ str, dict ] = OrderedDict(
            json.loads(ner_map_path.read_text(encoding = "utf-8"))
        )

        # parse each sentence
        lemma_iter: typing.Iterator[ typing.Tuple[ str, int ]] = pipe.get_ent_lemma_keys()

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


    def construct_lemma_graph (
        self,
        *,
        debug: bool = False,
        ) -> None:
        """
Construct the base level of the _lemma graph_ from the collected
elements. This gets represented in `NetworkX` as a directed graph
with parallel edges.
        """
        # add the nodes
        self.lemma_graph.add_nodes_from([
            node.node_id
            for node in self.nodes.values()
        ])

        # populate the minimum required node properties
        for node_key, node in self.nodes.items():
            nx_node = self.lemma_graph.nodes[node.node_id]
            nx_node["title"] = node_key
            nx_node["size"] = node.count
            nx_node["value"] = node.weight

            if debug:
                ic(nx_node)

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


    ######################################################################
    ## entity linking

    def _make_link (
        self,
        link: LinkedEntity,
        rel: str,
        *,
        debug: bool = False,
        ) -> None:
        """
Link to previously constructed entity node;
otherwise construct a new node for this linked entity.
        """
        if debug:
            ic(link)

        if link.iri in self.nodes:
            self.nodes[link.iri].count += 1

        else:
            self.nodes[link.iri] = Node(
                len(self.nodes),
                link.iri,
                link.span,
                link.wiki_ent.descrip,
                rel,
                NodeEnum.IRI,
                label = link.iri,
                length = link.length,
                count = 1,
            )

        dst_node: Node = self.nodes.get(link.iri)  # type: ignore

        if debug:
            ic(dst_node)

        # back-link to the parsed entity object
        self.tokens[link.token_id].entity.append(link)

        # construct a directed edge between them
        self._make_edge(
            self.tokens[link.token_id],
            dst_node,
            RelEnum.IRI,
            rel,
            link.prob,
        )


    def perform_entity_linking (
        self,
        pipe: Pipeline,
        *,
        dbpedia_search_api: str = DBPEDIA_SEARCH_API,
        min_alias: float = DBPEDIA_MIN_ALIAS,
        min_similarity: float = DBPEDIA_MIN_SIM,
        debug: bool = False,
        ) -> None:
        """
Perform _entity linking_ based on `DBPedia Spotlight` and other services.
        """
        # first pass: use DBPedia Spotlight
        iter_ents: typing.Iterator[ LinkedEntity ] = pipe.link_dbpedia_spotlight_entities(
            self.tokens,
            dbpedia_search_api,
            min_alias,
            min_similarity,
            debug = debug
        )

        for link in iter_ents:
            self._make_link(
                link,
                "dbpedia",
                debug = debug,
            )

        # second pass: use DBPedia search on unlinked entities
        iter_ents = pipe.link_dbpedia_search_entities(
            list(self.nodes.values()),
            dbpedia_search_api,
            min_alias,
            debug = debug
        )

        for link in iter_ents:
            self._make_link(
                link,
                "dbpedia",
                debug = debug,
            )


    ######################################################################
    ## relation extraction

    def _iter_entity_pairs (
        self,
        max_skip: int,
        *,
        debug: bool = True,
        ) -> typing.Iterator[ typing.Tuple[ Node, Node ]]:
        """
Iterator for entity pairs for which the algorithm infers relations.
        """
        ent_list: typing.List[ Node ] = [
            node
            for node in self.nodes.values()
            if node.kind in [ NodeEnum.ENT ]
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
                        yield ( src, dst, )
                except nx.NetworkXNoPath:
                    pass
                except Exception as ex:  # pylint: disable=W0718
                    ic(ex)
                    ic("ERROR", src, dst)
                    traceback.print_exc()


    def _iter_rel_opennre (
        self,
        pipe: Pipeline,
        wikidata_api: str,
        max_skip: int,
        opennre_min_prob: float,
        *,
        debug: bool = True,
        ) -> typing.Iterator[ Edge ]:
        """
Iterate on entity pairs to drive `OpenNRE`, to infer relations
        """
        # error-check the model config
        if self.factory.nre is None:
            return

        for src, dst in self._iter_entity_pairs(max_skip, debug = debug):
            rel, prob = self.factory.nre.infer({  # type: ignore
                "text": pipe.text,
                "h": { "pos": src.get_pos() },
                "t": { "pos": dst.get_pos() },
            })

            if prob >= opennre_min_prob:
                if debug:
                    ic(src.text, dst.text)
                    ic(rel, prob)

                # Wikidata lookup
                iri: typing.Optional[ str ] = pipe.wiki.resolve_wikidata_rel_iri(
                    rel,
                    wikidata_api,
                )

                if iri is None:
                    iri = "opennre:" + rel.replace(" ", "_")

                # construct an Edge
                edge: Edge = self._make_edge(  # type: ignore
                    src,
                    dst,
                    RelEnum.INF,
                    iri,
                    prob,
                )

                yield edge  # type: ignore


    def _iter_rel_rebel (
        self,
        pipe: Pipeline,
        wikidata_api: str,
        *,
        debug: bool = True,
        ) -> typing.Iterator[ Edge ]:
        """
Iterate on sentences to drive `REBEL`, yielding inferred relations.
        """
        rebel: Rebel = Rebel()

        for sent in pipe.ent_doc.sents:
            extract: str = rebel.tokenize_sent(str(sent).strip())
            triples: typing.List[ dict ] = rebel.extract_triplets_typed(extract)

            tok_map: dict = {
                token.text: self.tokens[token.i]
                for token in sent
            }

            if debug:
                ic(extract, triples)

            for triple in triples:
                src: typing.Optional[ Node ] = tok_map.get(triple["head"])
                dst: typing.Optional[ Node ] = tok_map.get(triple["tail"])
                rel: str = triple["rel"]

                if src is not None and dst is not None:
                    if debug:
                        ic(src, dst, rel)

                    # Wikidata lookup
                    iri: typing.Optional[ str ] = pipe.wiki.resolve_wikidata_rel_iri(
                        rel,
                        wikidata_api,
                    )

                    if iri is None:
                        iri = "mrebel:" + rel.replace(" ", "_")

                    # construct an Edge
                    edge = self._make_edge(  # type: ignore
                        src,
                        dst,
                        RelEnum.INF,
                        iri,
                        1.0,
                    )

                    yield edge  # type: ignore


    def infer_relations (
        self,
        pipe: Pipeline,
        *,
        wikidata_api: str = WIKIDATA_API,
        max_skip: int = MAX_SKIP,
        opennre_min_prob: float = OPENNRE_MIN_PROB,
        debug: bool = True,
        ) -> typing.List[ Edge ]:
        """
Multiple approaches infer relations among co-occurring entities:

  * `OpenNRE`
  * `REBEL`
        """
        inferred_edges: typing.List[ Edge ] = list(
            itertools.chain(
                self._iter_rel_opennre(
                    pipe,
                    wikidata_api,
                    max_skip,
                    opennre_min_prob,
                    debug = debug,
                ),
                self._iter_rel_rebel(
                    pipe,
                    wikidata_api,
                    debug = debug,
                ),
            )
        )

        # add edges from inferred relations
        self.lemma_graph.add_edges_from([
            (
                edge.src_node,
                edge.dst_node,
                {
                    "weight": edge.prob,
                    "title": edge.rel,
                },
            )
            for edge in inferred_edges
        ])

        return inferred_edges


    ######################################################################
    ## rank the extracted and linked phrases

    @classmethod
    def _solve_restack_coeffs (
        cls,
        sum_e: float,
        sum_l: float,
        min_e: float,
        max_l: float,
        *,
        debug: bool = False,
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
        debug: bool = False,
        ) -> typing.List[ float ]:
        """
Stack-rank the nodes so that entities have priority over lemmas.
        """
        # build a dataframe of node ranks and counts
        df1: pd.DataFrame = pd.DataFrame.from_dict([
            {
                "weight": ranks[node.node_id],
                "count": node.get_stacked_count(),
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

        if debug:
            ic(df1)

        # partition the lists to be stacked
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
        pr_alpha: float = PAGERANK_ALPHA,
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
        pipe: Pipeline,
        ) -> typing.Iterator[ dict ]:
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

            label: str = pipe.wiki.normalize_prefix(node.get_linked_label())  # type: ignore  # pylint: disable=C0301

            yield {
                "node_id": node.node_id,
                "text": node.text,
                "pos": node.pos,
                "label": label,
                "count": node.count,
                "weight": node.weight,
            }


    def get_phrases_as_df (
        self,
        pipe: Pipeline,
        ) -> pd.DataFrame:
        """
Return the ranked extracted entities as a `pandas.DataFrame`
        """
        return pd.DataFrame.from_dict(self.get_phrases(pipe))


    ######################################################################
    ## handle data exports

    def dump_lemma_graph (
        self,
        ) -> str:
        """
Dump the _lemma graph_ as a JSON string in _node-link_ format,
suitable for serialization and subsequent use in JavaScript,
Neo4j, Graphistry, etc.
        """
        # populate the optional node properties
        for node in self.nodes.values():
            nx_node = self.lemma_graph.nodes[node.node_id]
            nx_node["name"] = node.text
            nx_node["kind"] = str(node.kind)
            nx_node["iri"] = node.label
            nx_node["subobj"] = node.sub_obj
            nx_node["pos"] = node.pos
            nx_node["loc"] = str(node.loc)

        return json.dumps(
            nx.node_link_data(self.lemma_graph),
            sort_keys = True,
            indent =  2,
            separators = ( ",", ":" ),
        )
