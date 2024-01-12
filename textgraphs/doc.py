#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=C0302

"""
Implementation of an LLM-augmented `textgraph` algorithm for
constructing a _lemma graph_ from raw, unstructured text source.
The results provide elements for semi-automated construction or
augmentation of a _knowledge graph_.

This class maintains the state of a graph. Updates get applied by
running methods on `Pipeline` objects, typically per paragraph.

see copyright/license https://huggingface.co/spaces/DerwenAI/textgraphs/blob/main/README.md
"""

import asyncio
import logging
import os
import sys
import typing

from icecream import ic  # pylint: disable=E0401
import networkx as nx  # pylint: disable=E0401
import numpy as np  # pylint: disable=E0401
import pandas as pd  # pylint: disable=E0401
import pulp  # pylint: disable=E0401
import rdflib  # pylint: disable=E0401
import spacy  # pylint: disable=E0401
import transformers  # pylint: disable=E0401
import urllib3  # pylint: disable=E0401

from .defaults import PAGERANK_ALPHA
from .elem import Edge, Node, NodeEnum, RelEnum
from .graph import SimpleGraph
from .pipe import Pipeline, PipelineFactory
from .util import calc_quantile_bins, root_mean_square, stripe_column
from .vis import RenderPyVis


######################################################################
## repair the libraries which are borked:

# workaround: determine whether this is loading into a Jupyter
# notebook, to allow for `tqdm` progress bars
if "ipykernel" in sys.modules:
    from tqdm.notebook import tqdm  # pylint: disable=E0401,W0611
else:
    from tqdm import tqdm  # pylint: disable=E0401

# override: HF `transformers` and `tokenizers` have noisy logging
transformers.logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "0"

# override: `OpenNRE` uses `word2vec` which has noisy logging
logging.disable(logging.INFO)

# override: WikidMedia and others allow their SSL certs to expire
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


######################################################################
## class definitions

class TextGraphs (SimpleGraph):
    """
Construct a _lemma graph_ from the unstructured text source,
then extract ranked phrases using a `textgraph` algorithm.
    """
    IRI_BASE: str = "https://github.com/DerwenAI/textgraphs/ns/"


    def __init__ (
        self,
        *,
        factory: typing.Optional[ PipelineFactory ] = None,
        iri_base: str = IRI_BASE,
        ) -> None:
        """
Constructor.

    factory:
optional `PipelineFactory` used to configure components
        """
        super().__init__()

        self.iri_base = iri_base

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

    text_input:
raw text to be parsed by this pipeline

    returns:
a configured pipeline
        """
        return self.factory.create_pipeline(
            text_input,
        )


    def create_render (
        self
        ) -> RenderPyVis:
        """
Create an object for rendering the graph in `PyVis` HTML+JavaScript.

    returns:
a configured `RenderPyVis` object for generating graph visualizations
        """
        return RenderPyVis(
            self,
            self.factory.kg,
        )


    def _extract_phrases (  # pylint: disable=R0913
        self,
        pipe: Pipeline,
        sent_id: int,
        sent: spacy.tokens.span.Span,
        text_id: int,
        para_id: int,
        lemma_iter: typing.Iterator[ typing.Tuple[ str, int ]],
        *,
        debug: bool = False,
        ) -> typing.Iterator[ Node ]:
        """
Extract phrases from a parsed document to build nodes in the
_lemma graph_, while considering:

  1. NER entities+labels
  2. lemmatized nouns and verbs
  3. noun chunks that overlap with entities

as the ordered priorities.

    pipe:
configured pipeline for this document

    sent_id:
sentence identifier

    sent:
token span for the parsed sentence

    text_id:
text (top-level document) identifier

    para_id:
paragraph identitifer

    lemma_iter:
iterator for parsed lemmas

    debug:
debugging flag

    yields:
extracted entities represented as `Node` objects in the graph
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

                yield self.make_node(
                    pipe.tokens,
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
                yield self.make_node(
                    pipe.tokens,
                    Pipeline.get_lemma_key(token),
                    token,
                    NodeEnum.LEM,
                    text_id,
                    para_id,
                    sent_id,
                )

            else:
                # fall-through case: use token as a placeholder in the lemma graph
                yield self.make_node(
                    pipe.tokens,
                    Pipeline.get_lemma_key(token, placeholder = True),
                    token,
                    NodeEnum.DEP,
                    text_id,
                    para_id,
                    sent_id,
                    linked = False,
                )


    def _make_class_link (
        self,
        node: Node,
        pipe: Pipeline,
        *,
        debug: bool = False,
        ) -> None:
        """
Private helper method to construct a link to an entity's class.

    node:
recognized entity to be linked

    pipe:
configured pipeline for this document

    debug:
debugging flag
        """
        if debug:
            print("link:", node.label)

        # special case of `make_node()`
        if node.label in self.nodes:
            dst: Node = self.nodes[node.label]
            dst.count += 1

        else:
            # find class IRI metadata
            class_meta: typing.List[ str ] = [
                meta["definition"]
                for meta in pipe.kg.NER_MAP.values()
                if meta["iri"] == node.label
            ]

            dst = Node(
                len(self.nodes),
                node.label,  # type: ignore
                class_meta[0],
                str(rdflib.RDF.type),
                NodeEnum.IRI,
                label = node.label,
                length = node.length,
                count = 1,
            )

            self.nodes[node.label] = dst  # type: ignore

        node.annotated = True

        # construct a directed edge between them
        edge: Edge = self.make_edge(  # type: ignore
            node,
            dst,
            RelEnum.IRI,
            str(rdflib.RDF.type),
            node.weight,
            debug = debug,
        )

        if debug:
            ic(edge)

        if edge is not None:
            pipe.edges.append(edge)


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

    pipe:
configured pipeline for this document

    text_id:
text (top-level document) identifier

    para_id:
paragraph identitifer

    debug:
debugging flag
        """
        # scan the noun chunks for uniqueness
        for chunk in pipe.link_noun_chunks(self.nodes):
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
                        chunk.text,
                        "noun_chunk",
                        NodeEnum.CHU,
                        span = chunk.span,
                        loc = [ location ],
                        length = chunk.length,
                        count = 1,
                    )

                    self.nodes[chunk.lemma_key] = node

                # add the related edges, which do not necessarily
                # correspond 1:1 with the existing nodes
                for token_id in range(chunk.start, chunk.start + chunk.length):
                    if debug:
                        ic(pipe.tokens[token_id])

                    edge: Edge = self.make_edge(
                        node,  # type: ignore
                        pipe.tokens[token_id],
                        RelEnum.CHU,
                        "noun_chunk",
                        1.0,
                        debug = debug,
                    )

                    if edge is not None:
                        pipe.edges.append(edge)


    def collect_graph_elements (
        self,
        pipe: Pipeline,
        *,
        text_id: int = 0,
        para_id: int = 0,
        debug: bool = False,
        ) -> None:
        """
Collect the elements of a _lemma graph_ from the results of running
the `textgraph` algorithm. These elements include: parse dependencies,
lemmas, entities, and noun chunks.

Make sure to call beforehand: `TextGraphs.create_pipeline()`

    pipe:
configured pipeline for this document

    text_id:
text (top-level document) identifier

    para_id:
paragraph identitifer

    debug:
debugging flag
        """
        # parse each sentence
        lemma_iter: typing.Iterator[ typing.Tuple[ str, int ]] = pipe.get_ent_lemma_keys()

        for sent_id, sent in enumerate(pipe.ner_doc.sents):
            if debug:
                ic(sent_id, sent, sent.start)

            sent_nodes: typing.List[ Node ] = list(self._extract_phrases(
                pipe,
                sent_id,
                sent,
                text_id,
                para_id,
                lemma_iter,
            ))

            if debug:
                ic(sent_nodes)

            for node in sent_nodes:
                # re-map from OntoTypes4 to a formal IRI, if possible
                # then link the inferred class
                if node.kind == NodeEnum.ENT:
                    node.label = pipe.kg.remap_ner(node.label)

                    if node.label is not None and node.label.startswith("http"):
                        self._make_class_link(
                            node,
                            pipe,
                            debug = debug,
                        )

                # link parse elements, based on the token's head
                head_idx: int = node.span.head.i  # type: ignore

                if head_idx >= len(sent_nodes):
                    head_idx -= sent.start

                if debug:
                    ic(node, len(sent_nodes), node.span.head.i, node.span.head.text, head_idx)  # type: ignore  # pylint: disable=C0301

                edge: Edge = self.make_edge(  # type: ignore
                    node,
                    sent_nodes[head_idx],
                    RelEnum.DEP,
                    node.span.dep_,  # type: ignore
                    1.0,
                    debug = debug,
                )

                if edge is not None:
                    pipe.edges.append(edge)

                # annotate src nodes which are subjects or direct objects
                if node.span.dep_ in [ "nsubj", "pobj" ]:  # type: ignore
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

Make sure to call beforehand: `TextGraphs.collect_graph_elements()`

    debug:
debugging flag
        """
        # add the nodes
        self.lemma_graph.add_nodes_from([
            node.node_id
            for node in self.nodes.values()
        ])

        # populate the minimum required node properties
        for node_key, node in self.nodes.items():
            nx_node = self.lemma_graph.nodes[node.node_id]
            nx_node["lemma"] = node_key
            nx_node["count"] = node.count
            nx_node["weight"] = node.weight

            if node.kind in [ NodeEnum.DEP ]:
                nx_node["label"] = ""
            elif node.kind in [ NodeEnum.IRI ]:
                nx_node["label"] = self.factory.kg.normalize_prefix(node.label)  # type: ignore
            else:
                nx_node["label"] = node.text

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
                    "lemma": edge_key,
                    "weight": float(edge.count),
                    "prob": edge.prob,
                    "count": edge.count,
                },
            )
            for edge_key, edge in self.edges.items()
        ])


    ######################################################################
    ## entity linking

    def perform_entity_linking (
        self,
        pipe: Pipeline,
        *,
        debug: bool = False,
        ) -> None:
        """
Perform _entity linking_ based on the `KnowledgeGraph` object.

Make sure to call beforehand: `TextGraphs.collect_graph_elements()`

    pipe:
configured pipeline for this document

    debug:
debugging flag
        """
        pipe.kg.perform_entity_linking(
            self,
            pipe,
            debug = debug,
        )

        # by default, link the baseline semantics
        for node in pipe.tokens:
            if node.kind == NodeEnum.LEM and node.label is None:
                node.label = str(rdflib.OWL.Thing)


    ######################################################################
    ## relation extraction

    def _infer_rel_construct_edge (
        self,
        src: Node,
        iri: str,
        dst: Node,
        *,
        debug: bool = False,
        ) -> Edge:
        """
Create an edge for the linked IRI, based on the input triple.

    src:
source node in the triple

    iri:
predicate IRI in the triple

    dst:
destination node in the triple

    debug:
debugging flag

    returns:
the constructed `Edge` object
        """
        edge = self.make_edge(  # type: ignore
            src,
            dst,
            RelEnum.INF,
            iri,
            1.0,
            debug = debug,
        )

        if debug:
            ic(edge)

        return edge  # type: ignore


    async def _consume_infer_rel (
        self,
        queue: asyncio.Queue,
        inferred_edges: typing.List[ Edge ],
        *,
        debug: bool = False,
        ) -> None:
        """
Consume from queue: inferred relations represented as triples.

    queue:
queue of inference tasks to be performed

    inferred_edges:
a list collecting the `Edge` objects inferred during this processing

    debug:
debugging flag
        """
        while True:
            src, iri, dst = await queue.get()

            inferred_edges.append(
                self._infer_rel_construct_edge(
                    src,
                    iri,
                    dst,
                    debug = debug,
                )
            )

            queue.task_done()


    async def infer_relations_async (
        self,
        pipe: Pipeline,
        *,
        debug: bool = False,
        ) -> typing.List[ Edge ]:
        """
Gather triples representing inferred relations and build edges,
concurrently by running an async queue.
<https://stackoverflow.com/questions/52582685/using-asyncio-queue-for-producer-consumer-flow>

Make sure to call beforehand: `TextGraphs.collect_graph_elements()`

    pipe:
configured pipeline for this document

    debug:
debugging flag

    returns:
a list of the inferred `Edge` objects
        """
        inferred_edges: typing.List[ Edge ] = []
        queue: asyncio.Queue = asyncio.Queue()

        producer_tasks: typing.List[ asyncio.Task ] = [
            asyncio.create_task(
                producer.gen_triples_async(  # type: ignore
                    pipe,
                    queue,
                    debug = debug,
                )
            )
            for producer in pipe.infer_rels
        ]

        consumer_task: asyncio.Task = asyncio.create_task(
            self._consume_infer_rel(
                queue,
                inferred_edges,
                debug = debug,
            )
        )

        # wait for producers to finish,
        # await the remaining tasks,
        # then cancel the now-idle consumer
        await asyncio.gather(*producer_tasks)

        if debug:
            ic("Queue: done producing")

        await queue.join()
        consumer_task.cancel()

        if debug:
            ic("Queue: done consuming")

        # update the graph
        pipe.edges.extend(inferred_edges)

        return inferred_edges


    def infer_relations (
        self,
        pipe: Pipeline,
        *,
        debug: bool = False,
        ) -> typing.List[ Edge ]:
        """
Gather triples representing inferred relations and build edges.

Make sure to call beforehand: `TextGraphs.collect_graph_elements()`

    pipe:
configured pipeline for this document

    debug:
debugging flag

    returns:
a list of the inferred `Edge` objects
        """
        inferred_edges: typing.List[ Edge ] = [
            self._infer_rel_construct_edge(src, iri, dst, debug = debug)
            for infer_rel in pipe.infer_rels
            for src, iri, dst in infer_rel.gen_triples(pipe, debug = debug)
        ]

        # update the graph
        pipe.edges.extend(inferred_edges)

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

    sum_e:
sum of the entity ranks

    sum_l:
sum of the lemma ranks

    min_e:
minimum among the entity ranks

    max_l:
maximum among the entity ranks

    debug:
debugging flag

    returns:
the calculated rank coefficients
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

    ranks:
list of calculated ranks per node

    debug:
debugging flag

    returns:
ordered list of re-stacked nodes
        """
        # build a dataframe of node ranks and counts
        df1: pd.DataFrame = pd.DataFrame.from_dict([
            {
                "weight": ranks[node.node_id],
                "count": node.get_stacked_count(),
                "hood": node.neighbors,
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

Make sure to call beforehand: `TextGraphs.construct_lemma_graph()`

    pr_alpha:
optional `alpha` parameter for the PageRank algorithm

    debug:
debugging flag
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
                nx_node["hood"] = neighbors

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
        self
        ) -> typing.Iterator[ dict ]:
        """
Return the entities extracted from the document.

Make sure to call beforehand: `TextGraphs.calc_phrase_ranks()`

    yields:
extracted entities
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

            label: str = self.factory.kg.normalize_prefix(node.get_linked_label())  # type: ignore  # pylint: disable=C0301

            yield {
                "node_id": node.node_id,
                "text": node.text,
                "pos": node.pos,
                "label": label,
                "count": node.count,
                "weight": node.weight,
            }


    def get_phrases_as_df (
        self
        ) -> pd.DataFrame:
        """
Return the ranked extracted entities as a dataframe.

Make sure to call beforehand: `TextGraphs.calc_phrase_ranks()`

    returns:
a `pandas.DataFrame` of the extracted entities
        """
        return pd.DataFrame.from_dict(self.get_phrases())


    ######################################################################
    ## knowledge graph abstraction layer

    def extract_rdf (  # pylint: disable=R0914
        self,
        *,
        lang: str = "en",
        ) -> str:
        """
Extract the entities and relations which have IRIs as RDF triples.

    lang:
language identifier

    returns:
RDF triples N3 (Turtle) format as a string
        """
        node_list: typing.List[ Node ] = list(self.nodes.values())
        node_keys: typing.List[ str ] = list(self.nodes.keys())
        ref_dict: typing.Dict[ int, rdflib.URIRef ] = {}
        rdf_graph: rdflib.Graph = rdflib.Graph()

        # extract entities as RDF
        for node_id, node in enumerate(self.nodes.values()):
            if node.kind in [ NodeEnum.ENT, NodeEnum.LEM ]:
                iri: str = f"{self.iri_base}entity/{node.key.lower().replace(' ', '_').replace('.', '_')}"  # pylint: disable=C0301
                subj: rdflib.URIRef = rdflib.URIRef(iri)
                ref_dict[node_id] = subj

                rdf_graph.add((
                    subj,
                    rdflib.SKOS.prefLabel,
                    rdflib.Literal(node.text, lang = lang),
                ))

                if node.kind == NodeEnum.ENT and node.annotated:
                    cls_obj: rdflib.URIRef = rdflib.URIRef(node.label)
                    cls_id: int = node_keys.index(node.label)  # type: ignore
                    ref_dict[cls_id] = cls_obj

                    rdf_graph.add((
                        subj,
                        rdflib.RDF.type,
                        cls_obj,
                    ))

            elif node.kind == NodeEnum.IRI:
                subj = rdflib.URIRef(node.key)
                ref_dict[node_id] = subj

                desc = rdflib.Literal(node.text, lang = "en")

                rdf_graph.add((
                    subj,
                    rdflib.SKOS.prefLabel,
                    desc,
                ))

        # extract relations as RDF
        for edge in self.edges.values():
            if edge.kind == RelEnum.INF:
                if edge.src_node in ref_dict:
                    subj = ref_dict.get(edge.src_node)
                else:
                    src_node: Node = node_list[edge.src_node]
                    subj = rdflib.URIRef(src_node.label)
                    ref_dict[edge.src_node] = subj

                if edge.dst_node in ref_dict:
                    obj: rdflib.URIRef = ref_dict.get(edge.dst_node)
                else:
                    dst_node: Node = node_list[edge.dst_node]
                    obj = rdflib.URIRef(dst_node.label)
                    ref_dict[edge.dst_node] = obj

                rdf_graph.add((
                    subj,
                    rdflib.URIRef(edge.rel),
                    obj,
                ))

        # serialize as RDF triples
        for prefix, iri in self.factory.kg.NS_PREFIX.items():
            rdf_graph.bind(prefix, rdflib.Namespace(iri))

        n3_str: str = rdf_graph.serialize(
            format = "n3",
            base = self.iri_base,
        )

        return n3_str
