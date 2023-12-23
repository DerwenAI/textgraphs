#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
These classes provide wrappers for _relation extraction_ models:

  * ThuNLP `OpenNRE`
  * Babelscape `REBEL`

see copyright/license https://huggingface.co/spaces/DerwenAI/textgraphs/blob/main/README.md
"""

import typing

from icecream import ic  # pylint: disable=E0401
import networkx as nx  # pylint: disable=E0401
import opennre  # pylint: disable=E0401
import transformers  # pylint: disable=E0401

from .defaults import MAX_SKIP, MREBEL_MODEL, OPENNRE_MIN_PROB, OPENNRE_MODEL
from .elem import Node
from .pipe import InferRel, Pipeline


######################################################################
## class definitions

class InferRel_OpenNRE (InferRel):  # pylint: disable=C0103,R0903
    """
Perform relation extraction based on the `OpenNRE` model.
<https://github.com/thunlp/OpenNRE>
    """
    def __init__ (
        self,
        *,
        model: str = OPENNRE_MODEL,
        max_skip: int = MAX_SKIP,
        min_prob: float = OPENNRE_MIN_PROB,
        ) -> None:
        """
Constructor.
        """
        self.max_skip: int = max_skip
        self.min_prob: float = min_prob

        self.nre_pipeline: opennre.model.softmax_nn.SoftmaxNN = opennre.get_model(model)


    def gen_triples (
        self,
        pipe: Pipeline,
        *,
        debug: bool = False,
        ) -> typing.Iterator[typing.Tuple[ Node, str, Node ]]:
        """
Iterate on entity pairs to drive `OpenNRE`, inferring relations
represented as triples which get produced by a generator.
        """
        node_list: list = [
            node.node_id
            for node in pipe.tokens
        ]

        pipe_graph: nx.MultiGraph = nx.MultiGraph()
        pipe_graph.add_nodes_from(node_list)

        pipe_graph.add_edges_from([
            ( edge.src_node, edge.dst_node, )
            for edge in pipe.edges
            if edge is not None and edge.src_node in node_list and edge.dst_node in node_list
        ])

        for src, dst in pipe.iter_entity_pairs(pipe_graph, self.max_skip, debug = debug):
            rel, prob = self.nre_pipeline.infer({  # type: ignore
                "text": pipe.text,
                "h": { "pos": src.get_pos() },
                "t": { "pos": dst.get_pos() },
            })

            if prob >= self.min_prob:
                if debug:
                    ic(src.text, dst.text)
                    ic(rel, prob)

                # use the knowledge graph to resolve the IRI
                iri: typing.Optional[ str ] = pipe.kg.resolve_rel_iri(
                    rel,
                )

                if iri is None:
                    iri = "opennre:" + rel.replace(" ", "_")

                yield src, iri, dst


class InferRel_Rebel (InferRel):  # pylint: disable=C0103,R0903
    """
Perform relation extraction based on the `REBEL` model.
<https://github.com/Babelscape/rebel>
<https://huggingface.co/spaces/Babelscape/mrebel-demo>
    """

    def __init__ (
        self,
        *,
        lang: str = "en_XX",
        mrebel_model: str = MREBEL_MODEL,
        ) -> None:
        """
Constructor.
        """
        self.lang = lang

        self.hf_pipeline: transformers.pipeline = transformers.pipeline(
            "translation_xx_to_yy",
            model = mrebel_model,
            tokenizer = mrebel_model,
        )


    def tokenize_sent (
        self,
        text: str,
        ) -> str:
        """
Apply the tokenizer manually, since we need to extract special tokens.
        """
        tokenized: list = self.hf_pipeline(
            text,
            decoder_start_token_id = 250058,
            src_lang = self.lang,
            tgt_lang = "<triplet>",
            return_tensors = True,
            return_text = False,
        )

        extracted: list = self.hf_pipeline.tokenizer.batch_decode([
            tokenized[0]["translation_token_ids"]
        ])

        return extracted[0]


    def extract_triplets_typed (
        self,
        text: str,
        ) -> list:
        """
Parse the generated text and extract its triplets.
        """
        triplets: list = []
        current: str = "x"
        subject: str = ""
        subject_type: str = ""
        relation: str = ""
        object_: str = ""
        object_type: str = ""

        text = text.strip()\
                   .replace("<s>", "")\
                   .replace("<pad>", "")\
                   .replace("</s>", "")\
                   .replace("tp_XX", "")\
                   .replace("__en__", "")

        for token in text.split():
            if token in [ "<triplet>", "<relation>" ]:
                current = "t"

                if relation != "":
                    triplets.append({
                        "head": subject.strip(),
                        "head_type": subject_type,
                        "type": relation.strip(),
                        "tail": object_.strip(),
                        "tail_type": object_type,
                })

                    relation = ""

                subject = ""

            elif token.startswith("<") and token.endswith(">"):
                if current in [ "t", "o" ]:
                    current = "s"

                    if relation != "":
                        triplets.append({
                            "head": subject.strip(),
                            "head_type": subject_type,
                            "type": relation.strip(),
                            "tail": object_.strip(),
                            "tail_type": object_type,
                        })

                    object_ = ""
                    subject_type = token[1:-1]
                else:
                    current = "o"
                    object_type = token[1:-1]
                    relation = ""

            else:
                if current == "t":
                    subject += " " + token
                elif current == "s":
                    object_ += " " + token
                elif current == "o":
                    relation += " " + token

        if subject != "" and relation != "" and object_ != "" and object_type != "" and subject_type != "":  # pylint: disable=C0301
            triplets.append({
                "head": subject.strip(),
                "head_type": subject_type,
                "tail": object_.strip(),
                "tail_type": object_type,
                "rel": relation.strip(),
            })

        return triplets


    def gen_triples (
        self,
        pipe: Pipeline,
        *,
        debug: bool = False,
        ) -> typing.Iterator[typing.Tuple[ Node, str, Node ]]:
        """
Drive `REBEL` to infer relations for each sentence, represented as
triples which get produced by a generator.
        """
        for sent in pipe.ner_doc.sents:
            extract: str = self.tokenize_sent(str(sent).strip())
            triples: typing.List[ dict ] = self.extract_triplets_typed(extract)

            tok_map: dict = {
                token.text: pipe.tokens[token.i]
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

                    # use the knowledge graph to resolve the IRI
                    iri: typing.Optional[ str ] = pipe.kg.resolve_rel_iri(
                        rel,
                    )

                    if iri is None:
                        iri = "mrebel:" + rel.replace(" ", "_")

                    yield src, iri, dst


if __name__ == "__main__":
    _rebel: InferRel_Rebel = InferRel_Rebel()

    _para: list = [
        "Werner Herzog is a remarkable filmmaker and intellectual from Germany, the son of Dietrich Herzog.",  # pylint: disable=C0301
        "After the war, Werner fled to America to become famous.",
        "Instead, Herzog became President and decided to nuke Slovenia.",
    ]

    for _sent in _para:
        _extract: str = _rebel.tokenize_sent(_sent.strip())
        ic(_extract)

        _triples: list = _rebel.extract_triplets_typed(_extract)
        ic(_triples)
