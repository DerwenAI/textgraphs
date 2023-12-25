#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sample application to demo the `TextGraphs` library.

see copyright/license https://huggingface.co/spaces/DerwenAI/textgraphs/blob/main/README.md
"""

import asyncio
import sys  # pylint: disable=W0611
import traceback
import time

from icecream import ic  # pylint: disable=E0401
from pyinstrument import Profiler  # pylint: disable=E0401
import pandas as pd  # pylint: disable=E0401

import textgraphs


if __name__ == "__main__":
    SRC_TEXT: str = """
Werner Herzog is a remarkable filmmaker and an intellectual originally from Germany, the son of Dietrich Herzog.
After the war, Werner fled to America to become famous.
"""

    ## set up
    ## NB: profiler raises handler exceptions when `concur = False`
    debug: bool = False  # True
    concur: bool = True  # False
    profile: bool = True  # False

    if profile:
        profiler: Profiler = Profiler()
        profiler.start()

    try:
        start_time: float = time.time()

        tg: textgraphs.TextGraphs = textgraphs.TextGraphs(
            factory = textgraphs.PipelineFactory(
                spacy_model = textgraphs.SPACY_MODEL,
                ner = None, #textgraphs.NERSpanMarker(),
                kg = textgraphs.KGWikiMedia(
                    spotlight_api = textgraphs.DBPEDIA_SPOTLIGHT_API,
                    dbpedia_search_api = textgraphs.DBPEDIA_SEARCH_API,
                    dbpedia_sparql_api = textgraphs.DBPEDIA_SPARQL_API,
                    wikidata_api = textgraphs.WIKIDATA_API,
                ),
                infer_rels = [
                    textgraphs.InferRel_OpenNRE(
                        model = textgraphs.OPENNRE_MODEL,
                        max_skip = textgraphs.MAX_SKIP,
                        min_prob = textgraphs.OPENNRE_MIN_PROB,
                    ),
                    textgraphs.InferRel_Rebel(
                        lang = "en_XX",
                        mrebel_model = textgraphs.MREBEL_MODEL,
                    ),
                ],
            ),
        )

        duration: float = round(time.time() - start_time, 3)
        print(f"{duration:7.3f} sec: set up")


        ## NLP parse
        start_time = time.time()

        pipe: textgraphs.Pipeline = tg.create_pipeline(
            SRC_TEXT.strip(),
        )

        duration = round(time.time() - start_time, 3)
        print(f"{duration:7.3f} sec: parse text")


        ## collect graph elements from the parse
        start_time = time.time()

        tg.collect_graph_elements(
            pipe,
            debug = debug,
        )

        duration = round(time.time() - start_time, 3)
        print(f"{duration:7.3f} sec: collect elements")


        ## perform entity linking
        start_time = time.time()

        tg.perform_entity_linking(
            pipe,
            debug = debug,
        )

        duration = round(time.time() - start_time, 3)
        print(f"{duration:7.3f} sec: entity linking")


        ## perform concurrent relation extraction
        start_time = time.time()

        if concur:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            inferred_edges: list = loop.run_until_complete(
                tg.infer_relations_async(
                    pipe,
                    debug = debug,
                )
            )
        else:
            inferred_edges = tg.infer_relations(
                pipe,
                debug = debug,
            )

        duration = round(time.time() - start_time, 3)
        print(f"{duration:7.3f} sec: relation extraction")

        n_list: list = list(tg.nodes.values())

        df_rel: pd.DataFrame = pd.DataFrame.from_dict([
            {
                "src": n_list[edge.src_node].text,
                "dst": n_list[edge.dst_node].text,
                "rel": pipe.kg.normalize_prefix(edge.rel),
                "weight": edge.prob,
            }
            for edge in inferred_edges
        ])

        ic(df_rel)


        ## construct the _lemma graph_
        start_time = time.time()

        tg.construct_lemma_graph(
            debug = debug,
        )

        duration = round(time.time() - start_time, 3)
        print(f"{duration:7.3f} sec: construct graph")


        ## rank the extracted phrases
        start_time = time.time()

        tg.calc_phrase_ranks(
            pr_alpha = textgraphs.PAGERANK_ALPHA,
            debug = debug,
        )

        duration = round(time.time() - start_time, 3)
        print(f"{duration:7.3f} sec: rank phrases")


        ## show the results
        ic(tg.get_phrases_as_df())

        if debug:  # pylint: disable=W0101
            for key, node in tg.nodes.items():
                print(key, node)

            for key, edge in tg.edges.items():
                print(key, edge)

    except Exception as ex:  # pylint: disable=W0718
        ic(ex)
        traceback.print_exc()


    ## stack profiler report
    if profile:
        profiler.stop()
        profiler.print()


    ## EXPERIMENT
    #sys.exit(0)


    ## output lemma graph as JSON
    with open("lemma.json", "w", encoding = "utf-8") as fp:
        fp.write(tg.dump_lemma_graph())
