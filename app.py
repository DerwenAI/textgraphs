#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=C0301

"""
HuggingFace Spaces demo of the `TextGraphs` library using Streamlit

see copyright/license https://huggingface.co/spaces/DerwenAI/textgraphs/blob/main/README.md
"""

import pathlib
import time
import typing

import matplotlib.pyplot as plt  # pylint: disable=E0401
import pandas as pd  # pylint: disable=E0401
import pyvis  # pylint: disable=E0401
import spacy  # pylint: disable=E0401
import streamlit as st  # pylint: disable=E0401

import textgraphs


if __name__ == "__main__":
    # default text input
    SRC_TEXT: str = """
Werner Herzog is a remarkable filmmaker and intellectual originally from Germany, the son of Dietrich Herzog.
    """

    # store the initial value of widgets in session state
    if "visibility" not in st.session_state:
        st.session_state.visibility = "visible"
        st.session_state.disabled = False

    with st.container():
        st.title("demo: TextGraphs + LLMs to construct a 'lemma graph'")
        st.markdown(
            "_TextGraphs_ library is intended for iterating through a sequence of paragraphs.",
        )

        blurb_1: pathlib.Path = pathlib.Path("docs/demo/blurb.1.html")

        st.markdown(
            blurb_1.read_text(encoding = "utf-8"),
            unsafe_allow_html = True,
        )

        # collect input + config
        text_input: str = st.text_area(
            "Source Text:",
            value = SRC_TEXT.strip(),
        )

        llm_ner = st.checkbox(
            "enhance spaCy NER using: SpanMarker",
            value = False,
        )

        link_ents = st.checkbox(
            "link entities using: DBPedia Spotlight, WikiMedia API",
            value = False,
        )

        infer_rel = st.checkbox(
            "infer relations using: REBEL, OpenNRE, qwikidata",
            value = False,
        )

        if text_input or llm_ner or link_ents or infer_rel:
            ## parse the document
            st.subheader("parse the raw text", divider = "rainbow")
            start_time: float = time.time()

            # generally it is fine to use factory defaults,
            # although let's illustrate these settings here
            infer_rels: list = []

            if infer_rel:
                with st.spinner(text = "load rel models..."):
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
                    ]

            ner: typing.Optional[ textgraphs.Component ] = None

            if llm_ner:
                ner = textgraphs.NERSpanMarker(
                    ner_model = textgraphs.NER_MODEL,
                )

            tg: textgraphs.TextGraphs = textgraphs.TextGraphs(
                factory = textgraphs.PipelineFactory(
                    spacy_model = textgraphs.SPACY_MODEL,
                    ner = ner,
                    kg = textgraphs.KGWikiMedia(
                        spotlight_api = textgraphs.DBPEDIA_SPOTLIGHT_API,
                        dbpedia_search_api = textgraphs.DBPEDIA_SEARCH_API,
                        dbpedia_sparql_api = textgraphs.DBPEDIA_SPARQL_API,
                        wikidata_api = textgraphs.WIKIDATA_API,
                        min_alias = textgraphs.DBPEDIA_MIN_ALIAS,
                        min_similarity = textgraphs.DBPEDIA_MIN_SIM,
                    ),
                    infer_rels = infer_rels,
                ),
            )

            duration: float = round(time.time() - start_time, 3)
            st.write(f"set up: {round(duration, 3)} sec")

            with st.spinner(text = "parse text..."):
                start_time = time.time()

                pipe: textgraphs.Pipeline = tg.create_pipeline(
                    text_input.strip(),
                )

            duration = round(time.time() - start_time, 3)
            st.write(f"parse text: {round(duration, 3)} sec, {len(text_input)} characters")

            # render the entity html
            ent_html: str = spacy.displacy.render(
                pipe.ner_doc,
                style = "ent",
                jupyter = False,
            )

            st.markdown(
                ent_html,
                unsafe_allow_html = True,
            )

            # generate dependencies as an SVG
            dep_svg = spacy.displacy.render(
                pipe.ner_doc,
                style = "dep",
                jupyter = False,
            )

            st.image(
                dep_svg,
                width = 800,
                use_column_width = "never",
            )


            ## collect graph elements from the parse
            st.subheader("construct the base level of the lemma graph", divider = "rainbow")
            start_time = time.time()

            tg.collect_graph_elements(
                pipe,
                debug = False,
            )

            duration = round(time.time() - start_time, 3)
            st.write(f"collect elements: {round(duration, 3)} sec, {len(tg.nodes)} nodes, {len(tg.edges)} edges")

            ## perform entity linking
            if link_ents:
                st.subheader("extract entities and perform entity linking", divider = "rainbow")

                with st.spinner(text = "entity linking..."):
                    start_time = time.time()

                    tg.perform_entity_linking(
                        pipe,
                        debug = False,
                    )

                duration = round(time.time() - start_time, 3)
                st.write(f"entity linking: {round(duration, 3)} sec")


            ## perform relation extraction
            if infer_rel:
                st.subheader("infer relations", divider = "rainbow")
                st.write("NB: this part runs an order of magnitude more *slooooooowly* on HF Spaces")

                with st.spinner(text = "relation extraction..."):
                    start_time = time.time()

                    # NB: run this iteratively since Streamlit on HF Spaces is *sloooooooooow*
                    inferred_edges: list = tg.infer_relations(
                        pipe,
                        debug = False,
                    )

                duration = round(time.time() - start_time, 3)

                n_list: list = list(tg.nodes.values())

                df_rel: pd.DataFrame = pd.DataFrame.from_dict([
                    {
                        "src": n_list[edge.src_node].text,
                        "dst": n_list[edge.dst_node].text,
                        "rel": edge.rel,
                        "weight": edge.prob,
                    }
                    for edge in inferred_edges
                ])

                st.dataframe(df_rel)
                st.write(f"relation extraction: {round(duration, 3)} sec, {len(df_rel)} edges")


            ## construct the _lemma graph_
            start_time = time.time()

            tg.construct_lemma_graph(
                debug = False,
            )

            duration = round(time.time() - start_time, 3)
            st.write(f"construct graph: {round(duration, 3)} sec")


            ## rank the extracted phrases
            st.subheader("rank the extracted phrases", divider = "rainbow")
            start_time = time.time()

            tg.calc_phrase_ranks(
                pr_alpha = textgraphs.PAGERANK_ALPHA,
                debug = False,
            )

            df_ent: pd.DataFrame = tg.get_phrases_as_df()

            duration = round(time.time() - start_time, 3)
            st.write(f"extract: {round(duration, 3)} sec, {len(df_ent)} entities")

            st.dataframe(df_ent)


            ## generate a word cloud
            st.subheader("generate a word cloud", divider = "rainbow")

            render: textgraphs.RenderPyVis = tg.create_render()
            wordcloud = render.generate_wordcloud()

            st.image(
                wordcloud.to_image(),
                width = 700,
                use_column_width = "never",
            )


            ## visualize the lemma graph
            st.subheader("visualize the lemma graph", divider = "rainbow")
            st.markdown(
                """
                what you get at this stage is a relatively noisy,
                low-level detailed graph of the parsed text

                the most interesting nodes will probably be either
                subjects (`nsubj`) or direct objects (`pobj`)
                """
            )

            pv_graph: pyvis.network.Network = render.render_lemma_graph(
                debug = False,
            )

            pv_graph.force_atlas_2based(
                gravity = -38,
                central_gravity = 0.01,
                spring_length = 231,
                spring_strength = 0.7,
                damping = 0.8,
                overlap = 0,
            )

            pv_graph.show_buttons(filter_ = [ "physics" ])
            pv_graph.toggle_physics(True)

            py_html: pathlib.Path = pathlib.Path("vis.html")
            pv_graph.save_graph(py_html.as_posix())

            st.components.v1.html(
                py_html.read_text(encoding = "utf-8"),
                height = render.HTML_HEIGHT_WITH_CONTROLS,
                scrolling = False,
            )


            ## cluster the communities
            st.subheader("cluster the communities", divider = "rainbow")
            st.markdown(
                """
<details>
  <summary><strong>About this clustering...</strong></summary>
  <p>
In the tutorial
<a href="https://towardsdatascience.com/how-to-convert-any-text-into-a-graph-of-concepts-110844f22a1a" target="_blank">"How to Convert Any Text Into a Graph of Concepts"</a>,
Rahul Nayak uses the
<a href="https://en.wikipedia.org/wiki/Girvan%E2%80%93Newman_algorithm"><em>girvan-newman</em></a>
algorithm to split the graph into communities, then clusters on those communities.
His approach works well for unsupervised clustering of key phrases which have been extracted from a collection of many documents.
  </p>
  <p>
While Nayak was working with entities extracted from "chunks" of text, not with a text graph per se, this approach is useful for identifying network motifs which can be condensed, e.g., to extract a semantic graph overlay as an <em>abstraction layer</em> atop a lemma graph.
  </p>
</details>
<br/>
                """,
                unsafe_allow_html = True,
            )

            spring_dist_val = st.slider(
                "spring distance for NetworkX clusters",
                min_value = 0.0,
                max_value = 10.0,
                value = 1.2,
            )

            if spring_dist_val:
                start_time = time.time()
                fig, ax = plt.subplots()

                comm_map: dict = render.draw_communities(
                    spring_distance = spring_dist_val,
                )

                st.pyplot(fig)

                duration = round(time.time() - start_time, 3)
                st.write(f"cluster: {round(duration, 3)} sec, {max(comm_map.values()) + 1} clusters")


            ## download lemma graph
            st.subheader("download the results", divider = "rainbow")
            st.markdown(
                """
Download a serialized <em>lemma graph</em> in
<a href="https://networkx.org/documentation/stable/reference/readwrite/generated/networkx.readwrite.json_graph.node_link_data.html" target="_blank"><em>node-link</em></a> format, suitable for import to Neo4j, NetworkX, KùzuDB, etc.
                """,
                unsafe_allow_html = True,
            )

            st.download_button(
                label = "download",
                data = tg.dump_lemma_graph(),
                file_name = "lemma_graph.json",
                mime = "application/json",
            )


            ## WIP
            st.divider()
            st.write("(WIP)")

            thanks: str = """
This demo has completed, and thank you for running a Derwen space!
            """

            st.toast(
                thanks,
                icon ="😍",
            )
