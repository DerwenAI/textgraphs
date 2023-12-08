#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=C0301

"""
HuggingFace Spaces demo of `textgraph` using Streamlit

see copyright/license https://huggingface.co/spaces/DerwenAI/textgraph/blob/main/README.md
"""

import pathlib
import time

import matplotlib.pyplot as plt  # pylint: disable=E0401
import pandas as pd  # pylint: disable=E0401
import pyvis  # pylint: disable=E0401
import spacy  # pylint: disable=E0401
import streamlit as st  # pylint: disable=E0401

from textgraph import Pipeline, PipelineFactory, RenderPyVis, TextGraph


if __name__ == "__main__":
    tg: TextGraph = TextGraph()

    # sample text
    SRC_TEXT: str = """
Werner Herzog is a remarkable filmmaker and intellectual originally from Germany, the son of Dietrich Herzog.
    """

    # store the initial value of widgets in session state
    if "visibility" not in st.session_state:
        st.session_state.visibility = "visible"
        st.session_state.disabled = False

    with st.container():
        st.title("demo: TextGraph + LLMs")
        st.markdown(
            "the _TextGraph_ library is intended for processing a stream of paragraphs",
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
            "use SpanMarker to enhance spaCy NER",
            value = False,
        )

        infer_rel = st.checkbox(
            "use REBEL and OpenNRE to infer relations",
            value = False,
        )

        if text_input or llm_ner or infer_rel:
            ## parse the document
            st.subheader("parse the raw text", divider = "rainbow")
            start_time: float = time.time()

            fabrica: PipelineFactory = PipelineFactory(
                dbpedia_api = PipelineFactory.DBPEDIA_API,
                ner_model = PipelineFactory.NER_MODEL if llm_ner else None,
            )

            pipe: Pipeline = fabrica.build_pipeline(
                text_input.strip(),
            )

            duration: float = round(time.time() - start_time, 3)
            st.write(f"parse: {round(duration, 3)} sec, {len(text_input)} characters")

            # render the entity html
            ent_html: str = spacy.displacy.render(
                pipe.ent_doc,
                style = "ent",
                jupyter = False,
            )

            st.markdown(
                ent_html,
                unsafe_allow_html = True,
            )

            # generate dependencies as an SVG
            dep_svg = spacy.displacy.render(
                pipe.ent_doc,
                style = "dep",
                jupyter = False,
            )

            st.image(
                dep_svg,
                width = 800,
                use_column_width = "never",
            )


            ## build the lemma graph
            st.subheader("build the lemma graph, extracting and linking entities", divider = "rainbow")
            start_time = time.time()

            tg.build_graph_embeddings(
                pipe,
                debug = False,
            )

            duration = round(time.time() - start_time, 3)
            st.write(f"lemma graph: {round(duration, 3)} sec, {len(tg.nodes)} nodes, {len(tg.edges)} edges")


            ## infer relations
            if infer_rel:
                st.subheader("infer relations", divider = "rainbow")
                start_time = time.time()

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
                st.write(f"infer rel: {round(duration, 3)} sec, {len(df_rel)} edges")


            ## rank the extracted entities
            st.subheader("rank the extracted entities", divider = "rainbow")
            start_time = time.time()

            tg.calc_phrase_ranks(debug = False)
            df_ent: pd.DataFrame = tg.get_phrases_as_df()

            duration = round(time.time() - start_time, 3)
            st.write(f"extract: {round(duration, 3)} sec, {len(df_ent)} entities")

            st.dataframe(df_ent)


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

            render: RenderPyVis = RenderPyVis(
                tg.nodes,
                tg.edges,
                tg.lemma_graph,
            )

            pv_graph: pyvis.network.Network = render.build_lemma_graph(
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
                height = 1300,
                scrolling = True,
            )


            ## cluster the communities
            st.subheader("cluster the communities", divider = "rainbow")
            st.markdown(
                """
In the tutorial
<a href="https://towardsdatascience.com/how-to-convert-any-text-into-a-graph-of-concepts-110844f22a1a" target="_blank">"How to Convert Any Text Into a Graph of Concepts"</a>,
Rahul Nayak uses the
<a href="https://en.wikipedia.org/wiki/Girvan%E2%80%93Newman_algorithm"><em>girvan-newman</em></a>
algorithm to split the graph into communities, then clusters on those communities.
His approach works well for unsupervised clustering of key phrases which have been extracted from a collection of many documents.
In contrast, Nayak was working with entities extracted from "chunks" of text, not with a text graph.
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
            st.subheader("download the lemma graph", divider = "rainbow")
            st.markdown(
                """
Download a serialized <em>lemma graph</em> in
<a href="https://networkx.org/documentation/stable/reference/readwrite/generated/networkx.readwrite.json_graph.node_link_data.html" target="_blank"><em>node-link</em></a> format.
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
