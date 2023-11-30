#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright ©2023 Derwen, Inc. All rights reserved.
# pylint: disable=C0301

"""
HuggingFace Spaces demo of `textgraph` using Streamlit
"""

import pathlib
import time

import matplotlib.pyplot as plt  # pylint: disable=E0401
import pandas as pd  # pylint: disable=E0401
import pyvis  # pylint: disable=E0401
import spacy  # pylint: disable=E0401
import streamlit as st  # pylint: disable=E0401

from textgraph import RenderPyVis, TextGraph


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

        llm_nre = st.checkbox(
            "use OpenNER for relation extraction",
            value = False,
        )

        if text_input or llm_ner or llm_nre:
            ## parse the document
            st.subheader("parse the raw text", divider = "rainbow")

            start_time: float = time.time()

            sample_doc: spacy.tokens.doc.Doc = tg.build_doc(
                text_input,
                ner_model = TextGraph.NER_MODEL if llm_ner else None,
            )

            duration: float = round(time.time() - start_time, 3)
            st.write(f"parse: {round(duration, 3)} sec, {len(text_input)} characters")

            # render the entity html
            ent_html: str = spacy.displacy.render(
                sample_doc,
                style = "ent",
                jupyter = False,
            )

            st.markdown(
                ent_html,
                unsafe_allow_html = True,
            )

            # generate dependencies as an SVG
            dep_svg = spacy.displacy.render(
                sample_doc,
                style = "dep",
                jupyter = False,
            )

            st.image(
                dep_svg,
                width = 800,
                use_column_width = "never",
            )


            ## build the lemma graph
            st.subheader("build the lemma graph and extract entities", divider = "rainbow")

            start_time = time.time()

            tg.build_graph_embeddings(
                sample_doc,
                debug = False,
            )


            # extract ranked entities from the document
            tg.calc_phrase_ranks(debug = False)
            df: pd.DataFrame = tg.get_phrases_as_df()

            duration = round(time.time() - start_time, 3)
            st.write(f"extract: {round(duration, 3)} sec, {len(df)} entities")

            st.dataframe(df)


            ## visualize the lemma graph
            st.subheader("visualize the lemma graph", divider = "rainbow")
            st.markdown(
                """
                what you get at this stage is a relatively noisy,
                low-level detailed graph of the parsed text

                the most interesting parts will probably be:

                  * the src node of an `nsubj` edge
                  * the dst node of a `pobj` edge
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

            fig, ax = plt.subplots()

            render.draw_communities(
                spring_distance = 1.2,
            )

            st.pyplot(fig)


            ## infer relations
            if llm_nre:
                tg.infer_relations(
                    SRC_TEXT.strip(),
                    debug = False,
                )


            ## WIP
            st.divider()
            st.write("(WIP)")

            st.toast(
                "Demo completed! Thank you for running a Derwen space",
                icon ="😍",
            )
