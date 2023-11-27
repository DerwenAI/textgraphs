#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright ©2023 Derwen, Inc. All rights reserved.

"""
HuggingFace Spaces demo of `textgraph` using Streamlit
"""

import pathlib
import time

import pandas as pd  # pylint: disable=E0401
import spacy  # pylint: disable=E0401
import streamlit as st  # pylint: disable=E0401

from textgraph import TextGraph


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
        st.markdown(
            "<h2>demo: TextGraph + LLMs</h2>",
            unsafe_allow_html = True,
        )

        blurb_1: pathlib.Path = pathlib.Path("docs/demo/blurb.1.html")

        st.markdown(
            blurb_1.read_text(encoding = "utf-8"),
            unsafe_allow_html = True,
        )

        text_input: str = st.text_area(
            "Source Text:",
            value = SRC_TEXT.strip(),
        )

        llm_ner = st.checkbox(
            "use SpanMarker to enhance spaCy NER",
            value = True,
        )

        llm_nre = st.checkbox(
            "use OpenNER for relation extraction",
            value = True,
        )

        if text_input or llm_ner or llm_nre:
            # parse the document
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

            # build the lemma graph, then extract
            # ranked entities from the document
            start_time = time.time()

            tg.build_graph_embeddings(
                sample_doc,
                debug = False,
            )

            if llm_nre:
                tg.infer_relations(
                    SRC_TEXT.strip(),
                    debug = False,
                )

            tg.calc_phrase_ranks()
            df: pd.DataFrame = tg.get_phrases_as_df()

            duration = round(time.time() - start_time, 3)
            st.write(f"extract: {round(duration, 3)} sec, {len(df)} entities")

            st.dataframe(df)

            st.write("(WIP)")
