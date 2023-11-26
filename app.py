#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright ©2023 Derwen, Inc. All rights reserved.

"""
HuggingFace Spaces demo of `textgraph` using Streamlit
"""

import time

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

    col1, col2 = st.columns(2)

    with col1:
        text_input: str = st.text_area(
            "Source Text:",
            value = SRC_TEXT.strip(),
        )

        if text_input:
            # parse the document
            start_time: float = time.time()

            sample_doc: spacy.tokens.doc.Doc = tg.build_doc(
                text_input,
                use_llm = False,
            )

            duration: float = round(time.time() - start_time, 3)
            st.write(f"parse: {duration} sec, {len(text_input)} char")

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

            tg.infer_relations(
                SRC_TEXT.strip(),
                debug = False,
            )

            tg.calc_phrase_ranks()

            duration = round(time.time() - start_time, 3)
            st.write(f"extract: {duration} sec")

            st.dataframe(tg.get_phrases_as_df())
            #df.style.highlight_max(axis=0)
