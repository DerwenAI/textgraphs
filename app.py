#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright ©2023 Derwen, Inc. All rights reserved.

"""
HuggingFace Spaces demo of `textgraph` using Streamlit
"""

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
        intro_html: str = """
<h5>demo: TextGraph + LLMs</h5>
<details>
<summary><strong>details</strong></summary>
<p>
Use <code>spaCy</code> + <code>SpanMarkerNER</code> to construct a
<em>lemma graph</em>, as a prelude to inferring the nodes, edges,
properties, and probabilities for building a knowledge graph from
a raw unstructured text source.
</p>
<ol>
<li>use <code>spaCy</code> to parse a document, with <code>SpanMarkerNER</code> LLM assist</li>
<li>build a _lemma graph_ in <code>NetworkX</code> from the parse results</li>
<li>run a modified <code>textrank</code> algorithm plus graph analytics</li>
<li>use <code>OpenNRE</code> to infer relations among entities</li>
<li>approximate a pareto archive (hypervolume) to re-rank extracted entities</li>
<li>visualize the interactive graph in <code>PyVis</code></li>
<li>apply topological transforms to enhance embeddings (in progress)</li>
<li>run graph representation learning on the <em>graph of relations</em> (in progress)</li>
</ol>
<p>
...
</p>
<ol start="9">
<li>PROFIT!</li>
</ol>
</details>
<hr/>
        """

        st.markdown(
            intro_html,
            unsafe_allow_html = True,
        )

        text_input: str = st.text_area(
            "Source Text:",
            value = SRC_TEXT.strip(),
        )

        if text_input:
            # parse the document
            start_time: float = time.time()

            sample_doc: spacy.tokens.doc.Doc = tg.build_doc(
                text_input,
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
