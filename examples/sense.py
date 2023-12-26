#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
`sense2vec` demo from
<https://github.com/explosion/sense2vec>
"""

from icecream import ic  # pylint: disable=E0401
import spacy  # pylint: disable=E0401

if __name__ == "__main__":
    nlp = spacy.load("en_core_web_sm")
    s2v = nlp.add_pipe("sense2vec")
    s2v.from_disk("./s2v_old")

    text: str = """
A sentence about natural language, AI, and NLP.
    """

    doc = nlp(text.strip())

    for ent in doc.ents:
        ic(ent)

        try:
            for lemma_tuple, prob in ent._.s2v_most_similar(3):
                ic(lemma_tuple, prob)

            freq = ent._.s2v_freq
            ic(freq)
        except ValueError as ex:
            ic(ex)
