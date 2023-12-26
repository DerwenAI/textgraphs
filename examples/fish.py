#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
`spacyfishing` entity linking to Wikidata
<https://github.com/Lucaterre/spacyfishing>
"""

from icecream import ic  # pylint: disable=E0401
import spacy  # pylint: disable=E0401


SRC_TEXT: str = """
Werner Herzog is a remarkable filmmaker and an intellectual originally from Germany, the son of Dietrich Herzog, although they never spoke after the war.
"""

nlp = spacy.load(
    "en_core_web_sm",
    exclude = [ "ner" ],
)

nlp.add_pipe(
    "span_marker",
    config = {
        "model": "tomaarsen/span-marker-roberta-large-ontonotes5",
    },
)

nlp.add_pipe(
    "entityfishing",
    config = {
        "api_ef_base": "https://cloud.science-miner.com/nerd/service",
        "extra_info": True,
        "filter_statements": [ ],
    },
)

nlp.add_pipe(
    "merge_entities",
)


doc = nlp(SRC_TEXT.strip())

for ent in doc.ents:
    ic(
        ent.text,
        ent.label_,
        ent._.nerd_score,
        ent._.url_wikidata,
        ent._.description,
        ent._.other_ids,
    )
