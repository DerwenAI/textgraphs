#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
`spaCy-entity-linker` demo from
<https://github.com/egerber/spaCy-entity-linker/issues/18>
"""

from icecream import ic  # pylint: disable=E0401
import spacy  # pylint: disable=E0401
import spacy_entity_linker as sel  # pylint: disable=E0401


def link_wikidata (
    doc: spacy.tokens.doc.Doc,
    ) -> None:
    """
Run an entity linking classifier for wikidata
    """
    classifier = sel.EntityClassifier.EntityClassifier()

    for ent in doc.ents:
        print()
        ic(ent.text, ent.label_)

        # build a term (a simple span) then identify all
        # the candidate entities for it
        term: sel.TermCandidate = sel.TermCandidate.TermCandidate(ent)

        candidates: sel.EntityCandidates.EntityCandidates = term.get_entity_candidates()
        ic(candidates)

        if len(candidates) > 0:
            # select the best candidate
            entity: sel.EntityElement.EntityElement = classifier(candidates)

            ic(entity.__dict__)
            ic(entity.get_sub_entities(limit=10))
            ic(entity.get_super_entities(limit=10))


if __name__ == "__main__":
    SRC_TEXT: str = """
Werner Herzog is a remarkable filmmaker and an intellectual originally from Germany, the son of Dietrich Herzog.
After the war, Werner fled to America to become famous.
"""

    # initialize language model
    nlp: spacy.Language = spacy.load("en_core_web_sm")
    sample_doc: spacy.tokens.doc.Doc = nlp(SRC_TEXT.strip())

    link_wikidata(sample_doc)
