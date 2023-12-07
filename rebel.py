#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Babelscape `REBEL` demo from
<https://huggingface.co/spaces/Babelscape/mrebel-demo>
<https://github.com/Babelscape/rebel>
"""

from icecream import ic  # pylint: disable=E0401
from transformers import pipeline  # pylint: disable=E0401


REL_EXTRACT_PIPE: pipeline = pipeline(
    "translation_xx_to_yy",
    model = "Babelscape/mrebel-large",
    tokenizer = "Babelscape/mrebel-large",
)


def tokenize_sent (
    text: str,
    ) -> str:
    """
use the tokenizer manually since we need special tokens.

change `en_XX` to the language of the source
    """
    tokenized: list = REL_EXTRACT_PIPE(
        text,
        decoder_start_token_id = 250058,
        src_lang = "en_XX",
        tgt_lang = "<triplet>",
        return_tensors = True,
        return_text = False,
    )

    extracted: list = REL_EXTRACT_PIPE.tokenizer.batch_decode([
        tokenized[0]["translation_token_ids"]
    ])

    return extracted[0]


def extract_triplets_typed (
    text: str,
    ) -> list:
    """
parse the generated text and extract its triplets
    """
    triplets = []
    relation = ""
    current = "x"
    subject, relation, object_, object_type, subject_type = "","","","",""

    text = text.strip()\
               .replace("<s>", "")\
               .replace("<pad>", "")\
               .replace("</s>", "")\
               .replace("tp_XX", "")\
               .replace("__en__", "")

    for token in text.split():
        if token in [ "<triplet>", "<relation>" ]:
            current = "t"

            if relation != "":
                triplets.append({
                    "head": subject.strip(),
                    "head_type": subject_type,
                    "type": relation.strip(),
                    "tail": object_.strip(),
                    "tail_type": object_type,
                })

                relation = ""

            subject = ""

        elif token.startswith("<") and token.endswith(">"):
            if current in [ "t", "o" ]:
                current = "s"

                if relation != "":
                    triplets.append({
                        "head": subject.strip(),
                        "head_type": subject_type,
                        "type": relation.strip(),
                        "tail": object_.strip(),
                        "tail_type": object_type,
                    })

                object_ = ""
                subject_type = token[1:-1]
            else:
                current = "o"
                object_type = token[1:-1]
                relation = ""

        else:
            if current == "t":
                subject += " " + token
            elif current == "s":
                object_ += " " + token
            elif current == "o":
                relation += " " + token

    if subject != "" and relation != "" and object_ != "" and object_type != "" and subject_type != "":  # pylint: disable=C0301
        triplets.append({
            "head": subject.strip(),
            "head_type": subject_type,
            "tail": object_.strip(),
            "tail_type": object_type,
            "rel": relation.strip(),
        })

    return triplets


if __name__ == "__main__":
    para: list = [
        "Werner Herzog is a remarkable filmmaker and intellectual from Germany, the son of Dietrich Herzog.",  # pylint: disable=C0301
        "After the war, Werner fled to America to become famous.",
        "Instead, Herzog became President and decided to nuke Slovenia.",
    ]

    for sent in para:
        extract: str = tokenize_sent(sent.strip())
        ic(extract)

        triples: list = extract_triplets_typed(extract)
        ic(triples)
