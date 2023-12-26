#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
`replicate` demo from
<https://github.com/replicate/replicate-python#readme>
"""

import typing

import replicate  # pylint: disable=E0401


if __name__ == "__main__":
    # load `Notus` model: <https://huggingface.co/argilla/notus-7b-v1>
    model: replicate.model.Model = replicate.models.get(
        "titocosta/notus-7b-v1",
    )

    version: replicate.version.Version = model.versions.get(
        "dbcd2277b32873525e618545e13e64c3ba121b681cbd2b5f0ee7f95325e7a395",
    )

    prompt: str = """
Sentence: {}
Extract RDF predicate from the sentence in this format:
SUBJECT:<subject>
PREDICATE:<predicate>
OBJECT:<object, optional>
    """

    text: str = """
Werner Herzog is a German film director, screenwriter, author, actor, and opera director, regarded as a pioneer of New German Cinema.
    """

    output: typing.Iterator[ str ] = replicate.run(
        version,
        input = {
            "prompt": prompt.format(text.strip()).strip(),
        },
    )

    for item in output:
        print(item)
