#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate the `apidocs` markdown needed for the package reference.
"""

import sys
import typing

import pyfixdoc


######################################################################
## main entry point

if __name__ == "__main__":
    # NB: `inspect` is picky about paths and current working directory
    # this only works if run from the top-level directory of the repo
    sys.path.insert(0, "../")

    # customize the following, per use case
    import textgraphs  # pylint: disable=W0611

    class_list: typing.List[ str ] = [
        "TextGraphs",
        "SimpleGraph",
        "Node",
        "NodeEnum",
        "Edge",
        "RelEnum",
        "PipelineFactory",
        "Pipeline",
        "Component",
        "NERSpanMarker",
        "NounChunk",
        "KnowledgeGraph",
        "KGSearchHit",
        "KGWikiMedia",
        "LinkedEntity",
        "InferRel",
        "InferRel_OpenNRE",
        "InferRel_Rebel",
        "RenderPyVis",
        "NodeStyle",
    ]

    pkg_doc: pyfixdoc.PackageDoc = pyfixdoc.PackageDoc(
        "textgraphs",
        "https://github.com/DerwenAI/textgraphs/blob/main",
        class_list,
    )

    # NB: uncomment to analyze/troubleshoot the results of `inspect`
    #pkg_doc.show_all_elements(); sys.exit(0)

    # build the apidocs markdown
    pkg_doc.build()

    # output the apidocs markdown
    ref_md_file: str = sys.argv[1]
    pkg_doc.write_markdown(ref_md_file)
