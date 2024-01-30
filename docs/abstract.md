# Introduction

The primary goal of this project is to improve semi-automated KG construction from large collections of unstructured text sources, while leveraging feedback from domain experts and maintaining quality checks for the aggregated results.

Typical downstream use cases for these KGs include collecting data for industrial optimization use cases based on _operations research_, as mechanisms enabling structured LLM reasoning [#besta2024topo](biblio.md#besta2024topo), and potentially new methods of integrating KG linked data directly into LLM inference [#wen2023mindmap](biblio.md#wen2023mindmap)

To this point, this project explores hybrid applications which leverage LLMs to improve _natural language processing_ (NLP) pipeline components, which are also complemented by other deep learning models, graph queries, semantic inference, and related APIs.

Notably, LLMs come from NLP research.
Amidst an overwhelming avalanche of contemporary news headlines, pre-print papers, celebrity researchers, industry pundits, and so on ...
the hype begs a simple question: how good are LLMs at improving the results of natural language parsing and annotation in practice?

Granted, it is possible to use LLM chat interfaces to generate entire KGs from unstructured text sources.
Results from this brute-force approach tend to be mixed, especially when KGs rely on non-trivial controlled vocabularies and overlapping concepts.
For example an [example](https://medium.com/@peter.lawrence_47665/text-to-graph-via-llm-pre-training-prompting-or-tuning-3233d1165360).

Issues with LLM accuracy (halucination) may be partially addressed through use of _retrieval augmented generation_ (RAG).
Even so, this approach tends to be expensive, especially when large number of PDF documents need to be used as input.
Moreover, a fully-automated "black box" based on a LLM chat agent runs counter to curating a KG to represent an organization's domain expertise.

This project explores a different definition for "generative AI" in the context of working with KGs.
Rather than pursue an LLM to perform all required tasks, is it possible to combine the use of smaller, more specialized models for specific tasks within the reasonably well-understood process of KG construction?
In broad strokes, can this alternative provide counterfactuals to the contemporary trends for chat-based _prompt engineering_?

Seeking to integrate results from several other research projects implies substantial amounts of code reuse.
It would be intractable in terms of time and funding to rewrite code and then re-evaluate models for the many research projects which are within the scope of this work.
Therefore reproducibilty of published results -- based on open source code, models, evals, etc. -- becomes a crucial factor for determining whether others projects are suitable to be adapted into KG workflows.

For the sake of brevity, we do not define all of the terminology used, instead relying on broadly used terms in the literature. 
