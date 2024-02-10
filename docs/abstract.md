# Introduction

**DRAFT** (WIP)

The primary goal of this project is to improve semi-automated KG construction from large collections of unstructured text sources, while leveraging feedback from domain experts and maintaining quality checks for the aggregated results.

Typical downstream use cases for these KGs include collecting data for industrial optimization use cases based on _operations research_, as mechanisms enabling structured LLM reasoning [#besta2024topo](biblio.md#besta2024topo), and potentially new methods of integrating KG linked data directly into LLM inference [#wen2023mindmap](biblio.md#wen2023mindmap)

To this point, this project explores hybrid applications which leverage LLMs to improve _natural language processing_ (NLP) pipeline components, which are also complemented by other deep learning models, graph queries, semantic inference, and related APIs.

Notably, LLMs come from NLP research.
Amidst an overwhelming avalanche of contemporary news headlines, pre-print papers, celebrity researchers, industry pundits, and so on ...
the hype begs a simple question: how good are LLMs at improving the results of natural language parsing and annotation in practice?

Granted, it is possible to use LLM chat interfaces to generate entire KGs from unstructured text sources.
Results from this brute-force approach tend to be mixed, especially when KGs rely on non-trivial controlled vocabularies and overlapping concepts.
For examples, see [#lawrence2024ttg](biblio.md#lawrence2024ttg) and [#nizami2023llm](biblio.md#nizami2023llm).

Issues with LLM accuracy (e.g., hallucinations) may be partially addressed through use of _retrieval augmented generation_ (RAG).
Even so, this approach tends to be expensive, especially when large number of PDF documents need to be used as input.
Use of a fully-automated "black box" based on a LLM chat agent in production use cases also tends to contradict the benefits of curating a KG to collect representations of an organization's domain expertise.

There are perhaps some deeper issues implied in this work.
To leverage "generative AI" for KGs, we must cross multiple boundaries of representation.
For example, graph ML approaches which start from graph-theoretic descriptions are losing vital information.
On the one hand, these are generally focused on _node prediction_ or _edge prediction_ tasks, which seems overly reductionist and simplistic in the context of trying to generate streams of _composable elements_ for building graphs.
On the other hand, these approaches typically get trained on _node embeddings_, _edge embeddings_, or _graph embeddings_ -- which may not quite fit the problem at hand.
Rolling back even further, the transition from NLP parsing of unstructured text sources to the construction of KGs also tends to throw away a lot of potentially useful annotations and context available from the NLP workflows.
Commonly accepted means for training LLMs from text sources directly often use tokenization which is relatively naïve about what might be structured within the data, other than linear sequences of characters.
Notably, this ignores the relationships among surface forms of text and their co-occurence with predicted entities or relations.
Some contemporary approaches to RAG use "chunked" text, attempting to link between chunks, even though this approach arguably destroys information about what is structured within that input data.
These multiple disconnects between the source data, the representation methods used in training models, and the tactics employed for applications; however, quite arguably the "applications" targeted in research projects generally stop at comparisons of benchmarks.
Overall, these disconnects indicate the need for rethinking the problem at multiple points.

For industry uses of KGs, one frequent observation from those leading production projects is that the "last mile" of applications generally relies on _operations research_, not ML.
We must keep these needs in mind when applying "generative AI" approaches to industry use cases.
Are we developing representations which can subsequently be leveraged for dynamic programming, convex optimization, etc.?

This project explores a different definition for "generative AI" in the context of working with KGs for production use cases.
Rather than pursue an LLM to perform all required tasks, is it possible to combine the use of smaller, more specialized models for specific tasks within the reasonably well-understood process of KG construction?
In broad strokes, can this work alternative provide counterfactuals to the contemporary trends for chat-based _prompt engineering_?

Seeking to integrate results from several other research projects implies substantial amounts of code reuse.
It would be intractable in terms of time and funding to rewrite code and then re-evaluate models for the many research projects which are within the scope of this work.
Therefore reproducibilty of published results -- based on open source code, models, evals, etc. -- becomes a crucial factor for determining whether others projects are suitable to be adapted into KG workflows.

For the sake of brevity, we do not define all of the terminology used, instead relying on broadly used terms in the literature. 
