# Methods

  * use `spaCy` to parse a document, augmented by `SpanMarker` use of LLMs for NER
  * add noun chunks in parallel to entities, as "candidate" phrases for subsequent HITL confirmation
  * perform _entity linking_: `spaCy-DBpedia-Spotlight`, `WikiMedia API`, etc.
  * infer relations, plus graph inference: `REBEL`, `OpenNRE`, `qwikidata`, etc.
  * build a _lemma graph_ in `NetworkX` from the parse results
  * run a modified `textrank` algorithm plus graph analytics
  * approximate a _pareto archive_ (hypervolume) to re-rank extracted entities with `pulp`
  * visualize the _lemma graph_ interactively in ` PyVis`
  * cluster communities within the _lemma graph_
  * apply topological transforms to enhance graph ML and embeddings
  * build ML models based on the _graph of relations_ (in progress)


## Technical Approach

Construct a _lemma graph_, then perform _entity linking_ based on:
`spaCy`, `transformers`, `SpanMarkerNER`,
`spaCy-DBpedia-Spotlight`, `REBEL`, `OpenNRE`,
`qwikidata`, `pulp`


In other words, this hybrid approach integrates
_NLP parsing_, _LLMs_, _graph algorithms_, _semantic inference_,
_operations research_, and also provides UX affordances for including
_human-in-the-loop_ practices.

The demo app and the Hugging Face space both illustrate a relatively
small problem, although they address a much broader class of AI problems
in industry.

This step is a prelude before leveraging
_topological transforms_, _large language models_, _graph representation learning_,
plus _human-in-the-loop_ domain expertise to infer
the nodes, edges, properties, and probabilities needed for the
semi-automated construction of _knowledge graphs_ from
raw unstructured text sources.

In addition to providing a library for production use cases,
`TextGraphs` creates a "playground" or "gym"
in which to prototype and evaluate abstractions based on
["Graph Levels Of Detail"](https://blog.derwen.ai/graph-levels-of-detail-ea4226abba55)

  1. use `spaCy` to parse a document, augmented by `SpanMarker` use of LLMs for NER
  1. add noun chunks in parallel to entities, as "candidate" phrases for subsequent HITL confirmation
  1. perform _entity linking_: `spaCy-DBpedia-Spotlight`, `WikiMedia API`, etc.
  1. infer relations, plus graph inference: `REBEL`, `OpenNRE`, `qwikidata`, etc.
  1. build a _lemma graph_ in `NetworkX` from the parse results
  1. run a modified `textrank` algorithm plus graph analytics
  1. approximate a _pareto archive_ (hypervolume) to re-rank extracted entities with `pulp`
  1. visualize the _lemma graph_ interactively in `PyVis`
  1. cluster communities within the _lemma graph_
  1. apply topological transforms to enhance graph ML and embeddings
  1. build ML models based on the _graph of relations_ (in progress)

**...**

  23. PROFIT!
