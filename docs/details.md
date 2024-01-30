This project Implements an LLM-augmented `textgraph` algorithm for
constructing a _lemma graph_ from raw, unstructured text source.

The `TextGraphs` library is based on work developed by
[Derwen](https://derwen.ai/graph)
in 2023 Q2 for customer apps and used in our `Cysoni`
product.

This library integrates code from:

  * [`SpanMarker`](https://github.com/tomaarsen/SpanMarkerNER/)
  * [`spaCy-DBpedia-Spotlight`](https://github.com/MartinoMensio/spacy-dbpedia-spotlight)
  * [`REBEL`](https://github.com/Babelscape/rebel)
  * [`OpenNRE`](https://github.com/thunlp/OpenNRE/)
  * [`qwikidata`](https://github.com/kensho-technologies/qwikidata)
  * [`pulp`](https://github.com/coin-or/pulp)
  * [`spaCy`](https://spacy.io/)
  * [`HF transformers`](https://huggingface.co/docs/transformers/index)
  * [`PyTextRank`](https://github.com/DerwenAI/pytextrank/)


For more background about early efforts which led to this line of inquiry, see the recent talks:

  * ["Language, Graphs, and AI in Industry"](https://derwen.ai/s/mqqm)
  **Paco Nathan**, K1st World (2023-10-11)  ([video](https://derwen.ai/s/4h2kswhrm3gc))
  * ["Language Tools for Creators"](https://derwen.ai/s/rhvg)
  **Paco Nathan**, FOSSY (2023-07-13)


The `TextGraphs` library shows integrations of several of these kinds
of components, complemented with use of graph queries, graph algorithms,
and other related tooling.
Admittedly, the results present a "hybrid" approach:
it's not purely "generative" -- whatever that might mean.

A core principle here is to provide results from the natural language
workflows which may be used for expert feedback.
In other words, how can we support means for leveraging
_human-in-the-loop_ (HITL) process?

Another principle has been to create a Python library built to produced
configurable, extensible pipelines.
Care has been given to writing code that can be run concurrently
(e.g., leveraging `asyncio`), using dependencies which have
business-friendly licenses, and paying attention to security concerns.

The library provides three main affordances for AI applications:

  1. With the default settings, one can use `TextGraphs` to extracti ranked key phrases from raw text -- even without using any of the additional deep learning models.

  2. Going a few further steps, one can generate an RDF or LPG graph from raw texts, and make use of _entity linking_, _relation extraction_, and other techniques to ground the natural language parsing by leveraging some knowledge graph which represents a particular domain. Default examples use WikiMedia graphs: DBPedia, Wikidata, etc.

  3. A third set of goals for `TextGraphs` is to provide a "playground" or "gym" for evaluating _graph levels of detail_, i.e., abstraction layers for knowledge graphs, and explore some the emerging work to produced _foundation models_ for knowledge graphs through topological transforms.

Regarding the third point, consider how language parsing produces
graphs by definition, although NLP results tend to be quite _noisy_.
The annotations inferred by NLP pipelines often get thrown out.
This seemed like a good opportunity to generate sample data for
"condensing" graphs into more abstracted representations.
In other words, patterns within the relatively noisy parse results
can be condensed into relatively refined knowledge graph elements.

Note that while the `spaCy` library for NLP plays a central role, the
`TextGraphs` library is not intended to become a `spaCy` pipeline.
