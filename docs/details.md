# Technical Details

Implementation of an LLM-augmented `textgraph` algorithm for
constructing a _lemma graph_ from raw, unstructured text source.

The `TextGraphs` library is based on work developed by
[Derwen](https://derwen.ai/graph)
in 2023 Q2 for customer apps and used in our `Cysoni`
product.
This library integrates code from:

  * [`SpanMarkerNER`](https://github.com/tomaarsen/SpanMarkerNER/)
  * [`spaCy-DBpedia-Spotlight`](https://github.com/MartinoMensio/spacy-dbpedia-spotlight)
  * [`REBEL`](https://github.com/Babelscape/rebel)
  * [`OpenNRE`](https://github.com/thunlp/OpenNRE/)
  * [`qwikidata`](https://github.com/kensho-technologies/qwikidata)
  * [`pulp`](https://github.com/coin-or/pulp)
  * [`spaCy`](https://spacy.io/)
  * [`HF transformers`](https://huggingface.co/docs/transformers/index)
  * [`PyTextRank`](https://github.com/DerwenAI/pytextrank/)


For more details about this approach, see these talks:

  * ["Language, Graphs, and AI in Industry"](https://derwen.ai/s/mqqm)
  **Paco Nathan**, K1st World (2023-10-11)
  * ["Language Tools for Creators"](https://derwen.ai/s/rhvg)
  **Paco Nathan**, FOSSY (2023-07-13)


Other good tutorials (during 2023) which include related material:

  * ["Natural Intelligence is All You Need™"](https://youtu.be/C9p7suS-NGk?si=7Ohq3BV654ia2Im4)
  **Vincent Warmerdam**, PyData Amsterdam (2023-09-15)
  * ["How to Convert Any Text Into a Graph of Concepts"](https://towardsdatascience.com/how-to-convert-any-text-into-a-graph-of-concepts-110844f22a1a)
  **Rahul Nayak**, _Towards Data Science_ (2023-11-09)
  * ["Extracting Relation from Sentence using LLM"](https://medium.com/@nizami_muhammad/extracting-relation-from-sentence-using-llm-597d0c0310a8)
  **Muhammad Nizami** _Medium_ (2023-11-15)
