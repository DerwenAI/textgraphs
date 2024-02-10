Other projects have investigated related lines of inquiry, which help frame the problems encountered.

[#loganlpgs19](biblio.md#loganlpgs19),

  - primary goal is to generate entities and facts from a KG
  - emphasis on handling rare facts from a broad domain of topics and on improving perplexity
  - "we are interested in LMs that dynamically decide the facts to incorporate from the KG, guided by the discourse"
  - con: uses relatively simple `G = V,E` graph-theoretic notions of graph data, which is ostensibly RDF
  - "traditional LMs are only capable of remembering facts seen at training time, and often have difficulty recalling them"
  - introducing KGLM: enables the model to render information it has never seen before, as well as generate out-of-vocabulary tokens
  - generates conditional probability of mapping an entity to a parsed token, based on previous tokens and entities within the same stream
  - maintains a dynamically growing local KG, a subset of the KG that contains entities that have already been mentioned in the text, and their related entities
  - "one of the primary barriers to incorporating factual knowledge into LMs is that training data is hard to obtain"
  - provides the `Linked WikiText-2` dataset for running benchmarks, available on GitHub
  - "For most LMs, it is difficult to control their generation since factual knowledge is entangled with generation capabilities of the model"

> Standard language modeling corpora consist only of text, and thus are unable to describe which entities or facts each token is referring to. In contrast, while relation extraction datasets link text to a knowledge graph, the text is made up of disjoint sentences that do not provide sufficient context to train a powerful language model.


[#warmerdam2023pydata](biblio.md#warmerdam2023pydata),  20:35-ff

  - using `spaCy` to parse and annotate tokens with metadata
  - parse trees => graph => heuristics to map from phrases to concepts
  - `sense2vec` to find neighborhoods for surface forms (acronyms, synonyms, etc.)
  - UMAP, etc. => hinting toward: "descriptive but not computable"
  - UX: active learning vs. annotations of wrong examples using `prodigy`
      - "spend more effort per example" => coining term _active teaching_
  - rethinking beyond the "optimality trap"
  - "maybe familiarity is a liability in data analytics?" => doubt can be an advantage


[#wen2023mindmap](biblio.md#wen2023mindmap),

  - how to prompt LLMs with KGs
  - "build a prompting pipeline that endows LLMs with the capability of comprehending KG inputs and inferring with a combined implicit knowledge and the retrieved external knowledge"
  - in contrast, the _prompt engineering_ paradigm: "pre-train, prompt, and predict"
  - "goal of this work is to build a plug-and-play prompting approach to elicit the graph-of-thoughts reasoning capability in LLMs"
      1.consolidates the retrieved facts from KGs and the implicit knowledge from LLMs
      2. discovers new patterns in input KGs
      3. reasons over the mind map to yield final outputs
  - build multiple _evidence sub-graphs_ which get aggregated into _reasoning graphs_, then prompt LLMs and build a _mind map_ to explain the reasoning process
  - conjecture that LLMs can comprehend and extract knowledge from a reasoning graph that is described by natural language
  - prompting a GPT-3.5 with `MindMap` yields an overwhelming performance over GPT-4 consistently


[#tripathi2024deepnlp](biblio.md#tripathi2024deepnlp),

["Deep NLP on SF Literature"](https://github.com/kkrishna24/deep_nlp_on_sf_literature)
**Krishna Tripathi** _GitHub_ (2024-01-25)

  - processes texts using customized methods, NLTK, and spaCy
  - performs domain-specific named entity recognition in multiple stages
  - fine-tunes a RoBERTa model using GPT to generate annotated data
  - implements multicore LDA for efficient topic modeling and theme-extraction
  - modularized code makes this work highly reusable for other domain-specific literature tasks: code can be easily refitted for legal datasets, a corpus of classics etc.
  - goes the additional step of using these results to **rework training data** and train models


[#nayak2023tds](biblio.md#nayak2023tds)

["How to Convert Any Text Into a Graph of Concepts"](https://towardsdatascience.com/how-to-convert-any-text-into-a-graph-of-concepts-110844f22a1a)  
**Rahul Nayak**, _Towards Data Science_ (2023-11-09)

  - "a method to convert any text corpus into a _graph of concepts_" (aka KG)
  - use KGs to implement RAG and "chat with our documents"
  - Q: is this work solid enough to cite in an academic paper??



## counterexamples

[#nizami2023llm](biblio.md#nizami2023llm)

["Extracting Relation from Sentence using LLM"](https://medium.com/@nizami_muhammad/extracting-relation-from-sentence-using-llm-597d0c0310a8)
**Muhammad Nizami** _Medium_ (2023-11-15)


[#lawrence2024ttg](biblio.md#lawrence2024ttg)

["Text-to-Graph via LLM: pre-training, prompting, or tuning?"](https://medium.com/@peter.lawrence_47665/text-to-graph-via-llm-pre-training-prompting-or-tuning-3233d1165360)
**Peter Lawrence** _Medium_ (2024-01-16)

