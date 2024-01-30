# TextGraphs: raw texts, LLMs, and KGs, oh my!

<img src="assets/logo.png" width="113" alt="illustration of a lemma graph"/>

Welcome to the **TextGraphs** library...

## Overview

_Explore uses of large language models (LLMs) in semi-automated knowledge graph (KG) construction from unstructured text sources, with human-in-the-loop (HITL) affordances to incorporate guidance from domain experts._

What is "generative AI" in the context of working with knowledge graphs?
Initial attempts tend to fit a simple pattern based on _prompt engineering_: present text sources to a LLM-based chat interface, asking to generate an entire graph.
This is generally expensive and results are often poor.
Moreover, the lack of controls or curation in this approach represents a serious disconnect with how KGs get curated to represent an organization's domain expertise.

Can the definition of "generative" be reformulated for KGs?
Instead of trying to use a fully-automated "black box", what if it were possible to generate _composable elements_ which then get aggregated into a KG?
Some research in topological analysis of graphs indicates potential ways to decompose graphs, which can then be re-composed probabilistically.
While the mathematics may be sound, these techniques need to be understood in the context of a full range of tasks within KG-construction workflows to assess how they can apply for real-world graph data.

This project explores the use of LLM-augmented components within natural language workflows, focusing on small well-defined tasks within the scope of KG construction.
To address challenges in this problem, this project considers improved means of tokenization, for handling input.
In addition, a range of methods are considered for filtering and selecting elements of the output stream, re-composing them into KGs.
This has a side-effect of providing steps toward better pattern identification and variable abstraction layers for graph data, for _graph levels of detail_ (GLOD).

While many papers aim to evaluate benchmarks, this line of inquiry focuses on integration:
means of combining multiple complementary research projects;
how to evaluate the outcomes of other projects to assess their potential usefulness in production-quality libraries;
and suggested directions for improving the LLM-based components of NLP workflows used to construct KGs.

  - demo: <https://huggingface.co/spaces/DerwenAI/textgraphs>
  - code: <https://github.com/DerwenAI/textgraphs>
  - bibliography: <https://derwen.ai/docs/txg/biblio>


## Index Terms

_natural language processing_,
_knowledge graph construction_,
_large language models_,
_entity extraction_,
_entity linking_,
_relation extraction_,
_semantic random walk_,
_human-in-the-loop_,
_topological decomposition of graphs_,
_graph levels of detail_,
_network motifs_,
