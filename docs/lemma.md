# Lemma Graph

This project introduces the notion of a _lemma graph_ as an intermediate representation.
Effectively, this provides a kind of cache during the processing of each "chunk" of text.
Think of the end result as "enhanced tokenization" for text used to generate graph data elements.
Other projects might call this by different names:
an "evidence graph" in [#wen2023mindmap](biblio.md#wen2023mindmap)
or a "dynamically growing local KG" in [#loganlpgs19](biblio.md#loganlpgs19).

The lemma graph collects metadata from NLP parsing, entity linking, etc., which generally get discarded in many applications.
Therefore the lemma graph becomes rather "noisy", and in most cases would be too big to store across the analysis of a large corpus.

Leveraging this intermediate form, per chunk, collect the valuable information about nodes, edges, properties, probabilities, etc., to aggregate for the document analysis overall.

Consequently, this project explores the use of topological transforms on graphs to enhance representations for [_graph levels of detail_](https://blog.derwen.ai/graph-levels-of-detail-ea4226abba55), i.e., being able to understand a graph a varying levels of abstraction. 
Note that adjacent areas of interest include emerging work on:

  - _graph of relations_
  - _foundation models for KGs_

Means for "bootstrapping" a _lemma graph_ with initial semantic relations, allows for "sampling" from a curated KG to enhance the graph algorithms used, e.g., through _semantic random walks_ which allow for incorporating heterogeneous sources and relatively large-scale external KGs.
This mechanism also creates opportunities for distributed processing, because the "chunks" of text can follow a _task parallel_ pattern, accumulating the extracted results from each lemma graph into a graph database.
Augmenting a KG iteratively over time follows a similar pattern.
