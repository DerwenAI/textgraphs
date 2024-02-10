Rather than fully automatic KG construction, this approach emphasizes means of incorporating _domain experts_ through "human-in-the-loop" (HITL) techniques.

Multiple techniques can be employed to construct gradients for both the generated nodes and edges, starting with the quantitative scores from model inference.

  - gradient for recommending extracted entities: _named entity recognition_, _textrank_, _probabilistic soft logic_, etc.
  - gradient for recommending extracted relations: _relation extraction_, _graph of relations_, etc.

Results extracted from _lemma graphs_ provide gradients which can be leveraged to elicit feedback from domain experts:

  - high-pass filter: accept results as valid automated inference
  - low-pass filter: reject results as errors and noise

For the results which fall in-between, a recsys or similar UI can elicit review from domain experts, based on _active learning_, _weak supervision_, etc. see <https://argilla.io/>

subsequent to the HITL validation, the more valuable results collected within a _lemma graph_ can be extracted as the primary output from this approach.

Based on a process of iterating through a text document in chunks, the results from one iteration can be used to bootstrap the _lemma graph_ for the next iteration. this provides a natural means of accumulating (i.e., aggregating) results from the overall analysis.

By extension, this bootstrap/accumulation process can be used in the distributed processing of a corpus of documents, where the "data exhaust" of abstracted _lemma graphs_ used to bootstrap analysis workflows effectively becomes a _knowledge graph_, as a side-effect of the analysis.

<img src="../assets/hitl.png" width="750" />
