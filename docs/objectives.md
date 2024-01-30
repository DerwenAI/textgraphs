Consider three classes of composable elements which are needed for constructing KGs: *nodes*, *edges*, *properties*.
Several areas of machine learning (ML) research can be leveraged to generate these elements from unstructured text sources:

  - nodes: NER, node prediction
  - edges: relation extraction (RE), semantic inference, link prediction
  - properties: NLP parse, entity linking, graph analytics

Weights or probabilities from the analysis can also be used to construct *gradients* for ranking each class of elements in the generated output.
This supports multiple approaches for filtering, selection, and abstraction of the generated composable elements, and helps incorporate domain expertise.

A set of questions follows from this line of inquiry:

**RQ1**: can workflows be defined which integrate LLM-based components and generate _composable elements_ for KGs, while managing the quality of the generated results?

**RQ2**: can topological analysis and decomposition of graph data help inform better ways to generating graph elements, e.g., by leveraging patterns within graphs (motif) and graph abstraction layers?

**RQ3**: where might it be possible to improve data quality for -- training data, benchmarks, evals, etc. -- then iterate to train more effective LLM-based components?

**RQ4**: how can consistent evaluations of open source related to ML research be made, assessing opportunities for reusing code in production-quality libraries?
