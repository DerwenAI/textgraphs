# Conclusions

**DRAFT** (WIP)

`TextGraphs` library provides a highly configurable and extensible open source Python library for the integration and evaluation of several LLM components. This has been built with attention to allowing for concurrency and parallelism for high-performance computing on distributed systems.

  - HF space: <https://huggingface.co/spaces/DerwenAI/textgraphs>
  - repo: <https://github.com/DerwenAI/textgraphs>
  - docs: <https://derwen.ai/docs/txg/>
  - biblio: <https://derwen.ai/docs/txg/biblio/>
  - DOI: 10.5281/zenodo.10431783

TODO:

  - leverage co-reference
  - leverage closure constrained by domain/range
  - general => specific, uncertain => confident

The state of _relation extraction_ is arguably immature.
While the papers in this area compare against benchmarks, their training datasets mostly have been built from Wikidata sources, and inferred relations result in _labels_ not IRIs.
This precludes downstream use of the inferred relations for semantic inference.
Ultimately, how can better training data be developed -- e.g., for relation extraction -- to improve large models used in constructing/augmenting knowledge graphs?

Observation: 
Many existing projects produce results which are **descriptive, but not computable**.
However, given recent innovations, such as _DPO_, there appear to be many opportunities for reworking the training datasets used in
NRE and RE models, following the pattern of `Notus`


**R1**: demonstrated how to leverage LLM components while emphasizing HITL (domain experts) and quality of results


**R2**: suggested areas where investments in data quality 
may provide substantial gains

One key take-away from this project is that the model deployments are relatively haphazard across a wide spectrum of performance: some of the open source dependencies use efficient frameworks such as Hugging Face `transformers` to load models, while others use ad-hoc approaches which are much less performant. 

Granted, use of LLMs and other deep learning models is expected to increase computational requirements substantially.
Given the integration of APIs, the compute, memory, and network requirements for running the `TextGraphs` library in product can be quite large. 
Software engineering optimizations can reduce these requirements substantially through use of hardware acceleration, localized services, proxy/caching, and concurrency.

However, a more effective approach would be to make investments in data quality (training datasets, benchmarks, evals, etc.) for gains within the core technologies used here: NER, RE, etc.
Data-first iterations on the model dependencies can alleviate much of this problem.


**R3**: proposed rubric for evaluating/rating ML open source 
w.r.t. production use cases

This project integrates available open source projects across a wide range of NLP topics.
Perspectives were gained from evaluating many open source LLM projects related to NLP components, and the state of readiness for their use in production libraries overall.

Note that reproducibility rates are abysmally low for open source which accompanies machine learning research papers.
Few project install correctly, and fewer still run without exceptions.
Even among the better available OSS project for a given research topic (e.g., _graph embeddings_, _relation extraction_) tend to not have been maintained for years. Of the projects which run, few reproduce their published results, and most are oriented toward command-line (CLI) use to prove specific benchmarks claims.
These tend to be difficult to rework into production-quality libraries, due to concerns about performance, security, licensing, etc.

As an outcome of this inquiry, this project presents a rubric for evaluating research papers and their associated code, based on reproducibility and eventual usefulness in software implementations.
