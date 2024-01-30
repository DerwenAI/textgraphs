# Appendix: ML OSS Evaluation Rubric

The following checklist provides an evaluation rubric for open source code related to machine learning research. For any given code repository:

 - Does it use a business-friendly license?
 - Does the code install correctly with either `pip` or `conda` package managers?
 - Are the library dependencies reasonably current, not using pinned versions for popular libraries?
 - Has the project provided sample code which runs without exceptions?
 - Can the sample code reproduce the published results of the research?
 - Does the library provide affordances for data integration, i.e., it's not optimized for a particular benchmark?
 - Can the code be called programmatically as a library, i.e., not requiring a command line interface (CLI) to run, and not requiring container/microservice orchestration?
 - Does the code support concurrency and parallelization?
 - Has the repo been maintained within the past six months?
 - Will the library and its dependencies pass a reasonable level of security audit?

Based on the given checklist, this project integrates code from the following dependencies:

[`OpenNRE`](https://github.com/thunlp/OpenNRE/)

[`pulp`](https://github.com/coin-or/pulp)

[`qwikidata`](https://github.com/kensho-technologies/qwikidata)

[`REBEL`](https://github.com/Babelscape/rebel)

[`spaCy`](https://spacy.io/)

[`spaCy-DBpedia-Spotlight`](https://github.com/MartinoMensio/spacy-dbpedia-spotlight)

[`SpanMarker`](https://github.com/tomaarsen/SpanMarkerNER/)

[`transformers`](https://huggingface.co/docs/transformers/index)
