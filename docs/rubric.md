# Appendix: ML OSS Evaluation Rubric

The following checklist provides an evaluation rubric for open source code related to machine learning research.
For any given code repository, tally a score based on these questions:

 - **Q1:** Does the repository use a business-friendly license?
 - **Q2:** Does the code install correctly with either `pip` or `conda` package managers?
 - **Q3:** Are the library dependencies reasonably current, not using pinned versions for popular libraries?
 - **Q4:** Has the project provided sample code which runs without exceptions?
 - **Q5:** Can the sample code reproduce the published results of the research?
 - **Q6:** Does the library provide affordances for data integration, i.e., it's not optimized for a particular benchmark?
 - **Q7:** Can the code be called programmatically as a library, i.e., not run primarily through a command line interface (CLI), and not requiring container/microservice orchestration?
 - **Q8:** Will the library and its dependencies pass a reasonable level of security audit without structural changes?
 - **Q9:** Does the code support concurrency and parallelization?
 - **Q10:** Has the repo been maintained within the past six months?


## Dependency Evaluations

Based on this checklist, the dependencies integrated within this project scores as follows:

rubric | `OpenNRE` | `pulp` | `qwikidata` | `REBEL` | `spaCy` | `Spotlight` | `SpanMarker` | `transformers`
--- | --- | --- | --- | --- | --- | --- | ---
Q1 | x | x | x | x | x | x | x | x
Q2 | x | x | x | x | x | x | x | x
Q3 | x | x | x | x | x | x | x | x
Q4 | x | x | x | x | x | x | x | x
Q5 | x | x | x | x | x | x | x | x
Q6 | x | x | x | x | x | x | x | x
Q7 | x | x | x | x | x | x | x | x
Q8 | x | x | x | x | x | x | x | x
Q9 | x | x | x | x | x | x | x | x
Q10 | x | x | x | x | x | x | x | x


[`OpenNRE`](https://github.com/thunlp/OpenNRE/)

[`pulp`](https://github.com/coin-or/pulp)

[`qwikidata`](https://github.com/kensho-technologies/qwikidata)

[`REBEL`](https://github.com/Babelscape/rebel)

[`spaCy`](https://spacy.io/)

[`spaCy-DBpedia-Spotlight`](https://github.com/MartinoMensio/spacy-dbpedia-spotlight)

[`SpanMarker`](https://github.com/tomaarsen/SpanMarkerNER/)

[`transformers`](https://github.com/huggingface/transformers/)


There were many other open source code projects which were evaluated
but scored &lt; 8 and were therefore considered unusable for our work.

