**TODO**: summarize from <https://blog.derwen.ai/graph-levels-of-detail-ea4226abba55>

results from the combined analysis get collected into an intermediate form which is a probabilistic structure called a _lemma graph_.

note: NLP parsers tend to produce a wealth of annotations from raw text, most of which are thrown away in many application. what if instead this parse information got collected together, temporarily while analyzing a chunk of text?

an application running in production most likely would not want to persist the entirety of _lemma graph_ data generated during analysis of a full corpus. instead, consider this structure as a kind of temporary cache during the analysis for one unit of work, i.e., a "chunk" of text. 

from the pragmatics of writing, editing, and critical review, a natural size for this kind of chunking is to analyze at the paragraph level. in some domains, such as analysis of patent applications, chunking at the level of "claims" might be indicated.

the probabilistic aspects of the intermediate _lemma graph_ data become especially important in a linguistic context:

  * entities have many _surface forms_
  * synonyms (synsets) change meanings in different domains, especially when abbreviated
  * ambiguous references may exist, though not all are important to resolve based on "premature optimization"

Note that semantic modeling practices using RDF tend to have a relatively trivial notion of "synonyms", notably by annotating a subject with one _preferred label_ and zero or more additional labels.
This may be sufficiently descriptive for building taxonomies manually; however, this approach is not sufficient for making the modeled representation computable in light of the many kinds of _surface forms_ and possible sources of ambiguity.
The RDF representation uses `skos:broader` to connect surface forms, and the LPG representation uses probabilities to manage disambiguation these terms.
