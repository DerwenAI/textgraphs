# Reference: `textgraphs` package
<img src='../assets/nouns/api.png' alt='API by Adnen Kadri from the Noun Project' />
Package definitions for the `TextGraphs` library.

see copyright/license https://huggingface.co/spaces/DerwenAI/textgraphs/blob/main/README.md


## [`TextGraphs` class](#TextGraphs)

Construct a _lemma graph_ from the unstructured text source,
then extract ranked phrases using a `textgraph` algorithm.

---
#### [`infer_relations_async` method](#textgraphs.TextGraphs.infer_relations_async)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/doc.py#L397)

```python
infer_relations_async(pipe, debug=False)
```
Gather triples representing inferred relations and build edges,
concurrently by running an async queue.
<https://stackoverflow.com/questions/52582685/using-asyncio-queue-for-producer-consumer-flow>

Make sure to call beforehand:

  * `TextGraphs.collect_graph_elements()`



---
#### [`__init__` method](#textgraphs.TextGraphs.__init__)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/doc.py#L66)

```python
__init__(factory=None)
```
Constructor.



---
#### [`create_pipeline` method](#textgraphs.TextGraphs.create_pipeline)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/doc.py#L83)

```python
create_pipeline(text_input)
```
Use the pipeline factory to create a pipeline (e.g., `spaCy.Document`)
for each text input, which are typically paragraph-length.



---
#### [`create_render` method](#textgraphs.TextGraphs.create_render)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/doc.py#L96)

```python
create_render()
```
Create an object for rendering the graph in `PyVis` HTML+JavaScript.



---
#### [`collect_graph_elements` method](#textgraphs.TextGraphs.collect_graph_elements)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/doc.py#L247)

```python
collect_graph_elements(pipe, text_id=0, para_id=0, debug=False)
```
Collect the elements of a _lemma graph_ from the results of running
the `textgraph` algorithm. These elements include: parse dependencies,
lemmas, entities, and noun chunks.

Make sure to call beforehand:

  * `TextGraphs.create_pipeline()`



---
#### [`perform_entity_linking` method](#textgraphs.TextGraphs.perform_entity_linking)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/doc.py#L323)

```python
perform_entity_linking(pipe, debug=False)
```
Perform _entity linking_ based on the `KnowledgeGraph` object.

Make sure to call beforehand:

  * `TextGraphs.collect_graph_elements()`



---
#### [`infer_relations` method](#textgraphs.TextGraphs.infer_relations)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/doc.py#L454)

```python
infer_relations(pipe, debug=False)
```
Gather triples representing inferred relations and build edges.

Make sure to call beforehand:

  * `TextGraphs.collect_graph_elements()`



---
#### [`calc_phrase_ranks` method](#textgraphs.TextGraphs.calc_phrase_ranks)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/doc.py#L608)

```python
calc_phrase_ranks(pr_alpha=0.85, debug=False)
```
Calculate the weights for each node in the _lemma graph_, then
stack-rank the nodes so that entities have priority over lemmas.

Phrase ranks are normalized to sum to 1.0 and these now represent
the ranked entities extracted from the document.

Make sure to call beforehand:

  * `TextGraphs.collect_graph_elements()`



---
#### [`get_phrases` method](#textgraphs.TextGraphs.get_phrases)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/doc.py#L651)

```python
get_phrases()
```
Return the entities extracted from the document.

Make sure to call beforehand:

  * `TextGraphs.calc_phrase_ranks()`



---
#### [`get_phrases_as_df` method](#textgraphs.TextGraphs.get_phrases_as_df)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/doc.py#L683)

```python
get_phrases_as_df()
```
Return the ranked extracted entities as a `pandas.DataFrame`

Make sure to call beforehand:

  * `TextGraphs.calc_phrase_ranks()`



## [`SimpleGraph` class](#SimpleGraph)

An in-memory graph used to build a `MultiDiGraph` in NetworkX.

---
#### [`__init__` method](#textgraphs.SimpleGraph.__init__)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/graph.py#L30)

```python
__init__()
```
Constructor.



---
#### [`reset` method](#textgraphs.SimpleGraph.reset)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/graph.py#L41)

```python
reset()
```
Re-initialize the data structures, resetting all but the configuration.



---
#### [`make_node` method](#textgraphs.SimpleGraph.make_node)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/graph.py#L52)

```python
make_node(tokens, key, span, kind, text_id, para_id, sent_id, label=None, length=1, linked=True)
```
Lookup and return a `Node` object:

    * default: link matching keys into the same node
    * instantiate a new node if it does not exist already



---
#### [`make_edge` method](#textgraphs.SimpleGraph.make_edge)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/graph.py#L120)

```python
make_edge(src_node, dst_node, kind, rel, prob, debug=False)
```
Lookup an edge, creating a new one if it does not exist already,
and increment the count if it does.



---
#### [`construct_lemma_graph` method](#textgraphs.SimpleGraph.construct_lemma_graph)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/graph.py#L163)

```python
construct_lemma_graph(debug=False)
```
Construct the base level of the _lemma graph_ from the collected
elements. This gets represented in `NetworkX` as a directed graph
with parallel edges.



---
#### [`dump_lemma_graph` method](#textgraphs.SimpleGraph.dump_lemma_graph)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/graph.py#L206)

```python
dump_lemma_graph()
```
Dump the _lemma graph_ as a JSON string in _node-link_ format,
suitable for serialization and subsequent use in JavaScript,
Neo4j, Graphistry, etc.

Make sure to call beforehand:

  * `TextGraphs.calc_phrase_ranks()`



## [`Node` class](#Node)

A data class representing one node, i.e., an extracted phrase.

---
#### [`__repr__` method](#textgraphs.Node.__repr__)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/dataclasses.py#L232)

```python
__repr__()
```

---
#### [`get_linked_label` method](#textgraphs.Node.get_linked_label)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/elem.py#L119)

```python
get_linked_label()
```
When this node has a linked entity, return that IRI.
Otherwise return is `label` value.



---
#### [`get_name` method](#textgraphs.Node.get_name)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/elem.py#L132)

```python
get_name()
```
Return a brief name for the graphical depiction of this Node.



---
#### [`get_stacked_count` method](#textgraphs.Node.get_stacked_count)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/elem.py#L146)

```python
get_stacked_count()
```
Return a modified count, to redact verbs and linked entities from
the stack-rank partitions.



---
#### [`get_pos` method](#textgraphs.Node.get_pos)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/elem.py#L159)

```python
get_pos()
```
Generate a position span for OpenNRE.



## [`NodeEnum` class](#NodeEnum)

Enumeration for the kinds of node categories

## [`Edge` class](#Edge)

A data class representing an edge between two nodes.

---
#### [`__repr__` method](#textgraphs.Edge.__repr__)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/dataclasses.py#L232)

```python
__repr__()
```

## [`RelEnum` class](#RelEnum)

Enumeration for the kinds of edge relations

## [`PipelineFactory` class](#PipelineFactory)

Factory pattern for building a pipeline, which is one of the more
expensive operations with `spaCy`

---
#### [`__init__` method](#textgraphs.PipelineFactory.__init__)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/pipe.py#L319)

```python
__init__(spacy_model="en_core_web_sm", ner=None, kg=<textgraphs.pipe.KnowledgeGraph object at 0x12175f310>, infer_rels=[])
```
Constructor which instantiates the `spaCy` pipelines:

  * `tok_pipe` -- regular generator for parsed tokens
  * `ner_pipe` -- with entities merged
  * `aux_pipe` -- spotlight entity linking



---
#### [`create_pipeline` method](#textgraphs.PipelineFactory.create_pipeline)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/pipe.py#L379)

```python
create_pipeline(text_input)
```
Instantiate the document pipelines needed to parse the input text.



## [`Pipeline` class](#Pipeline)

Manage parsing of a document, which is assumed to be paragraph-sized.

---
#### [`__init__` method](#textgraphs.Pipeline.__init__)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/pipe.py#L152)

```python
__init__(text_input, tok_pipe, ner_pipe, aux_pipe, kg, infer_rels)
```
Constructor.



---
#### [`get_lemma_key` classmethod](#textgraphs.Pipeline.get_lemma_key)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/pipe.py#L185)

```python
get_lemma_key(span, placeholder=False)
```
Compose a unique, invariant lemma key for the given span.



---
#### [`get_ent_lemma_keys` method](#textgraphs.Pipeline.get_ent_lemma_keys)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/pipe.py#L217)

```python
get_ent_lemma_keys()
```
Iterate through the fully qualified lemma keys for an extracted entity.



---
#### [`link_noun_chunks` method](#textgraphs.Pipeline.link_noun_chunks)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/pipe.py#L227)

```python
link_noun_chunks(nodes, debug=False)
```
Link any noun chunks which are not already subsumed by named entities.



---
#### [`iter_entity_pairs` method](#textgraphs.Pipeline.iter_entity_pairs)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/pipe.py#L270)

```python
iter_entity_pairs(pipe_graph, max_skip, debug=True)
```
Iterator for entity pairs for which the algorithm infers relations.



## [`Component` class](#Component)

Abstract base class for a `spaCy` pipeline component.

---
#### [`augment_pipe` method](#textgraphs.Component.augment_pipe)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/pipe.py#L40)

```python
augment_pipe(factory)
```
Encapsulate a `spaCy` call to `add_pipe()` configuration.



## [`NERSpanMarker` class](#NERSpanMarker)

Configures a `spaCy` pipeline component for `SpanMarkerNER`

---
#### [`__init__` method](#textgraphs.NERSpanMarker.__init__)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/ner.py#L22)

```python
__init__(ner_model="tomaarsen/span-marker-roberta-large-ontonotes5")
```
Constructor.



---
#### [`augment_pipe` method](#textgraphs.NERSpanMarker.augment_pipe)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/ner.py#L33)

```python
augment_pipe(factory)
```
Encapsulate a `spaCy` call to `add_pipe()` configuration.



## [`NounChunk` class](#NounChunk)

A data class representing one noun chunk, i.e., a candidate as an extracted phrase.

---
#### [`__repr__` method](#textgraphs.NounChunk.__repr__)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/dataclasses.py#L232)

```python
__repr__()
```

## [`KnowledgeGraph` class](#KnowledgeGraph)

Base class for a _knowledge graph_ interface.

---
#### [`augment_pipe` method](#textgraphs.KnowledgeGraph.augment_pipe)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/pipe.py#L56)

```python
augment_pipe(factory)
```
Encapsulate a `spaCy` call to `add_pipe()` configuration.



---
#### [`remap_ner` method](#textgraphs.KnowledgeGraph.remap_ner)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/pipe.py#L66)

```python
remap_ner(label)
```
Remap the OntoTypes4 values from NER output to more general-purpose IRIs.



---
#### [`normalize_prefix` method](#textgraphs.KnowledgeGraph.normalize_prefix)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/pipe.py#L76)

```python
normalize_prefix(iri, debug=False)
```
Normalize the given IRI to use standard namespace prefixes.



---
#### [`perform_entity_linking` method](#textgraphs.KnowledgeGraph.perform_entity_linking)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/pipe.py#L88)

```python
perform_entity_linking(graph, pipe, debug=False)
```
Perform _entity linking_ based on "spotlight" and other services.



---
#### [`resolve_rel_iri` method](#textgraphs.KnowledgeGraph.resolve_rel_iri)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/pipe.py#L101)

```python
resolve_rel_iri(rel, lang="en", debug=False)
```
Resolve a `rel` string from a _relation extraction_ model which has
been trained on this knowledge graph.



## [`KGSearchHit` class](#KGSearchHit)

A data class representing a hit from a _knowledge graph_ search.

---
#### [`__repr__` method](#textgraphs.KGSearchHit.__repr__)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/dataclasses.py#L232)

```python
__repr__()
```

## [`KGWikiMedia` class](#KGWikiMedia)

Manage access to WikiMedia-related APIs.

---
#### [`__init__` method](#textgraphs.KGWikiMedia.__init__)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/kg.py#L147)

```python
__init__(spotlight_api="https://api.dbpedia-spotlight.org/en", dbpedia_search_api="https://lookup.dbpedia.org/api/search", dbpedia_sparql_api="https://dbpedia.org/sparql", wikidata_api="https://www.wikidata.org/w/api.php", ner_map=OrderedDict([('CARDINAL', {'iri': 'http://dbpedia.org/resource/Cardinal_number', 'definition': 'Numerals that do not fall under another type'}), ('DATE', {'iri': 'http://dbpedia.org/ontology/date', 'definition': 'Absolute or relative dates or periods'}), ('EVENT', {'iri': 'http://dbpedia.org/ontology/Event', 'definition': 'Named hurricanes, battles, wars, sports events, etc.'}), ('FAC', {'iri': 'http://dbpedia.org/ontology/Infrastructure', 'definition': 'Buildings, airports, highways, bridges, etc.'}), ('GPE', {'iri': 'http://dbpedia.org/ontology/Country', 'definition': 'Countries, cities, states'}), ('LANGUAGE', {'iri': 'http://dbpedia.org/ontology/Language', 'definition': 'Any named language'}), ('LAW', {'iri': 'http://dbpedia.org/ontology/Law', 'definition': 'Named documents made into laws '}), ('LOC', {'iri': 'http://dbpedia.org/ontology/Place', 'definition': 'Non-GPE locations, mountain ranges, bodies of water'}), ('MONEY', {'iri': 'http://dbpedia.org/resource/Money', 'definition': 'Monetary values, including unit'}), ('NORP', {'iri': 'http://dbpedia.org/ontology/nationality', 'definition': 'Nationalities or religious or political groups'}), ('ORDINAL', {'iri': 'http://dbpedia.org/resource/Ordinal_number', 'definition': 'Ordinal number, i.e., first, second, etc.'}), ('ORG', {'iri': 'http://dbpedia.org/ontology/Organisation', 'definition': 'Companies, agencies, institutions, etc.'}), ('PERCENT', {'iri': 'http://dbpedia.org/resource/Percentage', 'definition': 'Percentage'}), ('PERSON', {'iri': 'http://dbpedia.org/ontology/Person', 'definition': 'People, including fictional'}), ('PRODUCT', {'iri': 'http://dbpedia.org/ontology/product', 'definition': 'Vehicles, weapons, foods, etc. (Not services)'}), ('QUANTITY', {'iri': 'http://dbpedia.org/resource/Quantity', 'definition': 'Measurements, as of weight or distance'}), ('TIME', {'iri': 'http://dbpedia.org/ontology/time', 'definition': 'Times smaller than a day'}), ('WORK OF ART', {'iri': 'http://dbpedia.org/resource/Work_of_art', 'definition': 'Titles of books, songs, etc.'})]), ns_prefix=OrderedDict([('dbc', 'http://dbpedia.org/resource/Category:'), ('dbt', 'http://dbpedia.org/resource/Template:'), ('dbr', 'http://dbpedia.org/resource/'), ('yago', 'http://dbpedia.org/class/yago/'), ('dbd', 'http://dbpedia.org/datatype/'), ('dbo', 'http://dbpedia.org/ontology/'), ('dbp', 'http://dbpedia.org/property/'), ('units', 'http://dbpedia.org/units/'), ('dbpedia-commons', 'http://commons.dbpedia.org/resource/'), ('dbpedia-wikicompany', 'http://dbpedia.openlinksw.com/wikicompany/'), ('dbpedia-wikidata', 'http://wikidata.dbpedia.org/resource/'), ('wd', 'http://www.wikidata.org/'), ('rdf', 'http://www.w3.org/1999/02/22-rdf-syntax-ns#'), ('schema', 'https://schema.org/'), ('owl', 'http://www.w3.org/2002/07/owl#')]), min_alias=0.8, min_similarity=0.9)
```
Constructor.



---
#### [`augment_pipe` method](#textgraphs.KGWikiMedia.augment_pipe)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/kg.py#L177)

```python
augment_pipe(factory)
```
Encapsulate a `spaCy` call to `add_pipe()` configuration.



---
#### [`remap_ner` method](#textgraphs.KGWikiMedia.remap_ner)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/kg.py#L192)

```python
remap_ner(label)
```
Remap the OntoTypes4 values from NER output to more general-purpose IRIs.



---
#### [`normalize_prefix` method](#textgraphs.KGWikiMedia.normalize_prefix)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/kg.py#L214)

```python
normalize_prefix(iri, debug=False)
```
Normalize the given IRI to use the standard DBPedia namespace prefixes.



---
#### [`perform_entity_linking` method](#textgraphs.KGWikiMedia.perform_entity_linking)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/kg.py#L244)

```python
perform_entity_linking(graph, pipe, debug=False)
```
Perform _entity linking_ based on `DBPedia Spotlight` and other services.



---
#### [`resolve_rel_iri` method](#textgraphs.KGWikiMedia.resolve_rel_iri)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/kg.py#L299)

```python
resolve_rel_iri(rel, lang="en", debug=False)
```
Resolve a `rel` string from a _relation extraction_ model which has
been trained on this _knowledge graph_.

Defaults to the `WikiMedia` graphs.



---
#### [`wikidata_search` method](#textgraphs.KGWikiMedia.wikidata_search)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/kg.py#L458)

```python
wikidata_search(query, lang="en", debug=False)
```
Query the Wikidata search API.



---
#### [`dbpedia_search_entity` method](#textgraphs.KGWikiMedia.dbpedia_search_entity)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/kg.py#L510)

```python
dbpedia_search_entity(query, lang="en", debug=False)
```
Perform a DBPedia API search.



---
#### [`dbpedia_sparql_query` method](#textgraphs.KGWikiMedia.dbpedia_sparql_query)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/kg.py#L591)

```python
dbpedia_sparql_query(sparql, debug=False)
```
Perform a SPARQL query on DBPedia.



---
#### [`dbpedia_wikidata_equiv` method](#textgraphs.KGWikiMedia.dbpedia_wikidata_equiv)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/kg.py#L631)

```python
dbpedia_wikidata_equiv(dbpedia_iri, debug=False)
```
Perform a SPARQL query on DBPedia to find an equivalent Wikidata entity.



## [`LinkedEntity` class](#LinkedEntity)

A data class representing one linked entity.

---
#### [`__repr__` method](#textgraphs.LinkedEntity.__repr__)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/dataclasses.py#L232)

```python
__repr__()
```

## [`InferRel` class](#InferRel)

Abstract base class for a _relation extraction_ model wrapper.

---
#### [`gen_triples_async` method](#textgraphs.InferRel.gen_triples_async)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/pipe.py#L133)

```python
gen_triples_async(pipe, queue, debug=False)
```
Infer relations as triples produced to a queue _concurrently_.



---
#### [`gen_triples` method](#textgraphs.InferRel.gen_triples)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/pipe.py#L120)

```python
gen_triples(pipe, debug=False)
```
Infer relations as triples through a generator _iteratively_.



## [`InferRel_OpenNRE` class](#InferRel_OpenNRE)

Perform relation extraction based on the `OpenNRE` model.
<https://github.com/thunlp/OpenNRE>

---
#### [`__init__` method](#textgraphs.InferRel_OpenNRE.__init__)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/rel.py#L33)

```python
__init__(model="wiki80_cnn_softmax", max_skip=11, min_prob=0.9)
```
Constructor.



---
#### [`gen_triples` method](#textgraphs.InferRel_OpenNRE.gen_triples)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/rel.py#L49)

```python
gen_triples(pipe, debug=False)
```
Iterate on entity pairs to drive `OpenNRE`, inferring relations
represented as triples which get produced by a generator.



## [`InferRel_Rebel` class](#InferRel_Rebel)

Perform relation extraction based on the `REBEL` model.
<https://github.com/Babelscape/rebel>
<https://huggingface.co/spaces/Babelscape/mrebel-demo>

---
#### [`__init__` method](#textgraphs.InferRel_Rebel.__init__)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/rel.py#L103)

```python
__init__(lang="en_XX", mrebel_model="Babelscape/mrebel-large")
```
Constructor.



---
#### [`tokenize_sent` method](#textgraphs.InferRel_Rebel.tokenize_sent)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/rel.py#L121)

```python
tokenize_sent(text)
```
Apply the tokenizer manually, since we need to extract special tokens.



---
#### [`extract_triplets_typed` method](#textgraphs.InferRel_Rebel.extract_triplets_typed)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/rel.py#L144)

```python
extract_triplets_typed(text)
```
Parse the generated text and extract its triplets.



---
#### [`gen_triples` method](#textgraphs.InferRel_Rebel.gen_triples)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/rel.py#L223)

```python
gen_triples(pipe, debug=False)
```
Drive `REBEL` to infer relations for each sentence, represented as
triples which get produced by a generator.



## [`RenderPyVis` class](#RenderPyVis)

Render the _lemma graph_ as a `PyVis` network.

---
#### [`__init__` method](#textgraphs.RenderPyVis.__init__)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/vis.py#L75)

```python
__init__(graph, kg)
```
Constructor.



---
#### [`render_lemma_graph` method](#textgraphs.RenderPyVis.render_lemma_graph)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/vis.py#L87)

```python
render_lemma_graph(debug=True)
```
Prepare the structure of the `NetworkX` graph to use for building
and returning a `PyVis` network to render.

Make sure to call beforehand:

  * `TextGraphs.calc_phrase_ranks()`



---
#### [`draw_communities` method](#textgraphs.RenderPyVis.draw_communities)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/vis.py#L146)

```python
draw_communities(spring_distance=1.4, debug=False)
```
Cluster the communities in the _lemma graph_, then draw a
`NetworkX` graph of the notes with a specific color for each
community.

Make sure to call beforehand:

  * `TextGraphs.calc_phrase_ranks()`



---
#### [`generate_wordcloud` method](#textgraphs.RenderPyVis.generate_wordcloud)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/vis.py#L210)

```python
generate_wordcloud(background="black")
```
Generate a tag cloud from the given phrases.

Make sure to call beforehand:

  * `TextGraphs.calc_phrase_ranks()`



## [`NodeStyle` class](#NodeStyle)

Dataclass used for styling PyVis nodes.

---
#### [`__setattr__` method](#textgraphs.NodeStyle.__setattr__)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main<string>#L2)

```python
__setattr__(name, value)
```

---
## [module functions](#textgraphs)
---
#### [`calc_quantile_bins` function](#textgraphs.calc_quantile_bins)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/util.py#L20)

```python
calc_quantile_bins(num_rows)
```
Calculate the bins to use for a quantile stripe,
using [`numpy.linspace`](https://numpy.org/doc/stable/reference/generated/numpy.linspace.html)

  * `num_rows` : `int`
number of rows in the target dataframe

  * *returns* : `numpy.ndarray`
calculated bins, as a `numpy.ndarray`



---
#### [`get_repo_version` function](#textgraphs.get_repo_version)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/version.py#L50)

```python
get_repo_version()
```
Access the Git repository information and return items to identify
the version/commit running in production.



---
#### [`root_mean_square` function](#textgraphs.root_mean_square)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/util.py#L71)

```python
root_mean_square(values)
```
Calculate the [*root mean square*](https://mathworld.wolfram.com/Root-Mean-Square.html)
of the values in the given list.

  * `values` : `typing.List[float]`
list of values to use in the RMS calculation

  * *returns* : `float`
RMS metric as a float



---
#### [`stripe_column` function](#textgraphs.stripe_column)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/util.py#L43)

```python
stripe_column(values, bins)
```
Stripe a column in a dataframe, by interpolating quantiles into a set of discrete indexes.

  * `values` : `list`
list of values to stripe

  * `bins` : `int`
quantile bins; see [`calc_quantile_bins()`](#calc_quantile_bins-function)

  * *returns* : `numpy.ndarray`
the striped column values, as a `numpy.ndarray`



---
## [module types](#textgraphs)
