# Reference: `textgraphs` package
<img src='../assets/nouns/api.png' alt='API by Adnen Kadri from the Noun Project' />
Package definitions for the `TextGraphs` library.

see copyright/license https://huggingface.co/spaces/DerwenAI/textgraphs/blob/main/README.md


## [`TextGraphs` class](#TextGraphs)

Construct a _lemma graph_ from the unstructured text source,
then extract ranked phrases using a `textgraph` algorithm.
    
---
#### [`infer_relations_async` method](#textgraphs.TextGraphs.infer_relations_async)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/doc.py#L641)

```python
infer_relations_async(pipe, debug=False)
```
Gather triples representing inferred relations and build edges,
concurrently by running an async queue.
<https://stackoverflow.com/questions/52582685/using-asyncio-queue-for-producer-consumer-flow>

Make sure to call beforehand: `TextGraphs.collect_graph_elements()`

  * `pipe` : `textgraphs.pipe.Pipeline`  
configured pipeline for this document

  * `debug` : `bool`  
debugging flag

  * *returns* : `typing.List[textgraphs.elem.Edge]`  
a list of the inferred `Edge` objects



---
#### [`__init__` method](#textgraphs.TextGraphs.__init__)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/doc.py#L80)

```python
__init__(factory=None, iri_base="https://github.com/DerwenAI/textgraphs/ns/")
```
Constructor.

  * `factory` : `typing.Optional[textgraphs.pipe.PipelineFactory]`  
optional `PipelineFactory` used to configure components



---
#### [`create_pipeline` method](#textgraphs.TextGraphs.create_pipeline)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/doc.py#L103)

```python
create_pipeline(text_input)
```
Use the pipeline factory to create a pipeline (e.g., `spaCy.Document`)
for each text input, which are typically paragraph-length.

  * `text_input` : `str`  
raw text to be parsed by this pipeline

  * *returns* : `textgraphs.pipe.Pipeline`  
a configured pipeline



---
#### [`create_render` method](#textgraphs.TextGraphs.create_render)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/doc.py#L122)

```python
create_render()
```
Create an object for rendering the graph in `PyVis` HTML+JavaScript.

  * *returns* : `textgraphs.vis.RenderPyVis`  
a configured `RenderPyVis` object for generating graph visualizations



---
#### [`collect_graph_elements` method](#textgraphs.TextGraphs.collect_graph_elements)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/doc.py#L381)

```python
collect_graph_elements(pipe, text_id=0, para_id=0, debug=False)
```
Collect the elements of a _lemma graph_ from the results of running
the `textgraph` algorithm. These elements include: parse dependencies,
lemmas, entities, and noun chunks.

Make sure to call beforehand: `TextGraphs.create_pipeline()`

  * `pipe` : `textgraphs.pipe.Pipeline`  
configured pipeline for this document

  * `text_id` : `int`  
text (top-level document) identifier

  * `para_id` : `int`  
paragraph identitifer

  * `debug` : `bool`  
debugging flag



---
#### [`construct_lemma_graph` method](#textgraphs.TextGraphs.construct_lemma_graph)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/doc.py#L474)

```python
construct_lemma_graph(debug=False)
```
Construct the base level of the _lemma graph_ from the collected
elements. This gets represented in `NetworkX` as a directed graph
with parallel edges.

Make sure to call beforehand: `TextGraphs.collect_graph_elements()`

  * `debug` : `bool`  
debugging flag



---
#### [`perform_entity_linking` method](#textgraphs.TextGraphs.perform_entity_linking)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/doc.py#L534)

```python
perform_entity_linking(pipe, debug=False)
```
Perform _entity linking_ based on the `KnowledgeGraph` object.

Make sure to call beforehand: `TextGraphs.collect_graph_elements()`

  * `pipe` : `textgraphs.pipe.Pipeline`  
configured pipeline for this document

  * `debug` : `bool`  
debugging flag



---
#### [`infer_relations` method](#textgraphs.TextGraphs.infer_relations)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/doc.py#L705)

```python
infer_relations(pipe, debug=False)
```
Gather triples representing inferred relations and build edges.

Make sure to call beforehand: `TextGraphs.collect_graph_elements()`

  * `pipe` : `textgraphs.pipe.Pipeline`  
configured pipeline for this document

  * `debug` : `bool`  
debugging flag

  * *returns* : `typing.List[textgraphs.elem.Edge]`  
a list of the inferred `Edge` objects



---
#### [`calc_phrase_ranks` method](#textgraphs.TextGraphs.calc_phrase_ranks)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/doc.py#L893)

```python
calc_phrase_ranks(pr_alpha=0.85, debug=False)
```
Calculate the weights for each node in the _lemma graph_, then
stack-rank the nodes so that entities have priority over lemmas.

Phrase ranks are normalized to sum to 1.0 and these now represent
the ranked entities extracted from the document.

Make sure to call beforehand: `TextGraphs.construct_lemma_graph()`

  * `pr_alpha` : `float`  
optional `alpha` parameter for the PageRank algorithm

  * `debug` : `bool`  
debugging flag



---
#### [`get_phrases` method](#textgraphs.TextGraphs.get_phrases)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/doc.py#L940)

```python
get_phrases()
```
Return the entities extracted from the document.

Make sure to call beforehand: `TextGraphs.calc_phrase_ranks()`

  * *yields* :  
extracted entities



---
#### [`get_phrases_as_df` method](#textgraphs.TextGraphs.get_phrases_as_df)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/doc.py#L973)

```python
get_phrases_as_df()
```
Return the ranked extracted entities as a dataframe.

Make sure to call beforehand: `TextGraphs.calc_phrase_ranks()`

  * *returns* : `pandas.core.frame.DataFrame`  
a `pandas.DataFrame` of the extracted entities



---
#### [`export_rdf` method](#textgraphs.TextGraphs.export_rdf)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/doc.py#L990)

```python
export_rdf(lang="en")
```
Extract the entities and relations which have IRIs as RDF triples.

  * `lang` : `str`  
language identifier

  * *returns* : `str`  
RDF triples N3 (Turtle) format as a string



---
#### [`denormalize_iri` method](#textgraphs.TextGraphs.denormalize_iri)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/doc.py#L1085)

```python
denormalize_iri(uri_ref)
```
Discern between a parsed entity and a linked entity.

  * *returns* : `str`  
_lemma_key_ for a parsed entity, the full IRI for a linked entity



---
#### [`load_bootstrap_ttl` method](#textgraphs.TextGraphs.load_bootstrap_ttl)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/doc.py#L1103)

```python
load_bootstrap_ttl(ttl_str, debug=False)
```
Parse a TTL string with an RDF semantic graph representation to load
bootstrap definitions for the _lemma graph_ prior to parsing, e.g.,
for synonyms.

  * `ttl_str` : `str`  
RDF triples in TTL (Turtle/N3) format

  * `debug` : `bool`  
debugging flag



---
#### [`export_kuzu` method](#textgraphs.TextGraphs.export_kuzu)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/doc.py#L1215)

```python
export_kuzu(zip_name="lemma.zip", debug=False)
```
Export a labeled property graph for KùzuDB (openCypher).

  * `debug` : `bool`  
debugging flag

  * *returns* : `str`  
name of the generated ZIP file



## [`SimpleGraph` class](#SimpleGraph)

An in-memory graph used to build a `MultiDiGraph` in NetworkX.
    
---
#### [`__init__` method](#textgraphs.SimpleGraph.__init__)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/graph.py#L31)

```python
__init__()
```
Constructor.



---
#### [`reset` method](#textgraphs.SimpleGraph.reset)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/graph.py#L42)

```python
reset()
```
Re-initialize the data structures, resetting all but the configuration.



---
#### [`make_node` method](#textgraphs.SimpleGraph.make_node)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/graph.py#L53)

```python
make_node(tokens, key, span, kind, text_id, para_id, sent_id, label=None, length=1, linked=True)
```
Lookup and return a `Node` object.
By default, link matching keys into the same node.
Otherwise instantiate a new node if it does not exist already.

  * `tokens` : `typing.List[textgraphs.elem.Node]`  
list of parsed tokens

  * `key` : `str`  
lemma key (invariant)

  * `span` : `spacy.tokens.token.Token`  
token span for the parsed entity

  * `kind` : `<enum 'NodeEnum'>`  
the kind of this `Node` object

  * `text_id` : `int`  
text (top-level document) identifier

  * `para_id` : `int`  
paragraph identitifer

  * `sent_id` : `int`  
sentence identifier

  * `label` : `typing.Optional[str]`  
node label (for a new object)

  * `length` : `int`  
length of token span

  * `linked` : `bool`  
flag for whether this links to an entity

  * *returns* : `textgraphs.elem.Node`  
the constructed `Node` object



---
#### [`make_edge` method](#textgraphs.SimpleGraph.make_edge)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/graph.py#L167)

```python
make_edge(src_node, dst_node, kind, rel, prob, key=None, debug=False)
```
Lookup an edge, creating a new one if it does not exist already,
and increment the count if it does.

  * `src_node` : `textgraphs.elem.Node`  
source node in the triple

  * `dst_node` : `textgraphs.elem.Node`  
destination node in the triple

  * `kind` : `<enum 'RelEnum'>`  
the kind of this `Edge` object

  * `rel` : `str`  
relation label

  * `prob` : `float`  
probability of this `Edge` within the graph

  * `key` : `typing.Optional[str]`  
lemma key (invariant); generate a key if this is not provided

  * `debug` : `bool`  
debugging flag

  * *returns* : `typing.Optional[textgraphs.elem.Edge]`  
the constructed `Edge` object; this may be `None` if the input parameters indicate skipping the edge



---
#### [`dump_lemma_graph` method](#textgraphs.SimpleGraph.dump_lemma_graph)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/graph.py#L236)

```python
dump_lemma_graph()
```
Dump the _lemma graph_ as a JSON string in _node-link_ format,
suitable for serialization and subsequent use in JavaScript,
Neo4j, Graphistry, etc.

Make sure to call beforehand: `TextGraphs.calc_phrase_ranks()`

  * *returns* : `str`  
a JSON representation of the exported _lemma graph_ in



---
#### [`load_lemma_graph` method](#textgraphs.SimpleGraph.load_lemma_graph)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/graph.py#L299)

```python
load_lemma_graph(json_str, debug=False)
```
Load from a JSON string in
a JSON representation of the exported _lemma graph_ in
[_node-link_](https://networkx.org/documentation/stable/reference/readwrite/json_graph.html)
format

  * `debug` : `bool`  
debugging flag



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
Otherwise return its `label` value.

  * *returns* : `typing.Optional[str]`  
a label for the linked entity



---
#### [`get_name` method](#textgraphs.Node.get_name)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/elem.py#L135)

```python
get_name()
```
Return a brief name for the graphical depiction of this Node.

  * *returns* : `str`  
brief label to be used in a graph



---
#### [`get_stacked_count` method](#textgraphs.Node.get_stacked_count)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/elem.py#L152)

```python
get_stacked_count()
```
Return a modified count, to redact verbs and linked entities from
the stack-rank partitions.

  * *returns* : `int`  
count, used for re-ranking extracted entities



---
#### [`get_pos` method](#textgraphs.Node.get_pos)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/elem.py#L168)

```python
get_pos()
```
Generate a position span for `OpenNRE`.

  * *returns* : `typing.Tuple[int, int]`  
a position span needed for `OpenNRE` relation extraction



## [`Edge` class](#Edge)

A data class representing an edge between two nodes.
    
---
#### [`__repr__` method](#textgraphs.Edge.__repr__)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/dataclasses.py#L232)

```python
__repr__()
```

## [`EnumBase` class](#EnumBase)

A mixin for Enum codecs.
    
## [`NodeEnum` class](#NodeEnum)

Enumeration for the kinds of node categories
    
## [`RelEnum` class](#RelEnum)

Enumeration for the kinds of edge relations
    
## [`PipelineFactory` class](#PipelineFactory)

Factory pattern for building a pipeline, which is one of the more
expensive operations with `spaCy`
    
---
#### [`__init__` method](#textgraphs.PipelineFactory.__init__)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/pipe.py#L434)

```python
__init__(spacy_model="en_core_web_sm", ner=None, kg=<textgraphs.pipe.KnowledgeGraph object at 0x130529960>, infer_rels=[])
```
Constructor which instantiates the `spaCy` pipelines:

  * `tok_pipe` -- regular generator for parsed tokens
  * `ner_pipe` -- with entities merged
  * `aux_pipe` -- spotlight entity linking

which will be needed for parsing and entity linking.

  * `spacy_model` : `str`  
the specific model to use in `spaCy` pipelines

  * `ner` : `typing.Optional[textgraphs.pipe.Component]`  
optional custom NER component

  * `kg` : `textgraphs.pipe.KnowledgeGraph`  
knowledge graph used for entity linking

  * `infer_rels` : `typing.List[textgraphs.pipe.InferRel]`  
a list of components for inferring relations



---
#### [`create_pipeline` method](#textgraphs.PipelineFactory.create_pipeline)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/pipe.py#L508)

```python
create_pipeline(text_input)
```
Instantiate the document pipelines needed to parse the input text.

  * `text_input` : `str`  
raw text to be parsed

  * *returns* : `textgraphs.pipe.Pipeline`  
a configured `Pipeline` object



## [`Pipeline` class](#Pipeline)

Manage parsing of a document, which is assumed to be paragraph-sized.
    
---
#### [`__init__` method](#textgraphs.Pipeline.__init__)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/pipe.py#L216)

```python
__init__(text_input, tok_pipe, ner_pipe, aux_pipe, kg, infer_rels)
```
Constructor.

  * `text_input` : `str`  
raw text to be parsed

  * `tok_pipe` : `spacy.language.Language`  
the `spaCy.Language` pipeline used for tallying individual tokens

  * `ner_pipe` : `spacy.language.Language`  
the `spaCy.Language` pipeline used for tallying named entities

  * `aux_pipe` : `spacy.language.Language`  
the `spaCy.Language` pipeline used for auxiliary components (e.g., `DBPedia Spotlight`)

  * `kg` : `textgraphs.pipe.KnowledgeGraph`  
knowledge graph used for entity linking

  * `infer_rels` : `typing.List[textgraphs.pipe.InferRel]`  
a list of components for inferring relations



---
#### [`get_lemma_key` classmethod](#textgraphs.Pipeline.get_lemma_key)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/pipe.py#L267)

```python
get_lemma_key(span, placeholder=False)
```
Compose a unique, invariant lemma key for the given span.

  * `span` : `typing.Union[spacy.tokens.span.Span, spacy.tokens.token.Token]`  
span of tokens within the lemma

  * `placeholder` : `bool`  
flag for whether to create a placeholder

  * *returns* : `str`  
a composed lemma key



---
#### [`get_ent_lemma_keys` method](#textgraphs.Pipeline.get_ent_lemma_keys)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/pipe.py#L308)

```python
get_ent_lemma_keys()
```
Iterate through the fully qualified lemma keys for an extracted entity.

  * *yields* :  
the lemma keys within an extracted entity



---
#### [`link_noun_chunks` method](#textgraphs.Pipeline.link_noun_chunks)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/pipe.py#L321)

```python
link_noun_chunks(nodes, debug=False)
```
Link any noun chunks which are not already subsumed by named entities.

  * `nodes` : `dict`  
dictionary of `Node` objects in the graph

  * `debug` : `bool`  
debugging flag

  * *returns* : `typing.List[textgraphs.elem.NounChunk]`  
a list of identified noun chunks which are novel



---
#### [`iter_entity_pairs` method](#textgraphs.Pipeline.iter_entity_pairs)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/pipe.py#L373)

```python
iter_entity_pairs(pipe_graph, max_skip, debug=True)
```
Iterator for entity pairs for which the algorithm infers relations.

  * `pipe_graph` : `networkx.classes.multigraph.MultiGraph`  
a `networkx.MultiGraph` representation of the graph, reused for graph algorithms

  * `max_skip` : `int`  
maximum distance between entities for inferred relations

  * `debug` : `bool`  
debugging flag

  * *yields* :  
pairs of entities within a range, e.g., to use for relation extraction



## [`Component` class](#Component)

Abstract base class for a `spaCy` pipeline component.
    
---
#### [`augment_pipe` method](#textgraphs.Component.augment_pipe)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/pipe.py#L41)

```python
augment_pipe(factory)
```
Encapsulate a `spaCy` call to `add_pipe()` configuration.

  * `factory` : `PipelineFactory`  
a `PipelineFactory` used to configure components



## [`NERSpanMarker` class](#NERSpanMarker)

Configures a `spaCy` pipeline component for `SpanMarkerNER`
    
---
#### [`__init__` method](#textgraphs.NERSpanMarker.__init__)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/ner.py#L22)

```python
__init__(ner_model="tomaarsen/span-marker-roberta-large-ontonotes5")
```
Constructor.

  * `ner_model` : `str`  
model to be used in `SpanMarker`



---
#### [`augment_pipe` method](#textgraphs.NERSpanMarker.augment_pipe)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/ner.py#L36)

```python
augment_pipe(factory)
```
Encapsulate a `spaCy` call to `add_pipe()` configuration.

  * `factory` : `textgraphs.pipe.PipelineFactory`  
the `PipelineFactory` used to configure this pipeline component



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
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/pipe.py#L63)

```python
augment_pipe(factory)
```
Encapsulate a `spaCy` call to `add_pipe()` configuration.

  * `factory` : `PipelineFactory`  
a `PipelineFactory` used to configure components



---
#### [`remap_ner` method](#textgraphs.KnowledgeGraph.remap_ner)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/pipe.py#L76)

```python
remap_ner(label)
```
Remap the OntoTypes4 values from NER output to more general-purpose IRIs.

  * `label` : `typing.Optional[str]`  
input NER label, an `OntoTypes4` value

  * *returns* : `typing.Optional[str]`  
an IRI for the named entity



---
#### [`normalize_prefix` method](#textgraphs.KnowledgeGraph.normalize_prefix)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/pipe.py#L92)

```python
normalize_prefix(iri, debug=False)
```
Normalize the given IRI to use standard namespace prefixes.

  * `iri` : `str`  
input IRI, in fully-qualified domain representation

  * `debug` : `bool`  
debugging flag

  * *returns* : `str`  
the compact IRI representation, using an RDF namespace prefix



---
#### [`perform_entity_linking` method](#textgraphs.KnowledgeGraph.perform_entity_linking)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/pipe.py#L113)

```python
perform_entity_linking(graph, pipe, debug=False)
```
Perform _entity linking_ based on "spotlight" and other services.

  * `graph` : `textgraphs.graph.SimpleGraph`  
source graph

  * `pipe` : `Pipeline`  
configured pipeline for the current document

  * `debug` : `bool`  
debugging flag



---
#### [`resolve_rel_iri` method](#textgraphs.KnowledgeGraph.resolve_rel_iri)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/pipe.py#L135)

```python
resolve_rel_iri(rel, lang="en", debug=False)
```
Resolve a `rel` string from a _relation extraction_ model which has
been trained on this knowledge graph.

  * `rel` : `str`  
relation label, generation these source from Wikidata for many RE projects

  * `lang` : `str`  
language identifier

  * `debug` : `bool`  
debugging flag

  * *returns* : `typing.Optional[str]`  
a resolved IRI



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
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/kg.py#L165)

```python
__init__(spotlight_api="https://api.dbpedia-spotlight.org/en", dbpedia_search_api="https://lookup.dbpedia.org/api/search", dbpedia_sparql_api="https://dbpedia.org/sparql", wikidata_api="https://www.wikidata.org/w/api.php", ner_map=OrderedDict([('CARDINAL', {'iri': 'http://dbpedia.org/resource/Cardinal_number', 'definition': 'Numerals that do not fall under another type', 'label': 'cardinal number'}), ('DATE', {'iri': 'http://dbpedia.org/ontology/date', 'definition': 'Absolute or relative dates or periods', 'label': 'date'}), ('EVENT', {'iri': 'http://dbpedia.org/ontology/Event', 'definition': 'Named hurricanes, battles, wars, sports events, etc.', 'label': 'event'}), ('FAC', {'iri': 'http://dbpedia.org/ontology/Infrastructure', 'definition': 'Buildings, airports, highways, bridges, etc.', 'label': 'infrastructure'}), ('GPE', {'iri': 'http://dbpedia.org/ontology/Country', 'definition': 'Countries, cities, states', 'label': 'country'}), ('LANGUAGE', {'iri': 'http://dbpedia.org/ontology/Language', 'definition': 'Any named language', 'label': 'language'}), ('LAW', {'iri': 'http://dbpedia.org/ontology/Law', 'definition': 'Named documents made into laws', 'label': 'law'}), ('LOC', {'iri': 'http://dbpedia.org/ontology/Place', 'definition': 'Non-GPE locations, mountain ranges, bodies of water', 'label': 'place'}), ('MONEY', {'iri': 'http://dbpedia.org/resource/Money', 'definition': 'Monetary values, including unit', 'label': 'money'}), ('NORP', {'iri': 'http://dbpedia.org/ontology/nationality', 'definition': 'Nationalities or religious or political groups', 'label': 'nationality'}), ('ORDINAL', {'iri': 'http://dbpedia.org/resource/Ordinal_number', 'definition': 'Ordinal number, i.e., first, second, etc.', 'label': 'ordinal number'}), ('ORG', {'iri': 'http://dbpedia.org/ontology/Organisation', 'definition': 'Companies, agencies, institutions, etc.', 'label': 'organization'}), ('PERCENT', {'iri': 'http://dbpedia.org/resource/Percentage', 'definition': 'Percentage', 'label': 'percentage'}), ('PERSON', {'iri': 'http://dbpedia.org/ontology/Person', 'definition': 'People, including fictional', 'label': 'person'}), ('PRODUCT', {'iri': 'http://dbpedia.org/ontology/product', 'definition': 'Vehicles, weapons, foods, etc. (Not services)', 'label': 'product'}), ('QUANTITY', {'iri': 'http://dbpedia.org/resource/Quantity', 'definition': 'Measurements, as of weight or distance', 'label': 'quantity'}), ('TIME', {'iri': 'http://dbpedia.org/ontology/time', 'definition': 'Times smaller than a day', 'label': 'time'}), ('WORK OF ART', {'iri': 'http://dbpedia.org/resource/Work_of_art', 'definition': 'Titles of books, songs, etc.', 'label': 'work of art'})]), ns_prefix=OrderedDict([('dbc', 'http://dbpedia.org/resource/Category:'), ('dbt', 'http://dbpedia.org/resource/Template:'), ('dbr', 'http://dbpedia.org/resource/'), ('yago', 'http://dbpedia.org/class/yago/'), ('dbd', 'http://dbpedia.org/datatype/'), ('dbo', 'http://dbpedia.org/ontology/'), ('dbp', 'http://dbpedia.org/property/'), ('units', 'http://dbpedia.org/units/'), ('dbpedia-commons', 'http://commons.dbpedia.org/resource/'), ('dbpedia-wikicompany', 'http://dbpedia.openlinksw.com/wikicompany/'), ('dbpedia-wikidata', 'http://wikidata.dbpedia.org/resource/'), ('wd', 'http://www.wikidata.org/'), ('wd_ent', 'http://www.wikidata.org/entity/'), ('rdf', 'http://www.w3.org/1999/02/22-rdf-syntax-ns#'), ('schema', 'https://schema.org/'), ('owl', 'http://www.w3.org/2002/07/owl#')]), min_alias=0.8, min_similarity=0.9)
```
Constructor.

  * `spotlight_api` : `str`  
`DBPedia Spotlight` API or equivalent local service

  * `dbpedia_search_api` : `str`  
`DBPedia Search` API or equivalent local service

  * `dbpedia_sparql_api` : `str`  
`DBPedia SPARQL` API or equivalent local service

  * `wikidata_api` : `str`  
`Wikidata Search` API or equivalent local service

  * `ner_map` : `dict`  
named entity map for standardizing IRIs

  * `ns_prefix` : `dict`  
RDF namespace prefixes

  * `min_alias` : `float`  
minimum alias probability threshold for accepting linked entities

  * `min_similarity` : `float`  
minimum label similarity threshold for accepting linked entities



---
#### [`augment_pipe` method](#textgraphs.KGWikiMedia.augment_pipe)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/kg.py#L219)

```python
augment_pipe(factory)
```
Encapsulate a `spaCy` call to `add_pipe()` configuration.

  * `factory` : `textgraphs.pipe.PipelineFactory`  
a `PipelineFactory` used to configure components



---
#### [`remap_ner` method](#textgraphs.KGWikiMedia.remap_ner)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/kg.py#L237)

```python
remap_ner(label)
```
Remap the OntoTypes4 values from NER output to more general-purpose IRIs.

  * `label` : `typing.Optional[str]`  
input NER label, an `OntoTypes4` value

  * *returns* : `typing.Optional[str]`  
an IRI for the named entity



---
#### [`normalize_prefix` method](#textgraphs.KGWikiMedia.normalize_prefix)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/kg.py#L266)

```python
normalize_prefix(iri, debug=False)
```
Normalize the given IRI using the standard DBPedia namespace prefixes.

  * `iri` : `str`  
input IRI, in fully-qualified domain representation

  * `debug` : `bool`  
debugging flag

  * *returns* : `str`  
the compact IRI representation, using an RDF namespace prefix



---
#### [`perform_entity_linking` method](#textgraphs.KGWikiMedia.perform_entity_linking)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/kg.py#L306)

```python
perform_entity_linking(graph, pipe, debug=False)
```
Perform _entity linking_ based on `DBPedia Spotlight` and other services.

  * `graph` : `textgraphs.graph.SimpleGraph`  
source graph

  * `pipe` : `textgraphs.pipe.Pipeline`  
configured pipeline for the current document

  * `debug` : `bool`  
debugging flag



---
#### [`resolve_rel_iri` method](#textgraphs.KGWikiMedia.resolve_rel_iri)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/kg.py#L370)

```python
resolve_rel_iri(rel, lang="en", debug=False)
```
Resolve a `rel` string from a _relation extraction_ model which has
been trained on this _knowledge graph_, which defaults to using the
`WikiMedia` graphs.

  * `rel` : `str`  
relation label, generation these source from Wikidata for many RE projects

  * `lang` : `str`  
language identifier

  * `debug` : `bool`  
debugging flag

  * *returns* : `typing.Optional[str]`  
a resolved IRI



---
#### [`wikidata_search` method](#textgraphs.KGWikiMedia.wikidata_search)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/kg.py#L575)

```python
wikidata_search(query, lang="en", debug=False)
```
Query the Wikidata search API.

  * `query` : `str`  
query string

  * `lang` : `str`  
language identifier

  * `debug` : `bool`  
debugging flag

  * *returns* : `typing.Optional[textgraphs.elem.KGSearchHit]`  
search hit, if any



---
#### [`dbpedia_search_entity` method](#textgraphs.KGWikiMedia.dbpedia_search_entity)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/kg.py#L641)

```python
dbpedia_search_entity(query, lang="en", debug=False)
```
Perform a DBPedia API search.

  * `query` : `str`  
query string

  * `lang` : `str`  
language identifier

  * `debug` : `bool`  
debugging flag

  * *returns* : `typing.Optional[textgraphs.elem.KGSearchHit]`  
search hit, if any



---
#### [`dbpedia_sparql_query` method](#textgraphs.KGWikiMedia.dbpedia_sparql_query)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/kg.py#L738)

```python
dbpedia_sparql_query(sparql, debug=False)
```
Perform a SPARQL query on DBPedia.

  * `sparql` : `str`  
SPARQL query string

  * `debug` : `bool`  
debugging flag

  * *returns* : `dict`  
dictionary of query results



---
#### [`dbpedia_wikidata_equiv` method](#textgraphs.KGWikiMedia.dbpedia_wikidata_equiv)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/kg.py#L791)

```python
dbpedia_wikidata_equiv(dbpedia_iri, debug=False)
```
Perform a SPARQL query on DBPedia to find an equivalent Wikidata entity.

  * `dbpedia_iri` : `str`  
IRI in DBpedia

  * `debug` : `bool`  
debugging flag

  * *returns* : `typing.Optional[str]`  
equivalent IRI in Wikidata



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
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/pipe.py#L188)

```python
gen_triples_async(pipe, queue, debug=False)
```
Infer relations as triples produced to a queue _concurrently_.

  * `pipe` : `Pipeline`  
configured pipeline for the current document

  * `queue` : `asyncio.queues.Queue`  
queue of inference tasks to be performed

  * `debug` : `bool`  
debugging flag



---
#### [`gen_triples` method](#textgraphs.InferRel.gen_triples)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/pipe.py#L166)

```python
gen_triples(pipe, debug=False)
```
Infer relations as triples through a generator _iteratively_.

  * `pipe` : `Pipeline`  
configured pipeline for the current document

  * `debug` : `bool`  
debugging flag

  * *yields* :  
generated triples



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

  * `model` : `str`  
the specific model to be used in `OpenNRE`

  * `max_skip` : `int`  
maximum distance between entities for inferred relations

  * `min_prob` : `float`  
minimum probability threshold for accepting an inferred relation



---
#### [`gen_triples` method](#textgraphs.InferRel_OpenNRE.gen_triples)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/rel.py#L58)

```python
gen_triples(pipe, debug=False)
```
Iterate on entity pairs to drive `OpenNRE`, inferring relations
represented as triples which get produced by a generator.

  * `pipe` : `textgraphs.pipe.Pipeline`  
configured pipeline for the current document

  * `debug` : `bool`  
debugging flag

  * *yields* :  
generated triples as candidates for inferred relations



## [`InferRel_Rebel` class](#InferRel_Rebel)

Perform relation extraction based on the `REBEL` model.
<https://github.com/Babelscape/rebel>
<https://huggingface.co/spaces/Babelscape/mrebel-demo>
    
---
#### [`__init__` method](#textgraphs.InferRel_Rebel.__init__)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/rel.py#L121)

```python
__init__(lang="en_XX", mrebel_model="Babelscape/mrebel-large")
```
Constructor.

  * `lang` : `str`  
language identifier

  * `mrebel_model` : `str`  
tokenizer model to be used



---
#### [`tokenize_sent` method](#textgraphs.InferRel_Rebel.tokenize_sent)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/rel.py#L145)

```python
tokenize_sent(text)
```
Apply the tokenizer manually, since we need to extract special tokens.

  * `text` : `str`  
input text for the sentence to be tokenized

  * *returns* : `str`  
extracted tokens



---
#### [`extract_triplets_typed` method](#textgraphs.InferRel_Rebel.extract_triplets_typed)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/rel.py#L174)

```python
extract_triplets_typed(text)
```
Parse the generated text and extract its triplets.

  * `text` : `str`  
input text for the sentence to use in inference

  * *returns* : `list`  
a list of extracted triples



---
#### [`gen_triples` method](#textgraphs.InferRel_Rebel.gen_triples)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/rel.py#L259)

```python
gen_triples(pipe, debug=False)
```
Drive `REBEL` to infer relations for each sentence, represented as
triples which get produced by a generator.

  * `pipe` : `textgraphs.pipe.Pipeline`  
configured pipeline for the current document

  * `debug` : `bool`  
debugging flag

  * *yields* :  
generated triples as candidates for inferred relations



## [`RenderPyVis` class](#RenderPyVis)

Render the _lemma graph_ as a `PyVis` network.
    
---
#### [`__init__` method](#textgraphs.RenderPyVis.__init__)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/vis.py#L76)

```python
__init__(graph, kg)
```
Constructor.

  * `graph` : `textgraphs.graph.SimpleGraph`  
source graph to be visualized

  * `kg` : `textgraphs.pipe.KnowledgeGraph`  
knowledge graph used for entity linking



---
#### [`render_lemma_graph` method](#textgraphs.RenderPyVis.render_lemma_graph)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/vis.py#L94)

```python
render_lemma_graph(debug=True)
```
Prepare the structure of the `NetworkX` graph to use for building
and returning a `PyVis` network to render.

Make sure to call beforehand: `TextGraphs.calc_phrase_ranks()`

  * `debug` : `bool`  
debugging flag

  * *returns* : `pyvis.network.Network`  
<a `pyvis.network.Network` interactive visualization



---
#### [`draw_communities` method](#textgraphs.RenderPyVis.draw_communities)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/vis.py#L166)

```python
draw_communities(spring_distance=1.4, debug=False)
```
Cluster the communities in the _lemma graph_, then draw a
`NetworkX` graph of the notes with a specific color for each
community.

Make sure to call beforehand: `TextGraphs.calc_phrase_ranks()`

  * `spring_distance` : `float`  
`NetworkX` parameter used to separate clusters visually

  * `debug` : `bool`  
debugging flag

  * *returns* : `typing.Dict[int, int]`  
a map of the calculated communities



---
#### [`generate_wordcloud` method](#textgraphs.RenderPyVis.generate_wordcloud)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/vis.py#L237)

```python
generate_wordcloud(background="black")
```
Generate a tag cloud from the given phrases.

Make sure to call beforehand: `TextGraphs.calc_phrase_ranks()`

  * `background` : `str`  
background color for the rendering

  * *returns* : `wordcloud.wordcloud.WordCloud`  
the rendering as a `wordcloud.WordCloud` object, which can be used to generate PNG images, etc.



## [`NodeStyle` class](#NodeStyle)

Dataclass used for styling PyVis nodes.
    
---
#### [`__setattr__` method](#textgraphs.NodeStyle.__setattr__)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main<string>#L2)

```python
__setattr__(name, value)
```

## [`GraphOfRelations` class](#GraphOfRelations)

Attempt to reproduce results published in
"INGRAM: Inductive Knowledge Graph Embedding via Relation Graphs"
<https://arxiv.org/abs/2305.19987>
    
---
#### [`__init__` method](#textgraphs.GraphOfRelations.__init__)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/gor.py#L100)

```python
__init__(source)
```
Constructor.

  * `source` : `textgraphs.graph.SimpleGraph`  
source graph to be transformed



---
#### [`load_ingram` method](#textgraphs.GraphOfRelations.load_ingram)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/gor.py#L125)

```python
load_ingram(json_file, debug=False)
```
Load data for a source graph, as illustrated in _lee2023ingram_

  * `json_file` : `pathlib.Path`  
path for the JSON dataset to load

  * `debug` : `bool`  
debugging flag



---
#### [`seeds` method](#textgraphs.GraphOfRelations.seeds)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/gor.py#L197)

```python
seeds(debug=False)
```
Prep data for the topological transform illustrated in _lee2023ingram_

  * `debug` : `bool`  
debugging flag



---
#### [`trace_source_graph` method](#textgraphs.GraphOfRelations.trace_source_graph)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/gor.py#L241)

```python
trace_source_graph()
```
Output a "seed" representation of the source graph.



---
#### [`construct_gor` method](#textgraphs.GraphOfRelations.construct_gor)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/gor.py#L311)

```python
construct_gor(debug=False)
```
Perform the topological transform described by _lee2023ingram_,
constructing a _graph of relations_ (GOR) and calculating
_affinity scores_ between entities in the GOR based on their
definitions:

> we measure the affinity between two relations by considering how many
entities are shared between them and how frequently they share the same
entity

  * `debug` : `bool`  
debugging flag



---
#### [`tally_frequencies` classmethod](#textgraphs.GraphOfRelations.tally_frequencies)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/gor.py#L348)

```python
tally_frequencies(counter)
```
Tally the frequency of shared entities.

  * `counter` : `collections.Counter`  
`counter` data collection for the rel_b/entity pairs

  * *returns* : `int`  
tallied values for one relation



---
#### [`get_affinity_scores` method](#textgraphs.GraphOfRelations.get_affinity_scores)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/gor.py#L401)

```python
get_affinity_scores(debug=False)
```
Reproduce metrics based on the example published in _lee2023ingram_

  * `debug` : `bool`  
debugging flag

  * *returns* : `typing.Dict[tuple, float]`  
the calculated affinity scores



---
#### [`trace_metrics` method](#textgraphs.GraphOfRelations.trace_metrics)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/gor.py#L454)

```python
trace_metrics(scores)
```
Compare the calculated affinity scores with results from a published
example.

  * `scores` : `typing.Dict[tuple, float]`  
the calculated affinity scores between pairs of relations (i.e., observed values)

  * *returns* : `pandas.core.frame.DataFrame`  
a `pandas.DataFrame` where the rows compare expected vs. observed affinity scores



---
#### [`render_gor_plt` method](#textgraphs.GraphOfRelations.render_gor_plt)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/gor.py#L522)

```python
render_gor_plt(scores)
```
Visualize the _graph of relations_ using `matplotlib`

  * `scores` : `typing.Dict[tuple, float]`  
the calculated affinity scores between pairs of relations (i.e., observed values)



---
#### [`render_gor_pyvis` method](#textgraphs.GraphOfRelations.render_gor_pyvis)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/gor.py#L563)

```python
render_gor_pyvis(scores)
```
Visualize the _graph of relations_ interactively using `PyVis`

  * `scores` : `typing.Dict[tuple, float]`  
the calculated affinity scores between pairs of relations (i.e., observed values)

  * *returns* : `pyvis.network.Network`  
a `pyvis.networkNetwork` representation of the transformed graph



## [`TransArc` class](#TransArc)

A data class representing one transformed rel-node-rel triple in
a _graph of relations_.
    
---
#### [`__repr__` method](#textgraphs.TransArc.__repr__)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/dataclasses.py#L232)

```python
__repr__()
```

## [`RelDir` class](#RelDir)

Enumeration for the directions of a relation.
    
## [`SheafSeed` class](#SheafSeed)

A data class representing a node from the source graph plus its
partial edge, based on a _Sheaf Theory_ decomposition of a graph.
    
---
#### [`__repr__` method](#textgraphs.SheafSeed.__repr__)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/dataclasses.py#L232)

```python
__repr__()
```

## [`Affinity` class](#Affinity)

A data class representing the affinity scores from one entity
in the transformed _graph of relations_.

NB: there are much more efficient ways to calculate these
_affinity scores_ using sparse tensor algebra; this approach
illustrates the process -- for research and debugging.
    
---
#### [`__repr__` method](#textgraphs.Affinity.__repr__)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/dataclasses.py#L232)

```python
__repr__()
```

---
## [module functions](#textgraphs)
---
#### [`calc_quantile_bins` function](#textgraphs.calc_quantile_bins)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/util.py#L65)

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

  * *returns* : `typing.Tuple[str, str]`  
version tag and commit hash



---
#### [`root_mean_square` function](#textgraphs.root_mean_square)
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/util.py#L116)

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
[*\[source\]*](https://github.com/DerwenAI/textgraphs/blob/main/textgraphs/util.py#L88)

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
