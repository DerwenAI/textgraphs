While many papers proceed from a graph-theoretic definition `G = (V, E)` these typically fail to take into account two important aspects of graph technologies in industry practice:

  1. _labels_ and _properties_ (key/value attribute pairs) for more effective modeling of linked data
  2. _internationalized resource identifiers_ (IRIs) as unique identifiers that map into controlled vocabularies, which can be leveraged for graph queries and semantic inference

Industry analysts sometimes point to these two concerns being represented by competiting approaches, namely 
_labeled property graphs_ (LPG) representation versus
_semantic web standards_ defined by the World Wide Web Consortium (W3C).
Efforts are in progress to harmonize both of these needs within the same graphs, such as
[#gelling2023bgdm](biblio.md#gelling2023bgdm) and [#hartig14](biblio.md#hartig14), which are on track toward eventual standards.
Given some discipline in data modeling practices, both criteria can be met within current graph frameworks provided that:

  * nodes and edges each have specific labels which serve as IRIs that map to a set of controlled vocabularies
  * nodes and edges each have properties, which include probabilities from the point of generation

Building on definitions given in [#martonsv17](biblio.md#martonsv17), [#qin2023sgr](biblio.md#qin2023sgr), this project proceeds from the perspective of primarily using LPG graph representation, while adhering to the aforementioned data modeling discipline.

`G = (V, E, src, tgt, lbl, P)` is an edge-labeled directed multigraph with:

  - a set of nodes V
  - a set of edges E
  - function `src`: E → V` that associates each edge with its source vertex
  - function `tgt: E → V` that associates each edge with its target vertex
  - function `lbl: E → dom(S)` that associates each edge its label
  - function `P: (V ∪ E) → 2p` that associates nodes and edges with their properties

The project architecture enables a "map-reduce" style of distributed processing, so that "chunks" of text (e.g., paragraphs) can be processed independently, with results being aggregated at the end of a batch.
The intermediate processing of each "chunk" uses `NetworkX` [#hagberg2008](biblio.md#hagberg2008) to allow for running in-memory graph algorithms and analytics, and integrate more efficiently with graph machine learning libraries.
Then an `openCypher` representation [#martonsv17](biblio.md#martonsv17) is used to serialize end results, which get aggregated using the open source `KùzuDB` graph database [#feng2023kuzu](biblio.md#feng2023kuzu) and its Python API.
