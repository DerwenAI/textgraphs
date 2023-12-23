#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This class provides a wrapper for access to a _knowledge graph_, which
then runs _entity linking_ and other functions in the pipeline.

This could provide an interface to a graph database, such as Neo4j,
StarDog, KùzuDB, etc., or to an API.

In this default case, we wrap services available via the WikiMedia APIs:

  * DBPedia: Spotlight, SPARQL, Search
  * Wikidata: Search

see copyright/license https://huggingface.co/spaces/DerwenAI/textgraphs/blob/main/README.md
"""

from collections import OrderedDict
from difflib import SequenceMatcher
import http
import json
import time
import traceback
import typing
import urllib.parse

from bs4 import BeautifulSoup  # pylint: disable=E0401
from icecream import ic  # pylint: disable=E0401
from qwikidata.linked_data_interface import get_entity_dict_from_api  # pylint: disable=E0401
import markdown2  # pylint: disable=E0401
import requests  # type: ignore  # pylint: disable=E0401
import spacy  # pylint: disable=E0401

from .defaults import DBPEDIA_MIN_ALIAS, DBPEDIA_MIN_SIM, \
    DBPEDIA_SEARCH_API, DBPEDIA_SPARQL_API, DBPEDIA_SPOTLIGHT_API, \
    WIKIDATA_API
from .elem import Edge, KGSearchHit, LinkedEntity, Node, NodeEnum, RelEnum
from .graph import SimpleGraph
from .pipe import KnowledgeGraph, Pipeline, PipelineFactory


######################################################################
## class definitions

class KGWikiMedia (KnowledgeGraph):  # pylint: disable=R0902,R0903
    """
Manage access to WikiMedia-related APIs.
    """
    REL_ISA: str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
    REL_SAME: str = "http://www.w3.org/2002/07/owl#sameAs"

    NER_MAP: typing.Dict[ str, dict ] = OrderedDict({
        "CARDINAL": {
            "iri": "http://dbpedia.org/resource/Cardinal_number",
            "definition": "Numerals that do not fall under another type"
        },
        "DATE": {
            "iri": "http://dbpedia.org/ontology/date",
            "definition": "Absolute or relative dates or periods"
        },
        "EVENT": {
            "iri": "http://dbpedia.org/ontology/Event",
            "definition": "Named hurricanes, battles, wars, sports events, etc."
        },
        "FAC": {
            "iri": "http://dbpedia.org/ontology/Infrastructure",
            "definition": "Buildings, airports, highways, bridges, etc."
        },
        "GPE": {
            "iri": "http://dbpedia.org/ontology/Country",
            "definition": "Countries, cities, states"
        },
        "LANGUAGE": {
            "iri": "http://dbpedia.org/ontology/Language",
            "definition": "Any named language"
        },
        "LAW": {
            "iri": "http://dbpedia.org/ontology/Law",
            "definition": "Named documents made into laws "
        },
        "LOC": {
            "iri": "http://dbpedia.org/ontology/Place",
            "definition": "Non-GPE locations, mountain ranges, bodies of water"
        },
        "MONEY": {
            "iri": "http://dbpedia.org/resource/Money",
            "definition": "Monetary values, including unit"
        },
        "NORP": {
            "iri": "http://dbpedia.org/ontology/nationality",
            "definition": "Nationalities or religious or political groups"
        },
        "ORDINAL": {
            "iri": "http://dbpedia.org/resource/Ordinal_number",
            "definition": "Ordinal number, i.e., first, second, etc."
        },
        "ORG": {
            "iri": "http://dbpedia.org/ontology/Organisation",
            "definition": "Companies, agencies, institutions, etc."
        },
        "PERCENT": {
            "iri": "http://dbpedia.org/resource/Percentage",
            "definition": "Percentage"
        },
        "PERSON": {
            "iri": "http://dbpedia.org/ontology/Person",
            "definition": "People, including fictional"
        },
        "PRODUCT": {
            "iri": "http://dbpedia.org/ontology/product",
            "definition": "Vehicles, weapons, foods, etc. (Not services)"
        },
        "QUANTITY": {
            "iri": "http://dbpedia.org/resource/Quantity",
            "definition": "Measurements, as of weight or distance"
        },
        "TIME": {
            "iri": "http://dbpedia.org/ontology/time",
            "definition": "Times smaller than a day"
        },
        "WORK OF ART": {
            "iri": "http://dbpedia.org/resource/Work_of_art",
            "definition": "Titles of books, songs, etc."
        },
    })

    NS_PREFIX: typing.Dict[ str, str ] = OrderedDict({
        "dbc": "http://dbpedia.org/resource/Category:",
        "dbt": "http://dbpedia.org/resource/Template:",
        "dbr": "http://dbpedia.org/resource/",
        "yago":"http://dbpedia.org/class/yago/",
        "dbd": "http://dbpedia.org/datatype/",
        "dbo": "http://dbpedia.org/ontology/",
        "dbp": "http://dbpedia.org/property/",
        "units": "http://dbpedia.org/units/",
        "dbpedia-commons": "http://commons.dbpedia.org/resource/",
        "dbpedia-wikicompany": "http://dbpedia.openlinksw.com/wikicompany/",
        "dbpedia-wikidata": "http://wikidata.dbpedia.org/resource/",
        "wd": "http://www.wikidata.org/",
        "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "schema": "https://schema.org/",
        "owl": "http://www.w3.org/2002/07/owl#",
    })


    def __init__ (  # pylint: disable=W0102
        self,
        *,
        spotlight_api: str = DBPEDIA_SPOTLIGHT_API,
        dbpedia_search_api: str = DBPEDIA_SEARCH_API,
        dbpedia_sparql_api: str = DBPEDIA_SPARQL_API,
        wikidata_api: str = WIKIDATA_API,
        ner_map: dict = NER_MAP,
        ns_prefix: dict = NS_PREFIX,
        min_alias: float = DBPEDIA_MIN_ALIAS,
        min_similarity: float = DBPEDIA_MIN_SIM,
        ) -> None:
        """
Constructor.
        """
        self.spotlight_api: str = spotlight_api
        self.dbpedia_search_api: str = dbpedia_search_api
        self.dbpedia_sparql_api: str = dbpedia_sparql_api
        self.wikidata_api: str = wikidata_api
        self.ner_map: dict = ner_map
        self.ns_prefix: dict = ns_prefix
        self.min_alias: float = min_alias
        self.min_similarity: float = min_similarity

        self.ent_cache: dict = {}
        self.iri_cache: dict = {}

        self.markdowner = markdown2.Markdown()


    def augment_pipe (
        self,
        factory: PipelineFactory,
        ) -> None:
        """
Encapsulate a `spaCy` call to `add_pipe()` configuration.
        """
        factory.aux_pipe.add_pipe(
            "dbpedia_spotlight",
            config = {
                "dbpedia_rest_endpoint": self.spotlight_api,  # type: ignore
            },
        )


    def remap_ner (
        self,
        label: typing.Optional[ str ],
        ) -> typing.Optional[ str ]:
        """
Remap the OntoTypes4 values from NER output to more general-purpose IRIs.
        """
        if label is None:
            return None

        try:
            iri: typing.Optional[ dict ] = self.ner_map.get(label)

            if iri is not None:
                return iri["iri"]
        except TypeError as ex:
            ic(ex)
            print(f"unknown label: {label}")

        return None


    def normalize_prefix (
        self,
        iri: str,
        *,
        debug: bool = False,
        ) -> str:
        """
Normalize the given IRI to use the standard DBPedia namespace prefixes.
        """
        iri_parse: urllib.parse.ParseResult = urllib.parse.urlparse(iri)

        if debug:
            ic(iri_parse)

        for prefix, ns_fqdn in self.ns_prefix.items():
            ns_parse: urllib.parse.ParseResult = urllib.parse.urlparse(ns_fqdn)

            if debug:
                ic(prefix, ns_parse.netloc, ns_parse.path)

            if iri_parse.netloc == ns_parse.netloc and iri_parse.path.startswith(ns_parse.path):
                slug: str = iri_parse.path.replace(ns_parse.path, "")

                # return normalized IRI
                return f"{prefix}:{slug}"

        # normalization failed
        return iri


    def perform_entity_linking (
        self,
        graph: SimpleGraph,
        pipe: Pipeline,
        *,
        debug: bool = False,
        ) -> None:
        """
Perform _entity linking_ based on `DBPedia Spotlight` and other services.
        """
        # first pass: use "spotlight" API to markup text
        iter_ents: typing.Iterator[ LinkedEntity ] = self._link_spotlight_entities(
            pipe,
            debug = debug
        )

        for link in iter_ents:
            _ = self._make_link(
                graph,
                pipe,
                link,
                self.REL_ISA,
                debug = debug,
            )

            _ = self._secondary_entity_linking(
                graph,
                pipe,
                link,
                debug = debug,
            )

        # second pass: use KG search on entities which weren't linked by Spotlight
        iter_ents = self._link_kg_search_entities(
            graph,
            debug = debug,
        )

        for link in iter_ents:
            _ = self._make_link(
                graph,
                pipe,
                link,
                self.REL_ISA,
                debug = debug,
            )

            _ = self._secondary_entity_linking(
                graph,
                pipe,
                link,
                debug = debug,
            )


    def resolve_rel_iri (
        self,
        rel: str,
        *,
        lang: str = "en",
        debug: bool = False,
        ) -> typing.Optional[ str ]:
        """
Resolve a `rel` string from a _relation extraction_ model which has
been trained on this _knowledge graph_.

Defaults to the `WikiMedia` graphs.
        """
        # first, check the cache
        if rel in self.iri_cache:
            return self.iri_cache.get(rel)

        # otherwise construct a Wikidata API search
        try:
            hit: dict = self._wikidata_endpoint(
                rel,
                search_type = "property",
                lang = lang,
                debug = debug,
            )

            if debug:
                ic(hit["label"], hit["id"])

            # get the `claims` of the Wikidata property
            prop_id: str = hit["id"]
            prop_dict: dict = get_entity_dict_from_api(prop_id)
            claims: dict = prop_dict["claims"]

            if "P1628" in claims:
                # use `equivalent property` if available
                iri: str = claims["P1628"][0]["mainsnak"]["datavalue"]["value"]
            elif "P2235" in claims:
                # use `external superproperty` as a fallback
                iri = claims["P2235"][0]["mainsnak"]["datavalue"]["value"]
            else:
                ic("no related claims", rel)
                return None

            if debug:
                ic(iri)

            # update the cache
            self.iri_cache[rel] = iri
            return iri
        except Exception as ex:  # pylint: disable=W0718
            ic(ex)
            traceback.print_exc()
            return None


    ######################################################################
    ## private methods, customized per KG instance

    def _wikidata_endpoint (
        self,
        query: str,
        *,
        search_type: str = "item",
        lang: str = "en",
        debug: bool = False,
        ) -> dict:
        """
Call a generic endpoint for Wikidata API.
Raises various untrapped exceptions, to be handled by caller.
        """
        hit: dict = {}

        params: dict = {
            "action": "wbsearchentities",
            "type": search_type,
            "language": lang,
            "format": "json",
            "continue": "0",
            "search": query,
        }

        response: requests.models.Response = requests.get(
            self.wikidata_api,
            params = params,
            headers = {
                "Accept": "application/json",
            },
        )

        if debug:
            ic(response.status_code)

        # check for API success
        if http.HTTPStatus.OK == response.status_code:
            dat: dict = response.json()
            hit = dat["search"][0]

            #print(json.dumps(hit, indent = 2, sort_keys = True))

        return hit


    @classmethod
    def _match_aliases (
        cls,
        query: str,
        label: str,
        aliases: typing.List[ str ],
        *,
        debug: bool = False,
        ) -> typing.Tuple[ float, str ]:
        """
Find the best-matching aliases for a search term.
        """
        # best case scenario: the label is an exact match
        if query == label.lower():
            return ( 1.0, label, )

        # ...therefore the label is not an exact match
        prob_list: typing.List[ typing.Tuple[ float, str ]] = [
            ( SequenceMatcher(None, query, label.lower()).ratio(), label, )
        ]

        # fallback: test the aliases
        for alias in aliases:
            prob: float = SequenceMatcher(None, query, alias.lower()).ratio()

            if prob == 1.0:
                # early termination for success
                return ( prob, alias, )

            prob_list.append(( prob, alias, ))

        # find the closest match
        prob_list.sort(reverse = True)

        if debug:
            ic(prob_list)

        return prob_list[0]


    def _md_to_text (
        self,
        md_text: str,
        ) -> str:
        """
Convert markdown to plain text.
<https://stackoverflow.com/questions/761824/python-how-to-convert-markdown-formatted-text-to-text>
        """
        soup: BeautifulSoup = BeautifulSoup(
            self.markdowner.convert(md_text),
            features = "html.parser",
        )

        return soup.get_text().strip()


    def wikidata_search (
        self,
        query: str,
        *,
        lang: str = "en",
        debug: bool = False,
        ) -> typing.Optional[ KGSearchHit ]:
        """
Query the Wikidata search API.
        """
        try:
            hit: dict = self._wikidata_endpoint(
                query,
                search_type = "item",
                lang = lang,
                debug = debug,
            )

            # extract the needed properties
            url: str = hit["concepturi"]
            label: str = hit["label"]
            descrip: str = hit["description"]

            # determine match likelihood
            prob, _ = self._match_aliases(
                query.lower(),
                label,
                [],
                debug = debug,
            )

            if debug:
                ic(query, url, label, descrip, prob)

            # return a linked entity
            wiki_ent: KGSearchHit = KGSearchHit(
                url,
                label,
                descrip,
                [],
                prob,
            )

            return wiki_ent

        except Exception as ex:  # pylint: disable=W0718
            ic(ex)
            traceback.print_exc()

        return None


    def dbpedia_search_entity (  # pylint: disable=R0914
        self,
        query: str,
        *,
        lang: str = "en",
        debug: bool = False,
        ) -> typing.Optional[ KGSearchHit ]:
        """
Perform a DBPedia API search.
        """
        # first, check the cache
        key: str = "dbpedia:" + query.lower()

        if key in self.ent_cache:
            return self.ent_cache.get(key)

        params: dict = {
            "format": "json",
            "language": lang,
            "query": query,
        }

        try:
            response: requests.models.Response = requests.get(
                self.dbpedia_search_api,
                params = params,
                headers = {
                    "Accept": "application/json",
                },
            )

            if debug:
                ic(response.status_code)

            # check for failed API calls
            if http.HTTPStatus.OK != response.status_code:
                return None

            dat: dict = response.json()
            hit: dict = dat["docs"][0]

            if debug:
                ic(json.dumps(hit, indent = 2))

            iri: str = hit["resource"][0]
            label: str = self._md_to_text(hit["label"][0])
            descrip: str = self._md_to_text(hit["comment"][0])

            aliases: typing.List[ str ] = [
                self._md_to_text(alias)
                for alias in hit["redirectlabel"]
            ]

            prob, best_match = self._match_aliases(
                query.lower(),
                label,
                aliases,
                debug = debug,
            )

            if debug:
                ic(iri, label, descrip, aliases, prob, best_match)

            ent: KGSearchHit = KGSearchHit(
                iri,
                label,
                descrip,
                aliases,
                prob,
            )

            # update the cache
            self.ent_cache[key] = ent
            return ent

        except Exception as ex:  # pylint: disable=W0718
            ic(ex)
            traceback.print_exc()
            return None


    def dbpedia_sparql_query (
        self,
        sparql: str,
        *,
        debug: bool = False,
        ) -> dict:
        """
Perform a SPARQL query on DBPedia.
        """
        dat: dict = {}

        if debug:
            print(sparql)

        params: dict = {
            "query": sparql,
        }

        try:
            response: requests.models.Response = requests.get(
                self.dbpedia_sparql_api,
                params = params,
                headers = {
                    "Accept": "application/json",
                },
            )

            if debug:
                ic(response.status_code)

            # check for failed API calls
            if http.HTTPStatus.OK == response.status_code:
                dat = response.json()
        except Exception as ex:  # pylint: disable=W0718
            ic(ex)
            traceback.print_exc()

        return dat


    def dbpedia_wikidata_equiv (
        self,
        dbpedia_iri: str,
        *,
        debug: bool = False,
        ) -> typing.Optional[ str ]:
        """
Perform a SPARQL query on DBPedia to find an equivalent Wikidata entity.
        """
        # first, check the cache
        if dbpedia_iri in self.iri_cache:
            return self.iri_cache.get(dbpedia_iri)

        sparql: str = """
SELECT DISTINCT ?wikidata_concept
WHERE {{
   {} owl:sameAs ?wikidata_concept .
   FILTER(CONTAINS(STR(?wikidata_concept), "www.wikidata.org"))
}}
LIMIT 1000
        """.strip().replace("\n", " ").format(dbpedia_iri)

        dat: dict = self.dbpedia_sparql_query(
            sparql,
            debug = debug,
        )

        try:
            hit: dict = dat["results"]["bindings"][0]

            if debug:
                print(json.dumps(hit, indent = 2))

            equiv_iri: str = hit["wikidata_concept"]["value"]

            if debug:
                ic(equiv_iri)

            # update the cache
            self.iri_cache[dbpedia_iri] = equiv_iri
            return equiv_iri

        except Exception as ex:  # pylint: disable=W0718
            ic(ex)
            traceback.print_exc()
            return None


    ######################################################################
    ## entity linking

    def _link_spotlight_entities (  # pylint: disable=R0914
        self,
        pipe: Pipeline,
        *,
        debug: bool = False,
        ) -> typing.Iterator[ LinkedEntity ]:
        """
Iterator for the results of using `DBPedia Spotlight` to markup
text with _entity linking_
        """
        ents: typing.List[ spacy.tokens.span.Span ] = list(pipe.aux_doc.ents)

        if debug:
            ic(ents)

        ent_idx: int = 0
        tok_idx: int = 0

        for i, tok in enumerate(pipe.tokens):  # pylint: disable=R1702
            if debug:
                print()
                ic(tok_idx, tok.text, tok.pos)
                ic(ent_idx, len(ents))

            if ent_idx < len(ents):
                ent = ents[ent_idx]

                if debug:
                    ic(ent.start, tok_idx)

                if ent.start == tok_idx:
                    if debug:
                        ic(ent.text, ent.start, len(ent))
                        ic(ent.kb_id_, ent._.dbpedia_raw_result["@similarityScore"])
                        ic(ent._.dbpedia_raw_result)

                    prob: float = float(ent._.dbpedia_raw_result["@similarityScore"])
                    count: int = int(ent._.dbpedia_raw_result["@support"])

                    if tok.pos == "PROPN" and prob >= self.min_similarity:
                        kg_ent: typing.Optional[ KGSearchHit ] = self.dbpedia_search_entity(  # type: ignore  # pylint: disable=C0301
                            ent.text,
                            debug = debug,
                        )

                        if debug:
                            ic(kg_ent)

                        if kg_ent is not None and kg_ent.prob > self.min_alias:  # type: ignore
                            iri: str = ent.kb_id_

                            dbp_link: LinkedEntity = LinkedEntity(
                                ent,
                                iri,
                                len(ent),
                                "dbpedia",
                                prob,
                                i,
                                kg_ent,  # type: ignore
                                count = count,
                            )

                            if debug:
                                ic("found", dbp_link)

                            yield dbp_link

                    ent_idx += 1

            tok_idx += tok.length


    def _link_kg_search_entities (
        self,
        graph: SimpleGraph,
        *,
        debug: bool = False,
        ) -> typing.Iterator[ LinkedEntity ]:
        """
Iterator for the results of using `DBPedia Search` directly for
_entity linking_.
        """
        node_list: list = list(graph.nodes.values())

        for i, node in enumerate(node_list):
            if node.kind in [ NodeEnum.ENT ] and len(node.entity) < 1:
                kg_ent: typing.Optional[ KGSearchHit ] = self.dbpedia_search_entity(  # type: ignore  # pylint: disable=C0301
                    node.text,
                    debug = debug,
                )

                if kg_ent.prob > self.min_alias:  # type: ignore
                    dbp_link: LinkedEntity = LinkedEntity(
                        node.span,
                        kg_ent.iri,  # type: ignore
                        node.length,
                        "dbpedia",
                        kg_ent.prob,  # type: ignore
                        i,
                        kg_ent,  # type: ignore
                    )

                    if debug:
                        ic("found", dbp_link)

                    yield dbp_link


    def _make_link (
        self,
        graph: SimpleGraph,
        pipe: Pipeline,
        link: LinkedEntity,
        rel: str,
        *,
        debug: bool = False,
        ) -> Node:
        """
Link to previously constructed entity node;
otherwise construct a new node for this linked entity.
        """
        if debug:
            ic(link)

        # special case of `make_node()`
        if link.iri in graph.nodes:
            graph.nodes[link.iri].count += 1

        else:
            graph.nodes[link.iri] = Node(
                len(graph.nodes),
                link.iri,
                link.span,
                link.kg_ent.descrip,
                rel,
                NodeEnum.IRI,
                label = link.iri,
                length = link.length,
                count = 1,
            )

        src_node: Node = pipe.tokens[link.token_id]
        src_node.annotated = True

        dst_node: Node = graph.nodes.get(link.iri)  # type: ignore

        if debug:
            ic(src_node, dst_node)

        # back-link to the parsed entity object
        pipe.tokens[link.token_id].entity.append(link)

        # construct a directed edge between them
        edge: Edge = graph.make_edge(  # type: ignore
            src_node,
            dst_node,
            RelEnum.IRI,
            rel,
            link.prob,
            debug = debug,
        )

        if debug:
            ic(edge)

        if edge is not None:
            pipe.edges.append(edge)

        # return the linked node
        return dst_node


    def _secondary_entity_linking (
        self,
        graph: SimpleGraph,
        pipe: Pipeline,
        link: LinkedEntity,
        *,
        debug: bool = False,
        ) -> typing.Optional[ Edge ]:
        """
Perform secondary _entity linking_, e.g., based on Wikidata API.
        """
        wd_ent: typing.Optional[ KGSearchHit ] = self.wikidata_search(  # type: ignore
            link.kg_ent.label,
            debug = debug,
        )

        if debug:
            ic(link.span, wd_ent)

        if wd_ent is not None and wd_ent.prob > self.min_similarity:
            wd_link: LinkedEntity = LinkedEntity(
                link.span,
                wd_ent.iri,
                len(link.span),
                "wikidata",
                wd_ent.prob,
                link.token_id,
                wd_ent,
            )

            if debug:
                ic(wd_link)

            src_node: Node = graph.nodes.get(link.iri)  # type: ignore

            dst_node: Node = self._make_link(
                graph,
                pipe,
                wd_link,
                self.REL_ISA,
                debug = debug,
            )

            # add an equivalency edge between the two linked entities
            edge: Edge = graph.make_edge(  # type: ignore
                src_node,
                dst_node,
                RelEnum.IRI,
                self.REL_SAME,
                wd_link.prob,
                debug = debug,
            )

            if edge is not None:
                pipe.edges.append(edge)

            # return the constructed edge
            return edge

        return None


if __name__ == "__main__":
    kg: KGWikiMedia = KGWikiMedia()

    ## resolve rel => iri
    rel_list: typing.List[ str ] = [
        "country of citizenship",
        "father",
        "child",
        "significant event",
        "child",
        "foo",
    ]

    for test_rel in rel_list:
        start_time: float = time.time()

        result: typing.Optional[ str ] = kg.resolve_rel_iri(
            test_rel,
            debug = True,
        )

        duration: float = round(time.time() - start_time, 3)

        ic(test_rel, result)
        print(f"resolve: {round(duration, 3)} sec")

    ## search DBPedia
    query_list: typing.List[ str ] = [
        "filmmaking",
        "filmmaker",
        "Werner Herzog",
        "Werner Herzog",
        "Werner",
        "Marlene Dietrich",
        "Dietrich",
        "America",
    ]

    for test_query in query_list:
        start_time = time.time()

        _kg_ent: KGSearchHit = kg.dbpedia_search_entity(  # type: ignore  # pylint: disable=W0212
            test_query,
            debug = True,
        )

        duration = round(time.time() - start_time, 3)

        ic(test_query, _kg_ent)
        print(f"lookup: {round(duration, 3)} sec")


    ## find Wikidata IRIs that correpond to DBPedia IRIs
    dbp_iri_list: typing.List[ str ] = [
        "http://dbpedia.org/resource/Filmmaking",
        "http://dbpedia.org/resource/Werner_Herzog",
        "http://dbpedia.org/resource/United_States",
    ]

    for dbp_iri in dbp_iri_list:
        start_time = time.time()

        wd_iri: str = kg.dbpedia_wikidata_equiv(  # pylint: disable=W0212
            kg.normalize_prefix(dbp_iri, debug = False),  # type: ignore
            debug = False,
        )

        duration = round(time.time() - start_time, 3)

        ic(dbp_iri, wd_iri)
        print(f"query: {round(duration, 3)} sec")
