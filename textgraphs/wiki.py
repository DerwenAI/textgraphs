#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This class provides a wrapper for MediaWiki API access, supporting
use of:

  * DBpedia
  * Wikidata
  * KBPedia (in progress)

... plus related machine learning models which derive from the exports
of these projects.

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

from .defaults import DBPEDIA_SEARCH_API, DBPEDIA_SPARQL_API, WIKIDATA_API
from .elem import WikiEntity


class WikiDatum:  # pylint: disable=R0903
    """
Manage access to MediaWiki-related APIs.
    """
    DBPEDIA_NS_PREFIX: typing.Dict[ str, str ] = OrderedDict({
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
        "wd": "https://www.wikidata.org/wiki/",
        "schema": "https://schema.org/",
    })


    def __init__ (
        self,
        ) -> None:
        """
Constructor.
        """
        self.ent_cache: dict = {}
        self.iri_cache: dict = {}
        self.markdowner = markdown2.Markdown()


    @classmethod
    def normalize_prefix (
        cls,
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

        for prefix, ns_fqdn in cls.DBPEDIA_NS_PREFIX.items():
            ns_parse: urllib.parse.ParseResult = urllib.parse.urlparse(ns_fqdn)

            if debug:
                ic(prefix, ns_parse.netloc, ns_parse.path)

            if iri_parse.netloc == ns_parse.netloc and iri_parse.path.startswith(ns_parse.path):
                slug: str = iri_parse.path.replace(ns_parse.path, "")

                # return normalized IRI
                return f"{prefix}:{slug}"

        # normalization failed
        return iri


    def resolve_wikidata_rel_iri (
        self,
        rel: str,
        wikidata_api: str,
        *,
        lang: str = "en",
        debug: bool = False,
        ) -> typing.Optional[ str ]:
        """
Resolve a `rel` string from one of the _relation extraction_ models
which has been trained on `Wikidata`.
        """
        # first, check the cache
        if rel in self.iri_cache:
            return self.iri_cache.get(rel)

        # otherwise construct a Wikidata API search
        params: dict = {
            "language": lang,
            "format": "json",
            "type": "property",
            "action": "wbsearchentities",
            "search": rel,
        }

        try:
            response: requests.models.Response = requests.get(
                wikidata_api,
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

            if debug:
                ic(len(dat["search"]))
                #ic(dat)
                #ic(dat["search"])

            # take the first hit -- generally the most relevant
            hit: dict = dat["search"][0]

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


    def dbpedia_search_entity (  # pylint: disable=R0914
        self,
        query: str,
        dbpedia_search_api: str,
        *,
        lang: str = "en",
        debug: bool = False,
        ) -> typing.Optional[ WikiEntity ]:
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
                dbpedia_search_api,
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

            ent: WikiEntity = WikiEntity(
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
        dbpedia_sparql_api: str = DBPEDIA_SPARQL_API,
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
                dbpedia_sparql_api,
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
        dbpedia_sparql_api: str = DBPEDIA_SPARQL_API,
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
            dbpedia_sparql_api = dbpedia_sparql_api,
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


if __name__ == "__main__":
    wiki: WikiDatum = WikiDatum()

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

        result: typing.Optional[ str ] = wiki.resolve_wikidata_rel_iri(
            test_rel,
            WIKIDATA_API,
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

        wiki_ent: WikiEntity = wiki.dbpedia_search_entity(  # type: ignore
            test_query,
            DBPEDIA_SEARCH_API,
            debug = True,
        )

        duration = round(time.time() - start_time, 3)

        ic(test_query, wiki_ent)
        print(f"lookup: {round(duration, 3)} sec")


    ## find Wikidata IRIs that correpond to DBPedia IRIs
    dbp_iri_list: typing.List[ str ] = [
        "http://dbpedia.org/resource/Filmmaking",
        "http://dbpedia.org/resource/Werner_Herzog",
        "http://dbpedia.org/resource/United_States",
    ]

    for dbp_iri in dbp_iri_list:
        start_time = time.time()

        wikid_iri: str = wiki.dbpedia_wikidata_equiv(
            wiki.normalize_prefix(dbp_iri, debug = False),  # type: ignore
            debug = False,
        )

        duration = round(time.time() - start_time, 3)

        ic(dbp_iri, wikid_iri)
        print(f"query: {round(duration, 3)} sec")
