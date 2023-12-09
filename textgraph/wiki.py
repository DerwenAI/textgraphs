#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MediaWiki API access.
"""

import http
import time
import typing

from icecream import ic  # pylint: disable=E0401
from qwikidata.linked_data_interface import get_entity_dict_from_api  # pylint: disable=E0401
import requests  # type: ignore  # pylint: disable=E0401


class WikiDatum:  # pylint: disable=R0903
    """
Manage access to MediaWiki API.
    """

    def __init__ (
        self,
        ) -> None:
        """
Constructor.
        """
        self.iri_cache: dict = {}


    def resolve_iri (
        self,
        rel: str,
        *,
        wikidata_api: str = "https://www.wikidata.org/w/api.php",
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
            "search": rel.replace(" ", "+"),
        }

        get_enc: str = "&".join([
            f"{key}={val}"
            for key, val in params.items()
        ])

        url: str = f"{wikidata_api}?{get_enc}"

        if debug:
            ic(rel, url)

        response: requests.models.Response = requests.get(url)

        if debug:
            ic(response.status_code)

        # check for failed API calls
        if http.HTTPStatus.OK != response.status_code:
            return None

        dat: dict = response.json()

        if debug:
            ic(len(dat["search"]))

        # take the first hit, which is generally the most relevant
        hit: dict = dat["search"][0]

        if debug:
            ic(hit["label"], hit["id"])

        # get the `claims` of a Wikidata property
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


if __name__ == "__main__":
    wiki: WikiDatum = WikiDatum()

    rels: typing.List[ str ] = [
        "country of citizenship",
        "father",
        "child",
        "significant event",
        "child",
        "foo",
    ]

    for test_rel in rels:
        start_time: float = time.time()

        result: typing.Optional[ str ] = wiki.resolve_iri(
            test_rel,
            debug = True,
        )

        duration: float = round(time.time() - start_time, 3)

        ic(test_rel, result)
        print(f"lookup: {round(duration, 3)} sec")
