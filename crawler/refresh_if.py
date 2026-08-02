#!/usr/bin/env python3
"""Refresh data/if_cache.json from the easyscholar getPublicationRank API.

    export EASYSCHOLAR_KEY=...      # free key from the easyscholar personal center
    python crawler/refresh_if.py

For every journal we index (non-preprint venues in data/entries.json) this looks
up the Clarivate JCR impact factor (officialRank.all.sciif) by name and caches it,
keyed by the exact venue string. store.publish() stamps from this cache offline,
so refreshing the numbers never touches the LLM. A journal easyscholar cannot
find is cached as null; any HTTP or auth error raises instead of guessing.
"""

import json
import os
import pathlib
import sys
import time

import requests

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import store

API = "https://www.easyscholar.cc/open/getPublicationRank"
UA = {"User-Agent": "spatial-omics-radar (+https://github.com/)"}


def query_name(venue):
    """easyscholar matches on the bare journal name, so drop a trailing
    '(Oxford, England)'- or ': subtitle'-style suffix but keep the casing."""
    return venue.split("(")[0].split(":")[0].strip()


def fetch_sciif(key, name):
    """Return the JCR impact factor for `name` as a float, or None if easyscholar
    has no record of the journal. Raises on HTTP or API-level (non-200) errors."""
    r = requests.get(API, params={"secretKey": key, "publicationName": name},
                     headers=UA, timeout=30)
    r.raise_for_status()
    payload = r.json()
    if payload.get("code") != 200:
        raise RuntimeError(f"easyscholar error for {name!r}: "
                           f"{payload.get('code')} {payload.get('msg')}")
    official = ((payload.get("data") or {}).get("officialRank") or {}).get("all")
    if not official:                    # journal genuinely not found -> null
        return None
    sciif = official.get("sciif")
    return float(sciif) if sciif not in (None, "") else None


def main():
    key = os.environ.get("EASYSCHOLAR_KEY")
    if not key:
        sys.exit("EASYSCHOLAR_KEY not set — run `export EASYSCHOLAR_KEY=...` first.")

    entries = store.load_auto()
    venues = sorted({e["venue"] for e in entries.values()
                     if e.get("venue") and not e.get("is_preprint")})
    print(f"{len(venues)} distinct published journals to look up")

    by_query = {}          # cleaned name -> sciif, so each journal is hit once
    cache = {}
    for v in venues:
        q = query_name(v)
        if q not in by_query:
            by_query[q] = fetch_sciif(key, q)
            print(f"  {by_query[q]!s:>7}  {q}")
            time.sleep(0.3)   # be polite to a free service
        cache[v] = by_query[q]

    store.IF_CACHE.write_text(
        json.dumps(cache, indent=2, ensure_ascii=False, sort_keys=True))
    found = sum(1 for x in cache.values() if x is not None)
    print(f"wrote {store.IF_CACHE.relative_to(store.ROOT)}: "
          f"{found}/{len(cache)} journals with an impact factor")


if __name__ == "__main__":
    main()
