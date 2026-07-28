"""Fetch candidate records from Europe PMC and GitHub.

Both functions return a list of raw dicts in a common shape:
    {source, source_id, title, text, url, code_url, venue, date}
`text` is whatever prose we have (abstract / README excerpt) and is what the
prefilter and the LLM read.
"""

import os
import time
import datetime as dt

import requests

EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
GH = "https://api.github.com/search/repositories"
UA = {"User-Agent": "spatial-omics-radar (+https://github.com/)"}


def fetch_europepmc(query, since, page_size=100, max_pages=30):
    """Europe PMC covers PubMed *and* bioRxiv/medRxiv preprints in one index.

    `since` is a date string 'YYYY-MM-DD'; we filter on first publication date
    so a paper does not reappear when it moves from preprint to journal.

    max_pages caps the crawl at max_pages * page_size records per query. 30 is
    sized for a multi-year backfill; incremental runs never come close.
    """
    today = dt.date.today().isoformat()
    scoped = f'({query}) AND (FIRST_PDATE:[{since} TO {today}])'
    out, cursor, pages = [], "*", 0

    while pages < max_pages:
        r = requests.get(
            EPMC,
            params={
                "query": scoped,
                "format": "json",
                "pageSize": page_size,
                "cursorMark": cursor,
                "resultType": "core",  # includes abstractText
            },
            headers=UA,
            timeout=60,
        )
        r.raise_for_status()
        payload = r.json()
        results = payload.get("resultList", {}).get("result", [])
        if not results:
            break

        for item in results:
            doi = item.get("doi")
            out.append(
                {
                    "source": "europepmc",
                    "source_id": doi or f'{item.get("source")}:{item.get("id")}',
                    "title": (item.get("title") or "").rstrip(". "),
                    "text": item.get("abstractText") or "",
                    "url": f"https://doi.org/{doi}" if doi else
                           f'https://europepmc.org/article/{item.get("source")}/{item.get("id")}',
                    "code_url": None,
                    "venue": item.get("journalTitle") or item.get("source") or "",
                    "date": item.get("firstPublicationDate") or "",
                    "is_preprint": item.get("source") == "PPR",
                }
            )

        next_cursor = payload.get("nextCursorMark")
        if not next_cursor or next_cursor == cursor:
            break
        cursor, pages = next_cursor, pages + 1
        time.sleep(0.4)  # be polite; EPMC asks for <10 req/s

    return out


def fetch_github(query, since, per_page=50):
    """GitHub repo search, restricted to repos pushed since `since`.

    Set GITHUB_TOKEN to get 30 req/min instead of 10.
    """
    headers = dict(UA, Accept="application/vnd.github+json")
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    r = requests.get(
        GH,
        params={"q": f"{query} pushed:>={since}", "sort": "updated",
                "order": "desc", "per_page": per_page},
        headers=headers,
        timeout=60,
    )
    r.raise_for_status()

    out = []
    for repo in r.json().get("items", []):
        out.append(
            {
                "source": "github",
                "source_id": repo["full_name"],
                "title": repo["name"],
                "text": repo.get("description") or "",
                "url": repo["html_url"],
                "code_url": repo["html_url"],
                "venue": "GitHub",
                "date": (repo.get("created_at") or "")[:10],
                "is_preprint": False,
                "stars": repo.get("stargazers_count", 0),
            }
        )
    return out
