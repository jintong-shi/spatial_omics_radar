"""Persistence.

Three files, deliberately:
  data/entries.json     auto-crawled, machine-owned, freely overwritten
  data/overrides.json   hand-edited, human-owned, NEVER overwritten by a crawl
  docs/entries.json     the merged view the website reads

Editing overrides.json is how you fix a wrong tag, add a tool the crawler
missed, or blacklist a false positive. Because entries are keyed by source_id,
your fix survives every future crawl.
"""

import datetime
import email.utils
import json
import pathlib
import re
import xml.sax.saxutils

ROOT = pathlib.Path(__file__).resolve().parent.parent
AUTO = ROOT / "data" / "entries.json"
OVERRIDES = ROOT / "data" / "overrides.json"
IF_CACHE = ROOT / "data" / "if_cache.json"
PUBLISHED = ROOT / "docs" / "entries.json"
FEED = ROOT / "docs" / "feed.xml"
SEEN = ROOT / "data" / "seen.json"


def _read(path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text())


def _norm_journal(name):
    """Loose key so 'Bioinformatics (Oxford, England)' matches 'Bioinformatics'."""
    s = (name or "").lower().split("(")[0].split(":")[0]
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", s)).strip()


def _impact_factors():
    """Journal -> JIF map from the easyscholar cache (crawler/refresh_if.py),
    keyed loosely. Null entries (journals easyscholar can't find) fall out here
    because only numbers are kept."""
    return {_norm_journal(k): v for k, v in _read(IF_CACHE, {}).items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)}


def load_auto():
    return _read(AUTO, {})


def save_auto(entries):
    AUTO.write_text(json.dumps(entries, indent=2, ensure_ascii=False, sort_keys=True))


def load_seen():
    """Source_ids the LLM has already judged (keep OR reject). Kept apart from
    entries.json — which still holds only kept entries — so a chunked or resumed
    backfill never re-classifies a record it has already ruled on."""
    return set(_read(SEEN, []))


def save_seen(seen):
    SEEN.write_text(json.dumps(sorted(seen), indent=2, ensure_ascii=False))


def publish(auto, meta):
    """Apply overrides on top of auto data and write what the site consumes."""
    ov = _read(OVERRIDES, {"patch": {}, "hide": [], "add": {}})
    # Copy each entry so stamping impact_factor below never mutates the
    # machine-owned auto dict the caller may still be holding.
    merged = {sid: dict(e) for sid, e in auto.items()}

    for sid, patch in ov.get("patch", {}).items():
        if sid in merged:
            merged[sid] = {**merged[sid], **patch, "curated": True}

    merged.update({sid: {**e, "curated": True} for sid, e in ov.get("add", {}).items()})

    for sid in ov.get("hide", []):
        merged.pop(sid, None)

    # Enrich published papers with their journal impact factor (preprints get none).
    ifs = _impact_factors()
    for e in merged.values():
        jif = None if e.get("is_preprint") else ifs.get(_norm_journal(e.get("venue")))
        if jif is not None:
            e["impact_factor"] = jif

    rows = sorted(merged.values(), key=lambda e: e.get("date", ""), reverse=True)
    PUBLISHED.write_text(json.dumps(
        {"meta": {**meta, "count": len(rows)}, "entries": rows},
        indent=2, ensure_ascii=False))
    _write_feed(merged.values(), meta)
    return len(rows)


def _write_feed(entries, meta):
    """Write a static RSS 2.0 feed with a SINGLE item: a weekly digest of the
    entries indexed in the last 7 days. Slack's RSS app posts one message per new
    <item> and de-dupes on <guid>, so a guid that is stable within an ISO week
    yields exactly one Slack message per week. The item links back to the site
    filtered to that week (?since=<date>). `meta["weekly"]["highlight"]` is an
    optional LLM-written blurb; without it the item lists the week's names."""
    site = meta.get("site", {})
    weekly = meta.get("weekly") or {}
    esc = xml.sax.saxutils.escape
    now = datetime.datetime.now(datetime.timezone.utc)
    since = weekly.get("since") or (now - datetime.timedelta(days=7)).date().isoformat()
    week = sorted((e for e in entries if (e.get("added") or "") >= since),
                  key=lambda e: (e.get("added") or e.get("date") or ""), reverse=True)
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<rss version="2.0"><channel>',
        f'<title>{esc(site.get("title") or "Spatial Omics Radar")}</title>',
        f'<link>{esc(site.get("url") or site.get("repo_url") or "")}</link>',
        f'<description>{esc(site.get("subtitle") or "New tools, assays and benchmarks across spatial omics")}</description>',
        f'<lastBuildDate>{email.utils.format_datetime(now)}</lastBuildDate>',
    ]
    # No new entries this week -> emit an empty channel (no Slack message).
    if week:
        # Key the guid on the current ISO week: stable for any re-run within the
        # week (Slack de-dupes -> one message), rolls over exactly once a week.
        iso = now.isocalendar()
        wid = f"weekly-{iso[0]}-W{iso[1]:02d}"
        base = (site.get("url") or "").rstrip("/")
        link = f"{base}/?since={since}" if base else ""
        names = ", ".join(e.get("name") or e.get("title") or "untitled" for e in week[:8])
        if len(week) > 8:
            names += f", +{len(week) - 8} more"
        # Fall back to a plain name list if the LLM highlight is missing/failed:
        # real data, never fabricated.
        blurb = (weekly.get("highlight") or "").strip() or f"New this week: {names}."
        title = f"Weekly update · {len(week)} new " + ("entry" if len(week) == 1 else "entries")
        parts += [
            '<item>',
            f'<title>{esc(title)}</title>',
            f'<link>{esc(link)}</link>',
            f'<guid isPermaLink="false">{esc(wid)}</guid>',
            f'<pubDate>{email.utils.format_datetime(now)}</pubDate>',
            f'<description>{esc(blurb)}</description>',
            '</item>',
        ]
    parts.append('</channel></rss>')
    FEED.write_text("\n".join(parts), encoding="utf-8")
