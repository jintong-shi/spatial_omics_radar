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

FEED_MAX = 50   # most-recently-indexed entries to expose in the RSS feed


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


def _rfc822(d):
    """YYYY-MM-DD -> RFC 822 date (RSS pubDate). Empty string if unparseable."""
    try:
        dt = datetime.datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=datetime.timezone.utc)
        return email.utils.format_datetime(dt)
    except (ValueError, TypeError):
        return ""


def _write_feed(entries, meta):
    """Write a static RSS 2.0 feed of the most-recently-indexed entries, so
    anyone can subscribe from their own Slack or Teams (`/feed subscribe <url>`)
    with no backend. Ordered by `added` so this week's new entries are always
    present; Slack de-dupes on <guid> and only pushes ones it hasn't seen."""
    site = meta.get("site", {})
    esc = xml.sax.saxutils.escape
    now = datetime.datetime.now(datetime.timezone.utc)
    recent = (now - datetime.timedelta(days=7)).date().isoformat()
    items = sorted(entries, key=lambda e: (e.get("added") or e.get("date") or ""),
                   reverse=True)[:FEED_MAX]
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<rss version="2.0"><channel>',
        f'<title>{esc(site.get("title") or "Spatial Omics Radar")}</title>',
        f'<link>{esc(site.get("repo_url") or "")}</link>',
        f'<description>{esc(site.get("subtitle") or "New tools, assays and benchmarks across spatial omics")}</description>',
        f'<lastBuildDate>{email.utils.format_datetime(datetime.datetime.now(datetime.timezone.utc))}</lastBuildDate>',
    ]
    for e in items:
        tags = " · ".join([*(e.get("kind") or []), *(e.get("modality") or [])])
        venue = "preprint" if e.get("is_preprint") else (e.get("venue") or "")
        desc = e.get("one_liner") or e.get("title") or ""
        meta_bits = " — ".join(x for x in [tags, venue] if x)
        if meta_bits:
            desc = f"{desc} ({meta_bits})"
        added = e.get("added") or e.get("date") or ""
        # Slack's RSS app decides "new" by pubDate, so a recently-indexed entry
        # dated at midnight can read as older than the subscription and never get
        # pushed. Stamp entries indexed in the last 7 days with the build time so
        # they register as fresh; older ones keep their date and aren't re-pushed.
        pub = email.utils.format_datetime(now) if added >= recent else _rfc822(added)
        parts += [
            '<item>',
            f'<title>{esc(e.get("name") or e.get("title") or "untitled")}</title>',
            f'<link>{esc(e.get("url") or "")}</link>',
            f'<guid isPermaLink="false">{esc(e.get("id") or "")}</guid>',
            f'<pubDate>{pub}</pubDate>',
            f'<description>{esc(desc)}</description>',
            '</item>',
        ]
    parts.append('</channel></rss>')
    FEED.write_text("\n".join(parts), encoding="utf-8")
