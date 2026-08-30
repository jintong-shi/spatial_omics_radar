"""Persistence.

Three files, deliberately:
  data/entries.json     auto-crawled, machine-owned, freely overwritten
  data/overrides.json   hand-edited, human-owned, NEVER overwritten by a crawl
  docs/entries.json     the merged view the website reads

Editing overrides.json is how you fix a wrong tag, add a tool the crawler
missed, or blacklist a false positive. Because entries are keyed by source_id,
your fix survives every future crawl.
"""

import collections
import datetime
import email.utils
import html
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
EMAIL_FEED = ROOT / "docs" / "feed_email.xml"
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
    _write_email_feed(merged.values(), meta)
    return len(rows)


def _write_feed(entries, meta):
    """Write a static RSS 2.0 feed with a SINGLE item: a weekly digest of the
    entries indexed in the last 7 days. Slack's RSS app posts one message per new
    <item> and de-dupes on <guid>, so a guid that is stable within an ISO week
    yields exactly one Slack message per week. The item links back to the site
    filtered to that week (?since=<date>). The item body is a plain factual line:
    the entry count plus which omics fields the week touched (most-common first)."""
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
        n = len(week)
        noun = "entry" if n == 1 else "entries"
        title = f"Weekly update · {n} new {noun}"
        # Body: count + the omics fields this week touched, most-common first.
        # Real data only. If a week has >4 distinct fields, tail-collapse the rest.
        counts = collections.Counter(m for e in week for m in (e.get("modality") or []))
        fields = [m for m, _ in counts.most_common()]
        if len(fields) > 4:
            fields = fields[:4] + [f"{len(fields) - 4} more"]
        if not fields:
            blurb = f"{n} new {noun} this week."
        elif len(fields) == 1:
            blurb = f"{n} new {noun} this week, spanning {fields[0]}."
        elif len(fields) == 2:
            blurb = f"{n} new {noun} this week, spanning {fields[0]} and {fields[1]}."
        else:
            blurb = f"{n} new {noun} this week, spanning " + ", ".join(fields[:-1]) + f", and {fields[-1]}."
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


def _write_email_feed(entries, meta):
    """Write a second static RSS 2.0 feed for the EMAIL channel: a single weekly
    item whose body lists every entry indexed in the last 7 days (name, one-liner,
    link), one block per entry. An RSS-to-email service (e.g. follow.it) turns that
    one item into one email per week that a reader scrolls entry by entry. This is
    deliberately separate from feed.xml, which stays a one-line digest for Slack;
    the two feeds share the same weekly window (meta['weekly']['since'])."""
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
        f'<title>{esc((site.get("title") or "Spatial Omics Radar") + " — email digest")}</title>',
        f'<link>{esc(site.get("url") or site.get("repo_url") or "")}</link>',
        f'<description>{esc(site.get("subtitle") or "New tools, assays and benchmarks across spatial omics")}</description>',
        f'<lastBuildDate>{email.utils.format_datetime(now)}</lastBuildDate>',
    ]
    # No new entries this week -> emit an empty channel (no email).
    if week:
        # Same ISO-week guid scheme as feed.xml (distinct prefix) -> the RSS-to-email
        # service de-dupes to exactly one email per week, rolling over once a week.
        iso = now.isocalendar()
        wid = f"weekly-email-{iso[0]}-W{iso[1]:02d}"
        base = (site.get("url") or "").rstrip("/")
        link = f"{base}/?since={since}" if base else ""
        n = len(week)
        noun = "entry" if n == 1 else "entries"
        title = f"Weekly update · {n} new {noun}"
        # HTML body, one block per entry, CDATA-wrapped so the email client renders
        # it (links, line breaks) rather than showing raw tags. Every entry field is
        # html.escape'd, so a stray '<' or '>' in a title cannot break out -- which
        # also means the CDATA-closing sequence ']]>' can never appear in the body.
        blocks = []
        for e in week:
            name = html.escape(e.get("name") or e.get("title") or "Untitled")
            url = html.escape(e.get("url") or "", quote=True)
            one = html.escape(e.get("one_liner") or "")
            heading = f'<a href="{url}">{name}</a>' if url else name
            blocks.append(f"<p><strong>{heading}</strong><br>{one}</p>")
        body = "\n".join(blocks)
        parts += [
            '<item>',
            f'<title>{esc(title)}</title>',
            f'<link>{esc(link)}</link>',
            f'<guid isPermaLink="false">{esc(wid)}</guid>',
            f'<pubDate>{email.utils.format_datetime(now)}</pubDate>',
            f'<description><![CDATA[{body}]]></description>',
            '</item>',
        ]
    parts.append('</channel></rss>')
    EMAIL_FEED.write_text("\n".join(parts), encoding="utf-8")
