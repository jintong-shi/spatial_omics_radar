"""Persistence.

Three files, deliberately:
  data/entries.json     auto-crawled, machine-owned, freely overwritten
  data/overrides.json   hand-edited, human-owned, NEVER overwritten by a crawl
  docs/entries.json     the merged view the website reads

Editing overrides.json is how you fix a wrong tag, add a tool the crawler
missed, or blacklist a false positive. Because entries are keyed by source_id,
your fix survives every future crawl.
"""

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
AUTO = ROOT / "data" / "entries.json"
OVERRIDES = ROOT / "data" / "overrides.json"
PUBLISHED = ROOT / "docs" / "entries.json"


def _read(path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text())


def load_auto():
    return _read(AUTO, {})


def save_auto(entries):
    AUTO.write_text(json.dumps(entries, indent=2, ensure_ascii=False, sort_keys=True))


def publish(auto, meta):
    """Apply overrides on top of auto data and write what the site consumes."""
    ov = _read(OVERRIDES, {"patch": {}, "hide": [], "add": {}})
    merged = dict(auto)

    for sid, patch in ov.get("patch", {}).items():
        if sid in merged:
            merged[sid] = {**merged[sid], **patch, "curated": True}

    merged.update({sid: {**e, "curated": True} for sid, e in ov.get("add", {}).items()})

    for sid in ov.get("hide", []):
        merged.pop(sid, None)

    rows = sorted(merged.values(), key=lambda e: e.get("date", ""), reverse=True)
    PUBLISHED.write_text(json.dumps(
        {"meta": {**meta, "count": len(rows)}, "entries": rows},
        indent=2, ensure_ascii=False))
    return len(rows)
