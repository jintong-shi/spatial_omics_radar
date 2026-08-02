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
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
AUTO = ROOT / "data" / "entries.json"
OVERRIDES = ROOT / "data" / "overrides.json"
IF_CACHE = ROOT / "data" / "if_cache.json"
PUBLISHED = ROOT / "docs" / "entries.json"


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
    return len(rows)
