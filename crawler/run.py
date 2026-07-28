#!/usr/bin/env python3
"""Run one crawl cycle.

    python crawler/run.py --since 2024-01-01     # first backfill
    python crawler/run.py                        # incremental (last 45 days)
    python crawler/run.py --dry-run              # fetch + prefilter only, no LLM
"""

import argparse
import datetime as dt
import pathlib
import sys

import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import sources
import store
import triage

CONFIG = pathlib.Path(__file__).resolve().parent / "config.yml"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", help="YYYY-MM-DD; default = 45 days ago")
    ap.add_argument("--dry-run", action="store_true",
                    help="skip the LLM; report what the prefilter caught")
    ap.add_argument("--limit", type=int, help="cap LLM calls (cost guard)")
    args = ap.parse_args()

    cfg = yaml.safe_load(CONFIG.read_text())
    since = args.since or (dt.date.today() - dt.timedelta(days=45)).isoformat()
    known = store.load_auto()
    print(f"crawling since {since}; {len(known)} entries already known")

    # ---- fetch -------------------------------------------------------------
    raw = {}
    for q in cfg["europepmc_queries"]:
        hits = sources.fetch_europepmc(q, since)
        print(f"  europepmc  {len(hits):4d}  {q[:60]}...")
        for h in hits:
            raw.setdefault(h["source_id"], h)
    for q in cfg["github_queries"]:
        hits = sources.fetch_github(q, since)
        print(f"  github     {len(hits):4d}  {q[:60]}...")
        for h in hits:
            raw.setdefault(h["source_id"], h)

    fresh = {k: v for k, v in raw.items() if k not in known}
    print(f"{len(raw)} unique records, {len(fresh)} not seen before")

    # ---- stage 1 -----------------------------------------------------------
    passed = [r for r in fresh.values() if triage.prefilter(r, cfg["prefilter"])]
    print(f"prefilter kept {len(passed)}/{len(fresh)}")

    if args.dry_run:
        for r in passed[:40]:
            print(f'  · {r["date"]}  {r["title"][:88]}')
        return

    if args.limit:
        passed = passed[: args.limit]

    # ---- stage 2 -----------------------------------------------------------
    kept = 0
    for i, rec in enumerate(passed, 1):
        if not cfg["llm"]["enabled"]:
            verdict = {"keep": cfg["llm"]["fallback_keep_all"], "confidence": 0.0,
                       "reason": "llm disabled", "kind": [], "modality": [],
                       "platforms": []}
        else:
            verdict = triage.classify(rec, cfg["vocab"], cfg["llm"]["model"])
            if verdict is None:
                continue
            verdict = triage.sanitise(verdict, cfg["vocab"])

        if not verdict.get("keep"):
            continue

        known[rec["source_id"]] = {
            "id": rec["source_id"],
            "name": verdict.get("name") or rec["title"][:60],
            "title": rec["title"],
            "one_liner": verdict.get("one_liner", ""),
            "kind": verdict.get("kind", []),
            "modality": verdict.get("modality", []),
            "platforms": verdict.get("platforms", []),
            "url": rec["url"],
            "code_url": rec.get("code_url"),
            "venue": rec["venue"],
            "date": rec["date"],
            "is_preprint": rec.get("is_preprint", False),
            "stars": rec.get("stars"),
            "confidence": verdict.get("confidence"),
            "needs_review": (verdict.get("confidence") or 0) < 0.7,
            "added": dt.date.today().isoformat(),
            "curated": False,
        }
        kept += 1
        print(f'  [{i}/{len(passed)}] + {known[rec["source_id"]]["name"]}')

    store.save_auto(known)
    total = store.publish(known, {
        "updated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "site": cfg["site"],
        "vocab": cfg["vocab"],
    })
    print(f"added {kept}; published {total} entries")


if __name__ == "__main__":
    main()
