#!/usr/bin/env python3
"""Run one crawl cycle.

    python crawler/run.py --since 2024-01-01     # first backfill
    python crawler/run.py                        # incremental (last 45 days)
    python crawler/run.py --dry-run              # fetch + prefilter only, no LLM
    python crawler/run.py --backend cli          # bill the LLM to your subscription
"""

import argparse
import datetime as dt
import pathlib
import sys
import time

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
    ap.add_argument("--backend", choices=["api", "cli"],
                    help="api = Messages API billed to ANTHROPIC_API_KEY; "
                         "cli = `claude -p` billed to your Pro/Max subscription. "
                         "Default comes from config.yml (llm.backend).")
    ap.add_argument("--model", help="override llm.model from config.yml "
                                    "(e.g. claude-sonnet-5 for an accuracy A/B)")
    args = ap.parse_args()

    cfg = yaml.safe_load(CONFIG.read_text())
    backend = args.backend or cfg["llm"].get("backend", "api")
    model = args.model or cfg["llm"]["model"]
    since = args.since or (dt.date.today() - dt.timedelta(days=45)).isoformat()
    known = store.load_auto()
    print(f"crawling since {since}; {len(known)} entries already known "
          f"(llm backend: {backend}, model: {model})")

    # ---- fetch -------------------------------------------------------------
    t = time.perf_counter()
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
    t_fetch = time.perf_counter() - t
    print(f"{len(raw)} unique records, {len(fresh)} not seen before")

    # ---- stage 1 -----------------------------------------------------------
    t = time.perf_counter()
    passed = [r for r in fresh.values() if triage.prefilter(r, cfg["prefilter"])]
    t_prefilter = time.perf_counter() - t
    n_prefilter = len(passed)
    print(f"prefilter kept {len(passed)}/{len(fresh)}")

    if args.dry_run:
        for r in passed[:40]:
            print(f'  · {r["date"]}  {r["title"][:88]}')
        return

    if args.limit:
        passed = passed[: args.limit]

    # ---- stage 2 -----------------------------------------------------------
    t = time.perf_counter()
    kept = n_calls = tok_in = tok_out = 0
    cost = 0.0
    for i, rec in enumerate(passed, 1):
        if not cfg["llm"]["enabled"]:
            verdict = {"keep": cfg["llm"]["fallback_keep_all"], "confidence": 0.0,
                       "reason": "llm disabled", "kind": [], "modality": [],
                       "platforms": []}
        else:
            classifier = triage.classify_via_cli if backend == "cli" else triage.classify
            verdict, usage = classifier(rec, cfg["vocab"], model,
                                        cfg.get("learned_rules") or [])
            n_calls += 1
            # `input_tokens` alone is only the un-cached delta. With prompt
            # caching (always on in the CLI backend) the bulk of the real
            # input lands in the cache_* fields, so summing just input_tokens
            # under-reports input by ~100x. `total_cost_usd` (cost) is computed
            # by Claude Code from the full usage and is the trustworthy figure.
            tok_in += (usage.get("input_tokens", 0)
                       + usage.get("cache_creation_input_tokens", 0)
                       + usage.get("cache_read_input_tokens", 0))
            tok_out += usage.get("output_tokens", 0)
            cost += usage.get("cost_usd") or 0
            if verdict is None:
                continue
            verdict = triage.sanitise(verdict, cfg["vocab"])

        if not verdict.get("keep"):
            continue

        known[rec["source_id"]] = {
            "id": rec["source_id"],
            "name": verdict.get("name") or rec["title"],
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
    t_llm = time.perf_counter() - t

    store.save_auto(known)
    total = store.publish(known, {
        "updated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "site": cfg["site"],
        "vocab": cfg["vocab"],
    })
    print(f"added {kept}; published {total} entries")

    # ---- metrics report ----------------------------------------------------
    print("\n── crawl metrics ──────────────────────────────────")
    print(f"  fetch      {len(raw):5d} records ({len(fresh)} new)      {t_fetch:7.1f}s")
    print(f"  prefilter  {n_prefilter:5d} kept                       {t_prefilter:7.1f}s")
    print(f"  LLM [{backend}] {n_calls:5d} calls ({kept} kept)         {t_llm:7.1f}s")
    print(f"             tokens: in {tok_in:,} (incl. cached)  out {tok_out:,}")
    if cost:
        print(f"             api-equivalent cost: ${cost:.4f}")


if __name__ == "__main__":
    main()
