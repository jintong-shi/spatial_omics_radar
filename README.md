# Spatial Omics Radar

An auto-updating index of technologies, tools, AI models and benchmarks across
**all spatial omics modalities** — transcriptomics, proteomics, metabolomics,
epigenomics and multi-omics. A weekly GitHub Action queries Europe PMC and
GitHub, triages the hits, and commits the result. The site is a single static
HTML file served straight from `docs/` by GitHub Pages — no build step, no
framework.

Databases, portals and data repositories are deliberately out of scope.

```
crawler/config.yml   queries, keyword rules, controlled vocabulary  ← you edit this
crawler/sources.py   Europe PMC + GitHub fetchers
crawler/triage.py    stage 1 keyword prefilter, stage 2 LLM classifier
crawler/store.py     merges auto data with your manual corrections
crawler/run.py       CLI entry point
data/entries.json    machine-owned. Overwritten every crawl.
data/overrides.json  human-owned. Never touched by a crawl.  ← you edit this
docs/index.html      the site
docs/entries.json    merged output the site reads
```

## Why two stages

Every Europe PMC query returns mostly application papers that merely *used*
spatial data. Stage 1 is pure string matching — free, deterministic, tuned for
recall — and kills roughly 90% of hits. Only survivors reach the LLM, which
decides whether the record really introduces a resource and extracts structured
fields. Running the LLM on raw query output instead would cost ~10× more and be
no more accurate.

## Setup

1. Create the repo, push this tree.
2. **Settings → Pages** → Source: *Deploy from a branch*, Branch: `main`, Folder: `/docs`.
3. **Settings → Secrets and variables → Actions** → add `ANTHROPIC_API_KEY`.
4. Edit `crawler/config.yml`: set `site.repo_url`, adjust queries.
5. Backfill locally before letting the Action loose:

```bash
pip install -r requirements.txt

# See what the queries and prefilter catch. No API calls, no cost.
python crawler/run.py --since 2024-01-01 --dry-run

# Once the dry-run list looks sane, classify a small batch and inspect it.
export ANTHROPIC_API_KEY=sk-...
python crawler/run.py --since 2024-01-01 --limit 10
python -m http.server -d docs 8000     # open localhost:8000

# Happy? Do the full backfill.
python crawler/run.py --since 2023-01-01
git add data docs && git commit -m "backfill" && git push
```

The Action then runs every Monday and pushes incrementally (last 45 days).

## Tuning

**Too much junk getting through.** Add terms to `prefilter.reject`. Raise the
`needs_review` threshold in `run.py` so more entries show the *unreviewed* tag.

**Missing tools you know exist.** Almost always the query, not the filter. Test
a query directly at <https://europepmc.org/search> first, then paste it in.
`--dry-run` shows you what changed.

**Tags drifting.** Do not let the LLM invent categories — it will produce 200
near-synonyms within a month. `triage.sanitise()` discards anything outside
`vocab`, so widen `vocab` deliberately rather than loosening the prompt.

**`bioinformatics tool` vs `AI model` applied inconsistently.** These two are
deliberately non-exclusive: a task-specific deep-learning method that ships as
a package is both. If the split drifts, edit the decision rule inside
`triage.PROMPT`, not `vocab`.

**Too few non-transcriptomics entries.** The single most important metric for
this project. Tune query #2 and the proteomics/metabolomics/epigenomics blocks
of `prefilter.anchors`. Check the distribution after every backfill:

```bash
python -c "import json,collections; e=json.load(open('docs/entries.json'))['entries']; \
print(collections.Counter(m for x in e for m in x['modality']))"
```

## Fixing entries

Never hand-edit `data/entries.json` — the next crawl overwrites it. Use
`data/overrides.json`:

```json
{
  "patch": { "10.1038/s41592-021-01255-8": { "kind": ["bioinformatics tool", "AI model"] } },
  "hide":  [ "10.1038/some-false-positive" ],
  "add":   { "manual:xenium-5k": { "id": "manual:xenium-5k", "name": "Xenium 5K", "...": "" } }
}
```

`patch` merges over an auto entry, `hide` suppresses one, `add` injects something
the crawler cannot see (unpublished tools, GitHub-only projects, wet-lab
protocols). Anything touched this way is marked `curated: true` and survives
every future crawl. Keys are the `id` field shown in `docs/entries.json`.

## Cost

Steady state is roughly 45–90 records reaching the LLM per week, at ~1,800
input and ~150 output tokens each. On Haiku 4.5 that is order **$0.50–1.00 per
month**; a two-year backfill is a one-off **$5–12**. `--limit` caps calls per run so a runaway query cannot surprise you.
Europe PMC and GitHub search are free; set `GITHUB_TOKEN` to lift GitHub's
search rate limit from 10 to 30 requests/minute.

## Known gaps

- Recall for non-transcriptomics modalities is unverified. If proteomics and
  metabolomics counts stay in the single digits, the premise of this project
  does not hold — check the distribution before promoting the site.
- Vendor technology launches (10x, NanoString, Vizgen, Bruker) often have no
  paper for months. Add them via `overrides.json`.
- Stage 2 has not been load-tested at scale; watch the first full backfill for
  JSON parse failures, which are logged but skipped rather than retried.
- `data/entries.json` is a flat JSON dict. Past ~5k entries you will want to
  shard it by year, and the site will want pagination.
