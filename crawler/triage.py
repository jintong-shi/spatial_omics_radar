"""Two-stage triage.

Stage 1 (`prefilter`) is pure string matching: free, deterministic, tuned for
recall. Stage 2 (`classify`) sends only survivors to an LLM, which decides
whether the record is really a new resource and extracts structured fields
from the controlled vocabulary.

Roughly 90% of raw hits die in stage 1, which is what keeps the API bill flat.
"""

import json
import os
import shutil
import subprocess

import requests

API = "https://api.anthropic.com/v1/messages"

PROMPT = """You are triaging literature for a catalogue of spatial omics resources.

Decide whether this record introduces a NEW, USABLE RESOURCE across any spatial \
modality — transcriptomics, proteomics, metabolomics, epigenomics or multi-omics.

REJECT:
- application papers whose contribution is a biological finding, even if they \
used spatial data heavily
- reviews, perspectives, commentaries
- databases, web portals, data repositories and atlases (out of scope)

{rules_block}Controlled vocabulary — you MUST choose from these exact strings.

kind (1-2 values; these are NOT mutually exclusive): {kind}
  technology          = wet-lab assay, platform or protocol
  bioinformatics tool = installable software, pipeline or statistical method
  AI model            = the core contribution is a machine-learned model
  benchmark           = comparative evaluation of existing methods
  A task-specific deep-learning method that ships as a package is BOTH
  "bioinformatics tool" and "AI model". Plain software with no learned
  component is only "bioinformatics tool". A pretrained foundation model is
  "AI model", plus "bioinformatics tool" if it ships usable software.

modality (1-3 values): {modality}
platform (0-3 values, only if genuinely platform-specific): {platform}

Return ONLY a JSON object, no prose and no markdown fences:
{{"keep": true|false,
  "reason": "<=15 words",
  "name": "resource name, e.g. Squidpy (null if none is named)",
  "kind": ["1-2 values"],
  "modality": ["1-3 values"],
  "platforms": ["0-3 values"],
  "one_liner": "<=25 words, plain, what it does — not why it matters",
  "confidence": 0.0-1.0}}

RECORD
Title: {title}
Venue: {venue}
Text: {text}"""


def prefilter(rec, rules):
    """True if the record is worth spending an LLM call on."""
    title = rec["title"].lower()
    blob = f'{rec["title"]} {rec["text"]}'.lower()
    if not blob.strip():
        return False

    # Title-only drops are decisive: a paper titled "...: a database" is a
    # database no matter what the abstract says.
    if any(term in title for term in rules.get("reject_title", ())):
        return False

    if any(term in blob for term in rules["reject"]) and \
       not any(term in blob for term in ("benchmark", "we present", "we develop")):
        return False

    if not any(a in blob for a in rules["anchors"]):
        return False
    # A GitHub repo is itself the software artifact, and its description is far
    # too short to contain phrases like "we present". Anchor match is enough.
    if rec["source"] == "github":
        return True
    return any(t in blob for t in rules["artifacts"])


def _build_prompt(rec, vocab, learned=()):
    # Rules distilled from past corrections ride along in every classification;
    # this is the only channel by which accumulated experience reaches the LLM.
    rules_block = ""
    if learned:
        rules_block = ("ADDITIONAL RULES (learned from past corrections — follow strictly):\n"
                       + "\n".join(f"- {r}" for r in learned) + "\n\n")
    return PROMPT.format(
        rules_block=rules_block,
        kind=", ".join(vocab["kind"]),
        modality=", ".join(vocab["modality"]),
        platform=", ".join(vocab["platform"]),
        title=rec["title"],
        venue=rec["venue"],
        text=rec["text"][:4000],
    )


def _parse_verdict(raw, rec):
    """Parse the model's JSON reply. Both backends funnel through here so the
    two paths cannot drift apart. Returns dict or None."""
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        print(f'  ! unparseable LLM reply for {rec["source_id"]}')
        return None


def classify(rec, vocab, model, learned=()):
    """API backend. Calls the raw Messages API, billed per-token against
    ANTHROPIC_API_KEY. Returns (verdict|None, usage) where usage carries
    input_tokens / output_tokens."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")

    prompt = _build_prompt(rec, vocab, learned)

    r = requests.post(
        API,
        headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                 "content-type": "application/json"},
        json={"model": model, "max_tokens": 700,
              "messages": [{"role": "user", "content": prompt}]},
        timeout=120,
    )
    r.raise_for_status()

    body = r.json()
    raw = "".join(b["text"] for b in body["content"] if b["type"] == "text")
    return _parse_verdict(raw, rec), body.get("usage", {})


def classify_via_cli(rec, vocab, model, learned=()):
    """CLI backend. Shells out to `claude -p`, which authenticates with your
    Pro/Max subscription and draws on its usage limits instead of an API bill.

    Requires `claude login` first. ANTHROPIC_API_KEY is stripped from the child
    environment on purpose: if it is present, Claude Code authenticates with the
    key and you are silently billed for API usage rather than the subscription.

    NOTE (2026-07): the subscription path for `claude -p` is officially in flux
    and may change with notice. Confirm it still works with a small `--limit`
    run before a large backfill. Returns (verdict|None, usage) where usage
    carries input_tokens / output_tokens and, if reported, cost_usd.
    """
    exe = shutil.which("claude")
    if not exe:
        raise RuntimeError(
            "`claude` CLI not found on PATH. Install Claude Code and run "
            "`claude login`, or use --backend api.")

    prompt = _build_prompt(rec, vocab, learned)
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}

    proc = subprocess.run(
        [exe, "-p", prompt, "--output-format", "json", "--model", model],
        capture_output=True, text=True, env=env, timeout=180,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"claude CLI exited {proc.returncode}: {proc.stderr.strip()[:300]}")

    try:
        envelope = json.loads(proc.stdout)
    except json.JSONDecodeError:
        print(f'  ! unparseable CLI envelope for {rec["source_id"]}')
        return None, {}
    if envelope.get("is_error"):
        print(f'  ! claude reported an error for {rec["source_id"]}')
        return None, {}
    # `--output-format json` wraps the assistant text in a result envelope; the
    # JSON we asked for is in "result", token counts in "usage", $ in top-level.
    usage = dict(envelope.get("usage", {}))
    if envelope.get("total_cost_usd") is not None:
        usage["cost_usd"] = envelope["total_cost_usd"]
    return _parse_verdict(envelope.get("result", ""), rec), usage


def sanitise(verdict, vocab):
    """Drop any value the LLM invented outside the controlled vocabulary."""
    verdict["kind"] = [k for k in verdict.get("kind") or [] if k in vocab["kind"]]
    verdict["modality"] = [m for m in verdict.get("modality") or []
                           if m in vocab["modality"]]
    verdict["platforms"] = [p for p in verdict.get("platforms") or []
                            if p in vocab["platform"]]
    # A record with no valid kind is unusable as a catalogue entry.
    if not verdict["kind"]:
        verdict["keep"] = False
    return verdict
