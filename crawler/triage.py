"""Two-stage triage.

Stage 1 (`prefilter`) is pure string matching: free, deterministic, tuned for
recall. Stage 2 (`classify`) sends only survivors to an LLM, which decides
whether the record is really a new resource and extracts structured fields
from the controlled vocabulary.

Roughly 90% of raw hits die in stage 1, which is what keeps the API bill flat.
"""

import json
import os

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

Controlled vocabulary — you MUST choose from these exact strings.

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


def classify(rec, vocab, model):
    """Ask the LLM to judge and structure one record. Returns dict or None."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")

    prompt = PROMPT.format(
        kind=", ".join(vocab["kind"]),
        modality=", ".join(vocab["modality"]),
        platform=", ".join(vocab["platform"]),
        title=rec["title"],
        venue=rec["venue"],
        text=rec["text"][:4000],
    )

    r = requests.post(
        API,
        headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                 "content-type": "application/json"},
        json={"model": model, "max_tokens": 700,
              "messages": [{"role": "user", "content": prompt}]},
        timeout=120,
    )
    r.raise_for_status()

    raw = "".join(b["text"] for b in r.json()["content"] if b["type"] == "text")
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        print(f'  ! unparseable LLM reply for {rec["source_id"]}')
        return None


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
