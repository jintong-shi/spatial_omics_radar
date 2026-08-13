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

_INSTRUCTIONS = """You are triaging literature for a catalogue of spatial omics resources.

Decide whether this record introduces a NEW, USABLE RESOURCE across any spatial \
modality — transcriptomics, proteomics, metabolomics, epigenomics or multi-omics.

REJECT:
- application papers whose contribution is a biological finding, even if they \
used spatial data heavily
- reviews, perspectives, commentaries
- databases, web portals, data repositories and atlases (out of scope)

{rules_block}Controlled vocabulary — you MUST choose from these exact strings.

kind (1-2 values; these are NOT mutually exclusive): {kind}
  assay     = a new wet-lab technique or library-prep chemistry
  method    = a new computational method: software, pipeline, statistical
              method or learned model. If its contribution is something that
              runs on a computer, it is a method.
  benchmark = a STANDALONE study whose OWN contribution is a systematic,
              third-party comparison or evaluation of existing methods, and
              which introduces no substantial new method of its own.
              Do NOT add "benchmark" to a method or assay paper just because
              it compares itself against baselines — every method paper does
              that, and it does NOT make the paper a benchmark.
  A record can still be several: a new assay shipping analysis code is
  [assay, method].

approach (return EXACTLY 1 value when kind includes "method", otherwise []): {approach}
  The single most dominant paradigm of the method. Definitions:
  - foundation model        = pretrained on a large corpus AND explicitly claims
    transfer or zero-shot to downstream tasks. Judge by pretraining + transfer,
    NOT by parameter count.
  - generative              = the method's DELIVERABLE is generated or sampled
    data — it models the data distribution and outputs new samples (diffusion,
    VAE, GAN, flow, autoregressive). Typical uses: imputation, super-resolution,
    generating spatial expression from scRNA, in-silico perturbation.
    Do NOT pick "generative" just because the model contains a GAN/adversarial
    or VAE component: if that component is only a training device for a
    discriminative task (clustering, domain detection, classification, trajectory
    inference), the approach is "deep learning". Classify by what the method
    produces, not its architecture.
  - LLM / agent             = puts an LLM in the pipeline: agentic pipeline,
    natural-language interface, or an LLM doing annotation or reasoning.
  - deep learning           = a neural network trained from scratch for a single
    task (GNN, CNN, transformer with no pretraining).
  - classical / statistical = no learned neural component: GLM, Gaussian process
    (SpatialDE, SPARK), optimal transport, HMM, graph algorithms.
  Precedence: "generative" and "foundation model" outrank "deep learning" (a
  diffusion model is generative, not deep learning; a pretrained transferable
  network is a foundation model, not deep learning). Use "deep learning" only for
  a from-scratch, single-task, non-generative net, and "classical / statistical"
  only when there is no neural component at all. If two of {{foundation model,
  generative, LLM / agent}} genuinely apply, pick the one that is the paper's
  core contribution.

modality (1-3 values): {modality}
platform (0-3 values, only if genuinely platform-specific): {platform}"""

# The verdict object every record gets. Kept as a plain string (not a .format
# template) so its braces need no escaping when composed into either prompt.
_VERDICT_SHAPE = '''{"keep": true|false,
  "reason": "<=15 words",
  "name": "resource name, e.g. Squidpy (null if none is named)",
  "kind": ["1-2 values"],
  "approach": ["1 value if kind includes method, else empty"],
  "modality": ["1-3 values"],
  "platforms": ["0-3 values"],
  "one_liner": "<=25 words, plain, what it does — not why it matters",
  "confidence": 0.0-1.0}'''


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


def _rules_block(learned):
    # Rules distilled from past corrections ride along in every classification;
    # this is the only channel by which accumulated experience reaches the LLM.
    if not learned:
        return ""
    return ("ADDITIONAL RULES (learned from past corrections — follow strictly):\n"
            + "\n".join(f"- {r}" for r in learned) + "\n\n")


def _instructions(vocab, learned):
    return _INSTRUCTIONS.format(
        rules_block=_rules_block(learned),
        kind=", ".join(vocab["kind"]),
        approach=", ".join(vocab["approach"]),
        modality=", ".join(vocab["modality"]),
        platform=", ".join(vocab["platform"]),
    )


def _record_block(rec):
    return (f'Title: {rec["title"]}\n'
            f'Venue: {rec["venue"]}\n'
            f'Text: {rec["text"][:4000]}')


def _build_prompt(rec, vocab, learned=()):
    return (_instructions(vocab, learned)
            + "\n\nReturn ONLY a JSON object, no prose and no markdown fences:\n"
            + _VERDICT_SHAPE
            + "\n\nRECORD\n" + _record_block(rec))


def _build_batch_prompt(recs, vocab, learned=()):
    """One prompt classifying many records at once. Amortises the fixed
    per-call overhead (large in the CLI backend) across the whole batch."""
    records = "\n\n".join(f"[RECORD {i}]\n{_record_block(r)}"
                          for i, r in enumerate(recs, 1))
    return (_instructions(vocab, learned)
            + f"\n\nYou are given {len(recs)} records. Return ONLY a JSON array of "
            + f"exactly {len(recs)} objects, no prose and no markdown fences. Each "
            + 'object adds an "n" field set to its record number and otherwise has '
            + "this shape:\n"
            + _VERDICT_SHAPE
            + f"\n\nRECORDS\n{records}")


def _parse_verdict(raw, rec):
    """Parse the model's JSON reply. Both backends funnel through here so the
    two paths cannot drift apart. Returns dict or None."""
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        print(f'  ! unparseable LLM reply for {rec["source_id"]}')
        return None


def _parse_batch(raw, recs):
    """Parse a JSON array of verdicts and align it to `recs`. Returns a list the
    same length as recs; any record the model dropped or mangled comes back as
    None so the caller can retry it individually."""
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        arr = json.loads(raw)
    except json.JSONDecodeError:
        return [None] * len(recs)
    if not isinstance(arr, list):
        return [None] * len(recs)

    def clean(o):
        if not isinstance(o, dict):
            return None
        o = dict(o)
        o.pop("n", None)
        return o

    # Preferred alignment: the model echoes each record's 1-based "n".
    by_n = {o["n"]: o for o in arr
            if isinstance(o, dict) and isinstance(o.get("n"), int)}
    if by_n:
        return [clean(by_n.get(i)) for i in range(1, len(recs) + 1)]
    # No usable "n" fields — trust positional order only when the count matches
    # exactly, otherwise force per-record retries for the whole batch.
    if len(arr) == len(recs):
        return [clean(o) for o in arr]
    return [None] * len(recs)


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


def classify_via_cli(rec, vocab, model, learned=(), extra_args=()):
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
        [exe, "-p", prompt, "--output-format", "json", "--model", model, *extra_args],
        capture_output=True, text=True, env=env, timeout=180,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"claude CLI exited {proc.returncode}: "
            f"{(proc.stderr.strip() or proc.stdout.strip())[:500]}")

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


def classify_batch(recs, vocab, model, learned=()):
    """API backend, many records per call. Returns (list[verdict|None], usage).
    List is aligned to `recs`; a None means that record needs a single retry."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")

    prompt = _build_batch_prompt(recs, vocab, learned)
    r = requests.post(
        API,
        headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                 "content-type": "application/json"},
        json={"model": model, "max_tokens": 300 * len(recs) + 200,
              "messages": [{"role": "user", "content": prompt}]},
        timeout=300,
    )
    r.raise_for_status()

    body = r.json()
    raw = "".join(b["text"] for b in body["content"] if b["type"] == "text")
    return _parse_batch(raw, recs), body.get("usage", {})


def classify_batch_via_cli(recs, vocab, model, learned=(), extra_args=()):
    """CLI backend, many records per call — the backfill workhorse. Returns
    (list[verdict|None], usage) aligned to `recs`; a None means the batch failed
    to return that record and the caller should retry it individually."""
    exe = shutil.which("claude")
    if not exe:
        raise RuntimeError(
            "`claude` CLI not found on PATH. Install Claude Code and run "
            "`claude login`, or use --backend api.")

    prompt = _build_batch_prompt(recs, vocab, learned)
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}

    proc = subprocess.run(
        [exe, "-p", prompt, "--output-format", "json", "--model", model, *extra_args],
        capture_output=True, text=True, env=env, timeout=600,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"claude CLI exited {proc.returncode}: "
            f"{(proc.stderr.strip() or proc.stdout.strip())[:500]}")

    try:
        envelope = json.loads(proc.stdout)
    except json.JSONDecodeError:
        print(f"  ! unparseable CLI envelope for a batch of {len(recs)}")
        return [None] * len(recs), {}
    if envelope.get("is_error"):
        print(f"  ! claude reported an error for a batch of {len(recs)}")
        return [None] * len(recs), {}
    usage = dict(envelope.get("usage", {}))
    if envelope.get("total_cost_usd") is not None:
        usage["cost_usd"] = envelope["total_cost_usd"]
    return _parse_batch(envelope.get("result", ""), recs), usage


def _build_week_prompt(entries):
    """Prompt for the weekly Slack digest blurb: this week's new entries in, a
    short SYNTHESISED highlight out (themes + a few named standouts, never a list
    of names). Each entry carries venue / impact factor / axes so the model can
    judge what is actually notable."""
    lines = []
    for e in entries:
        tags = " · ".join([*(e.get("kind") or []), *(e.get("approach") or []),
                           *(e.get("modality") or [])])
        venue = "preprint" if e.get("is_preprint") else (e.get("venue") or "")
        if e.get("impact_factor"):
            venue = f"{venue}, IF {e['impact_factor']}".lstrip(", ")
        one = e.get("one_liner") or e.get("title") or ""
        name = e.get("name") or e.get("title") or "untitled"
        head = f"{name} ({venue})" if venue else name
        lines.append(f"- {head}: {one}" + (f" [{tags}]" if tags else ""))
    catalogue = "\n".join(lines)
    return (
        "You write the highlight for a weekly digest of new spatial-omics resources "
        "(tools, methods, benchmarks, assays, models), read by working spatial-omics "
        f"researchers in Slack. This week's {len(entries)} new entries, each as "
        "'name (venue, impact factor): one-liner [kind · approach · modality]':\n\n"
        f"{catalogue}\n\n"
        "Write 3-5 sentences (max ~110 words) that SYNTHESISE the week. Do NOT enumerate "
        "entries. Follow this shape:\n"
        "1. Lead with the week's dominant theme, or which kinds/modalities dominate.\n"
        "2. Call out 2-3 genuinely notable standouts, named, each with WHY it matters — "
        "judge notability from the venue and impact factor (a journal paper or a benchmark "
        "outweighs yet another preprint method).\n"
        "3. Explicitly flag any non-transcriptomics work (proteomics / metabolomics / "
        "epigenomics / multi-omics); cross-modality coverage is the point of this digest.\n"
        "Plain text only: no markdown, no bullet points, no preamble, no sign-off.\n\n"
        "BAD (never do this): 'New this week: ToolA, ToolB, ToolC, and 12 more.'\n"
        "GOOD (do this): 'Spatial proteomics led the week: a Nature Methods antibody-panel "
        "assay and a benchmark of three deconvolution tools stood out, while two graph-based "
        "methods target cell-cell communication. Beyond transcriptomics, a new MALDI workflow "
        "pushes metabolite imaging toward sub-micron resolution.'"
    )


def summarise_week(entries, model, backend, extra_args=()):
    """Return a plain-text highlight paragraph for the weekly digest, produced by
    the same backend as classification (cli = subscription quota, api = per-token).
    Raises on any failure; the caller decides whether to degrade to a name list."""
    prompt = _build_week_prompt(entries)
    if backend == "cli":
        exe = shutil.which("claude")
        if not exe:
            raise RuntimeError(
                "`claude` CLI not found on PATH. Install Claude Code and run "
                "`claude login`, or use --backend api.")
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        proc = subprocess.run(
            [exe, "-p", prompt, "--output-format", "json", "--model", model, *extra_args],
            capture_output=True, text=True, env=env, timeout=180,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"claude CLI exited {proc.returncode}: "
                f"{(proc.stderr.strip() or proc.stdout.strip())[:500]}")
        envelope = json.loads(proc.stdout)
        if envelope.get("is_error"):
            raise RuntimeError("claude CLI reported an error generating the weekly summary")
        return (envelope.get("result") or "").strip()

    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    r = requests.post(
        API,
        headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                 "content-type": "application/json"},
        json={"model": model, "max_tokens": 550,
              "messages": [{"role": "user", "content": prompt}]},
        timeout=120,
    )
    r.raise_for_status()
    body = r.json()
    return "".join(b["text"] for b in body["content"] if b["type"] == "text").strip()


def sanitise(verdict, vocab):
    """Drop any value the LLM invented outside the controlled vocabulary."""
    verdict["kind"] = [k for k in verdict.get("kind") or [] if k in vocab["kind"]]
    # approach is a single-valued axis that only applies to methods; keep at most
    # the first valid value, and drop it entirely for non-method entries.
    verdict["approach"] = [a for a in verdict.get("approach") or []
                           if a in vocab["approach"]][:1]
    if "method" not in verdict["kind"]:
        verdict["approach"] = []
    verdict["modality"] = [m for m in verdict.get("modality") or []
                           if m in vocab["modality"]]
    verdict["platforms"] = [p for p in verdict.get("platforms") or []
                            if p in vocab["platform"]]
    # A record with no valid kind is unusable as a catalogue entry.
    if not verdict["kind"]:
        verdict["keep"] = False
    return verdict
