"""
eval/rubric.py
Rubric definition, scoring weights, examples, and the Judge LLM prompt.
Imported by eval_agent.py.
"""

import json
from pathlib import Path

# ── Scoring thresholds ────────────────────────────────────────────────────────
PASS_THRESHOLD   = 3.5
REVIEW_THRESHOLD = 2.5

# ── Dimension weights (must sum to 1.0) ───────────────────────────────────────
WEIGHTS = {
    "extraction_fidelity":  0.25,
    "insight_depth":        0.30,
    "novelty_calibration":  0.20,
    "pipeline_integrity":   0.10,
    "structural_quality":   0.10,
    "strategic_relevance":  0.05,
}

DIMENSION_DESCRIPTIONS = {
    "extraction_fidelity": (
        "Does the output accurately reflect what was in the source? "
        "No hallucinations or invented facts? Key ideas from the source are captured?"
    ),
    "insight_depth": (
        "Are the claims and learnings genuinely non-obvious and specific? "
        "Would a strategy consultant learn something actionable? "
        "Or are they generic summaries anyone could write without reading the source?"
    ),
    "novelty_calibration": (
        "Is the NEW / RELATED / COVERED verdict accurate given the existing KB? "
        "NEW = genuinely not covered before. RELATED = overlaps but adds angles. "
        "COVERED = substantially already in KB."
    ),
    "structural_quality": (
        "Are all fields populated meaningfully? "
        "Title: short topic label (not an insight sentence). "
        "Summary: 2-3 sentences, specific. "
        "Key claims: 3-5 specific non-obvious statements. "
        "Concrete learnings: named tools, frameworks, concepts — not vague. "
        "Key concepts: precise topic tags."
    ),
    "pipeline_integrity": (
        "Did the entry move correctly through the full pipeline without silent failures? "
        "Checks: (1) source was fetched from the correct location (e.g. right Gmail label, not INBOX default), "
        "(2) dedup prevented reprocessing the same source on re-runs, "
        "(3) entry appeared in Reading List if NEW/RELATED (not if COVERED), "
        "(4) all required fields (Source_Type, Date_Added, Source) are populated, "
        "(5) no silent errors — failures surfaced visibly rather than swallowed by continue-on-error."
    ),
    "strategic_relevance": (
        "Is this content genuinely useful for roles in AI strategy, GTM strategy, "
        "product strategy, or strategy consulting? "
        "Would someone targeting AI/strategy roles find this worth keeping?"
    ),
}

# ── Anchored scoring scale (1 / 3 / 5 per dimension) ──────────────────────────
# Inspired by the Golden Dataset pattern: explicit anchors make the Judge LLM
# more calibrated than a generic 1-5 scale.
SCORING_ANCHORS = {
    "extraction_fidelity": {
        1: "Output contains invented facts, misattributed claims, or misses the source's core argument. Could not have been written from the source alone.",
        3: "Main ideas captured but some details are paraphrased loosely or minor claims are missing. No hallucinations but not precise.",
        5: "Accurately reflects the source with no hallucinations. Key argument, specific data points, and supporting evidence are all correctly captured.",
    },
    "insight_depth": {
        1: "All claims are obvious — things anyone in the industry already knows. No specific data, no named mechanisms, no non-obvious connections. Generic enough to apply to any source.",
        3: "Some non-obvious content but mixed with filler. At least one specific claim or data point. A consultant might skim it but wouldn't save it.",
        5: "Claims are specific, non-obvious, and actionable. Named mechanisms, frameworks, or data points. A strategy consultant would learn something they couldn't have inferred without reading the source.",
    },
    "novelty_calibration": {
        1: "Verdict is clearly wrong — e.g., calls something NEW when the KB has multiple entries on the same topic, or COVERED when the content introduces a genuinely distinct angle.",
        3: "Verdict is directionally right but imprecise. Could argue either way between adjacent categories (NEW vs RELATED, or RELATED vs COVERED).",
        5: "Verdict is accurate and well-calibrated. NEW means no prior KB entries touch this angle. RELATED means real overlap but a distinct sub-topic or lens. COVERED means the KB already contains substantially the same insight.",
    },
    "structural_quality": {
        1: "Multiple fields missing or meaningless. Title is a vague sentence. Summary is one line or a copy-paste of the title. Claims are bullet points of obvious facts. Learnings name no specific tools or concepts.",
        3: "Most fields present but one or two are weak. Title is acceptable. Summary is present but could be tighter. At least 2 specific claims. Learnings include some named concepts.",
        5: "All fields complete and precise. Title is a short topic label (3-6 words). Summary is 2-3 specific sentences. 3-5 claims that are non-obvious. Learnings name specific tools, frameworks, percentages, or model names. Key concepts are precise tags.",
    },
    "pipeline_integrity": {
        1: (
            "Entry has one or more critical failures: required fields missing (Source_Type, Date_Added, or Source blank); "
            "OR a NEW/RELATED entry has no Reading List counterpart; "
            "OR source was fetched from the wrong location (e.g. INBOX instead of the correct Gmail label) meaning entire source categories were silently skipped; "
            "OR the same source was reprocessed on re-runs due to broken dedup (e.g. .seen_ids file not persisting across CI runs); "
            "OR a pipeline failure was swallowed silently (continue-on-error) with no visible signal."
        ),
        3: (
            "Required fields populated and entry reached the correct pipeline stage, but one reliability issue present — "
            "e.g. dedup works locally but not in CI, or errors surface in logs but not as job failures, "
            "or Source field shows a raw filename rather than a meaningful identifier."
        ),
        5: (
            "All required fields populated. Entry moved correctly through all stages: "
            "Ingest Queue item processed and removed, NEW/RELATED entry in Reading List, COVERED entry absent. "
            "Source fetched from the correct mailbox/location. "
            "Dedup is stable across re-runs (same source not reprocessed). "
            "Failures surface visibly — no silent swallowing of errors."
        ),
    },
    "strategic_relevance": {
        1: "Content is tangential or irrelevant to AI strategy, GTM strategy, product strategy, or consulting. Would not be useful in any strategy or advisory role.",
        3: "Generally relevant domain but too high-level to be actionable in a specific role. Good background knowledge but not decision-relevant.",
        5: "Directly applicable to AI strategy, GTM, product, or consulting roles. A senior colleague in those roles would recognize it as substantive, citable domain knowledge.",
    },
}

# ── Few-shot examples for the Judge ──────────────────────────────────────────
FEW_SHOT_EXAMPLES = [
    {
        "label": "PIPELINE FAILURE example (pipeline_integrity = 1)",
        "entry": {
            "title": "Notion API Patterns for Agents",
            "summary": "Notion's API has breaking changes in v2.2.1 that drop databases.query. Agents must use direct httpx calls instead of the SDK.",
            "key_claims": [
                "notion-client v2.2.1 dropped databases.query silently",
                "Direct httpx.post with Notion-Version: 2022-06-28 is the stable pattern",
                "SDK updates can silently break downstream agents without test coverage",
            ],
            "concrete_learnings": [
                "Use httpx.post to https://api.notion.com/v1/databases/{id}/query directly",
                "Pin Notion-Version header to 2022-06-28 for stability",
            ],
            "key_concepts": ["Notion API", "SDK versioning", "Agent reliability"],
            "verdict": "NEW",
        },
        "scores": {
            "extraction_fidelity": 4,
            "insight_depth": 4,
            "novelty_calibration": 3,
            "pipeline_integrity": 1,
            "structural_quality": 4,
            "strategic_relevance": 3,
        },
        "reasoning": {
            "extraction_fidelity": "Accurately reflects the source content with no hallucinations.",
            "insight_depth": "SDK version-breaking-change is a non-obvious and actionable discovery.",
            "novelty_calibration": "Directionally NEW but KB may have related API reliability entries.",
            "pipeline_integrity": (
                "Critical pipeline failures on this entry: "
                "(1) newsletter fetched from INBOX instead of INBOX/Newsletter — entire newsletter category was silently skipped for weeks; "
                "(2) continue-on-error: true on the newsletter sync step swallowed Gmail auth failures with no visible signal; "
                "(3) .seen_ids file not persisted in CI means same emails reprocessed every daily run, burning tokens; "
                "(4) YouTube items marked Failed in Notion with no local retry path — items permanently stuck. "
                "This entry only appeared in KB by chance; the pipeline integrity is broken."
            ),
            "structural_quality": "Fields complete and specific. Claims name exact version numbers and API patterns.",
            "strategic_relevance": "Relevant to AI product and agent-building roles.",
        },
        "weighted_score": 2.95,
        "verdict": "REVIEW",
    },
    {
        "label": "PASS example",
        "entry": {
            "title": "Claude Model Tiering & Cost Optimization",
            "summary": "Claude should be deployed as a tiered system where Haiku handles chunk-level extraction, Sonnet handles mid-size aggregation, and Opus handles large content. Model selection is a performance optimization decision, not a capability one.",
            "key_claims": [
                "Model selection by chunk count reduces cost 60-80% vs always using Opus",
                "Haiku is sufficient for structured extraction tasks with clear output schemas",
                "Aggregation model should scale with content complexity, not default to most capable",
            ],
            "concrete_learnings": [
                "Haiku: ≤3 chunks, Sonnet: 4-8 chunks, Opus: >8 chunks",
                "Structured JSON output schemas make Haiku reliable for extraction",
                "claude-haiku-4-5-20251001, claude-sonnet-4-6, claude-opus-4-6 model IDs",
            ],
            "key_concepts": ["Model tiering", "Cost optimization", "Claude API", "LLM deployment"],
            "verdict": "NEW",
        },
        "scores": {
            "extraction_fidelity": 5,
            "insight_depth": 4,
            "novelty_calibration": 5,
            "pipeline_integrity": 5,
            "structural_quality": 5,
            "strategic_relevance": 5,
        },
        "reasoning": {
            "extraction_fidelity": "Accurately reflects the source's core argument on tiered deployment.",
            "insight_depth": "Tiering logic is specific and non-obvious; cost % is estimated but reasonable.",
            "novelty_calibration": "KB had no prior entries on model selection strategy — correctly NEW.",
            "pipeline_integrity": "All required fields populated, Reading List entry present, queue item removed.",
            "structural_quality": "All fields precise. Title is a clean topic label. Claims are specific.",
            "strategic_relevance": "Directly applicable to AI product strategy and build decisions.",
        },
        "weighted_score": 4.85,
        "verdict": "PASS",
    },
    {
        "label": "FAIL example",
        "entry": {
            "title": "AI is changing everything",
            "summary": "AI will transform many industries and companies need to adapt quickly.",
            "key_claims": [
                "AI is growing fast",
                "Companies are adopting AI",
                "AI will create new jobs and eliminate others",
            ],
            "concrete_learnings": [
                "AI tools are becoming more accessible",
                "Leadership needs to understand AI",
            ],
            "key_concepts": ["AI", "Transformation", "Future of work"],
            "verdict": "NEW",
        },
        "scores": {
            "extraction_fidelity": 2,
            "insight_depth": 1,
            "novelty_calibration": 2,
            "pipeline_integrity": 2,
            "structural_quality": 2,
            "strategic_relevance": 3,
        },
        "reasoning": {
            "extraction_fidelity": "Too generic — doesn't reflect any specific source argument.",
            "insight_depth": "All claims are obvious. Nothing a strategy consultant couldn't already know.",
            "novelty_calibration": "These concepts are definitely covered in KB. Should be COVERED not NEW.",
            "pipeline_integrity": "Source field is blank and Date_Added missing — required fields not populated.",
            "structural_quality": "Title is a sentence fragment. Claims are boilerplate. No named tools or frameworks.",
            "strategic_relevance": "AI is relevant domain but nothing actionable here.",
        },
        "weighted_score": 1.9,
        "verdict": "FAIL",
    },
    {
        "label": "REVIEW example",
        "entry": {
            "title": "Agentic AI Workflow Design",
            "summary": "AI agents work best when broken into discrete tasks with clear handoffs. Orchestration patterns matter more than model capability for production reliability.",
            "key_claims": [
                "Task decomposition improves agent reliability",
                "Orchestration layer is becoming the competitive moat in AI",
                "Human-in-the-loop checkpoints reduce failure cascades",
            ],
            "concrete_learnings": [
                "Multi-agent systems outperform single agents on complex tasks",
                "Tool use and memory are the two key agent capabilities",
            ],
            "key_concepts": ["AI agents", "Orchestration", "Workflow design"],
            "verdict": "NEW",
        },
        "scores": {
            "extraction_fidelity": 4,
            "insight_depth": 3,
            "novelty_calibration": 2,
            "pipeline_integrity": 4,
            "structural_quality": 3,
            "strategic_relevance": 4,
        },
        "reasoning": {
            "extraction_fidelity": "Reasonably accurate but somewhat surface-level.",
            "insight_depth": "Orchestration-as-moat is a real insight; task decomposition is fairly well-known.",
            "novelty_calibration": "KB has multiple entries on orchestration and AI infrastructure — should be RELATED not NEW.",
            "pipeline_integrity": "Required fields populated and Reading List entry present, but Source field shows a raw filename. Also: dedup relies on filename matching — if the same source is re-fetched with a different filename (e.g. timestamp-based naming), it will be reprocessed and consume tokens without producing new KB value.",
            "structural_quality": "Claims are decent but learnings are too vague. No named frameworks or tools.",
            "strategic_relevance": "Relevant to AI strategy and product roles.",
        },
        "weighted_score": 3.05,
        "verdict": "REVIEW",
    },
]


# ── Load golden dataset ───────────────────────────────────────────────────────
def _load_golden_dataset() -> list[dict]:
    """Load human-graded good/bad pairs from golden_dataset.json."""
    path = Path(__file__).parent / "golden_dataset.json"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except Exception:
        return []


def _build_golden_block() -> str:
    """
    Build a compact 'Good vs Bad' section from the golden dataset.
    One pair per unique primary dimension — keeps the prompt focused.
    """
    examples = _load_golden_dataset()
    if not examples:
        return ""

    # One example per primary dimension (first word before newline)
    seen_dims: set[str] = set()
    selected = []
    for ex in examples:
        primary_dim = ex.get("dimension", "").split("\n")[0].strip()
        if primary_dim and primary_dim not in seen_dims:
            seen_dims.add(primary_dim)
            selected.append(ex)

    block = "\n## Human-Graded Calibration Examples\n"
    block += (
        "These good/bad pairs were graded by the user. "
        "Use them to calibrate your scores — especially for insight_depth, "
        "novelty_calibration, strategic_relevance, and pipeline_integrity.\n"
    )

    for ex in selected:
        primary_dim = ex.get("dimension", "").split("\n")[0].strip()
        source = ex.get("source", "").split("\n")[0].strip()
        block += f"\n### [{primary_dim}] Source: {source}\n"
        block += f"**Scenario:** {ex.get('scenario', '').strip()}\n\n"
        block += f"**BAD output** (score 1–2 on {primary_dim}):\n"
        block += ex.get("bad_output", "").strip() + "\n"
        block += f"*Why bad:* {ex.get('why_bad', '').strip()}\n\n"
        block += f"**GOOD output** (score 4–5 on {primary_dim}):\n"
        block += ex.get("good_output", "").strip() + "\n"
        block += f"*Why good:* {ex.get('why_good', '').strip()}\n"

    return block


# ── Build the Judge system prompt ─────────────────────────────────────────────
def build_judge_prompt() -> str:
    dim_block = "\n".join(
        f"- **{k}** (weight {int(v*100)}%): {DIMENSION_DESCRIPTIONS[k]}"
        for k, v in WEIGHTS.items()
    )

    # Build anchored scoring table (1 / 3 / 5 per dimension)
    anchor_block = ""
    for dim, anchors in SCORING_ANCHORS.items():
        anchor_block += f"\n### {dim}\n"
        anchor_block += f"- **1 (Failed):** {anchors[1]}\n"
        anchor_block += f"- **3 (Acceptable):** {anchors[3]}\n"
        anchor_block += f"- **5 (Excellent):** {anchors[5]}\n"

    examples_block = ""
    for ex in FEW_SHOT_EXAMPLES:
        e = ex["entry"]
        scores_str = "\n".join(f"  {k}: {v}/5 — {ex['reasoning'][k]}" for k, v in ex["scores"].items())
        examples_block += f"""
### {ex['label']} (weighted score: {ex['weighted_score']:.2f} → {ex['verdict']})
Title: {e['title']}
Summary: {e['summary']}
Key Claims: {e['key_claims']}
Concrete Learnings: {e['concrete_learnings']}
Key Concepts: {e['key_concepts']}
Novelty Verdict: {e['verdict']}

Scores:
{scores_str}
"""

    golden_block = _build_golden_block()

    return f"""You are a Judge LLM evaluating the output quality of a Content Intelligence Agent.
The agent ingests YouTube videos, newsletters, and meeting notes, extracts structured knowledge using Claude, and scores novelty against an existing Knowledge Base.

Your job is to evaluate a KB entry on 6 dimensions and return a JSON score.

## Rubric Dimensions
{dim_block}

## Scoring Anchors
Use these to calibrate your scores. Scores 2 and 4 fall between the anchors.
{anchor_block}

## Verdict Thresholds
- PASS: weighted score ≥ {PASS_THRESHOLD}
- REVIEW: {REVIEW_THRESHOLD} to {PASS_THRESHOLD - 0.01:.2f}
- FAIL: < {REVIEW_THRESHOLD}

## Scored Examples
{examples_block}
{golden_block}

## Output Format
Return ONLY valid JSON, no markdown fences:
{{
  "scores": {{
    "extraction_fidelity": <1-5>,
    "insight_depth": <1-5>,
    "novelty_calibration": <1-5>,
    "pipeline_integrity": <1-5>,
    "structural_quality": <1-5>,
    "strategic_relevance": <1-5>
  }},
  "reasoning": {{
    "extraction_fidelity": "<one sentence>",
    "insight_depth": "<one sentence>",
    "novelty_calibration": "<one sentence>",
    "pipeline_integrity": "<one sentence — note any missing fields, wrong source location, reprocessing, or silent failures>",
    "structural_quality": "<one sentence>",
    "strategic_relevance": "<one sentence>"
  }},
  "weighted_score": <float — do NOT compute this yourself, it will be recalculated>,
  "verdict": "PASS" | "REVIEW" | "FAIL",
  "summary": "<one sentence overall assessment>"
}}"""
