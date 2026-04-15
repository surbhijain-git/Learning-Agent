"""
summariser.py
Usage: python summariser.py <path_to_transcript_json>

Pipeline:
  1. Chunk-level extraction  — Haiku (speed)
  2. Aggregation             — model chosen by chunk count
  3. Notion KB write
  4. Self-test
"""

import sys
import os
import json
import re
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
import anthropic
from notion_client import Client

load_dotenv()

# ── Clients ───────────────────────────────────────────────────────────────────
ai = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
notion = Client(auth=os.getenv("NOTION_TOKEN"))

NOTION_DB_ID = os.getenv("NOTION_DB_ID")
EXPORTS_DIR = Path("exports")
EXPORTS_DIR.mkdir(exist_ok=True)

# ── Model constants ───────────────────────────────────────────────────────────
HAIKU  = "claude-haiku-4-5-20251001"
SONNET = "claude-sonnet-4-6"
OPUS   = "claude-opus-4-6"


def pick_aggregation_model(chunk_count: int) -> str:
    if chunk_count <= 3:
        return HAIKU
    elif chunk_count <= 8:
        return SONNET
    else:
        return OPUS


# ── Step 1: Chunk-level extraction ───────────────────────────────────────────
CHUNK_SYSTEM = (
    "You are a knowledge extraction engine for a strategy consultant with an "
    "engineering background. From this content chunk, extract two types of knowledge:\n\n"
    "INSIGHT: a genuinely non-obvious strategic idea, implication, or framework — "
    "something that changes how you think. Skip obvious statements.\n"
    "LEARN: a specific tool, feature, capability, named concept, command, technique, "
    "or product behaviour introduced in this chunk — something concrete and referenceable "
    "(e.g. 'Claude Code has a skills feature that expands slash commands into full prompts', "
    "'hooks run shell commands automatically before/after tool calls').\n\n"
    "Output each item on its own line prefixed exactly with its type:\n"
    "INSIGHT: <single sentence>\n"
    "LEARN: <single sentence>\n\n"
    "Skip anything already obvious. Be precise for LEARN items — name the tool/feature explicitly."
)


def extract_claims_from_chunk(chunk_text: str) -> dict[str, list[str]]:
    """Returns {'insights': [...], 'learnings': [...]}"""
    response = ai.messages.create(
        model=HAIKU,
        max_tokens=1024,
        messages=[{"role": "user", "content": chunk_text}],
        system=CHUNK_SYSTEM,
    )
    raw = response.content[0].text.strip()
    insights, learnings = [], []
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("INSIGHT:"):
            text = line[len("INSIGHT:"):].strip()
            if text:
                insights.append(text)
        elif line.startswith("LEARN:"):
            text = line[len("LEARN:"):].strip()
            if text:
                learnings.append(text)
    return {"insights": insights, "learnings": learnings}


def extract_all_claims(chunks: list[str]) -> dict[str, list[str]]:
    """Returns {'insights': [...], 'learnings': [...]} deduplicated."""
    all_insights, all_learnings = [], []
    for i, chunk in enumerate(chunks):
        print(f"  Extracting chunk {i+1}/{len(chunks)}...", end=" ", flush=True)
        result = extract_claims_from_chunk(chunk)
        print(f"{len(result['insights'])} insights, {len(result['learnings'])} learnings")
        all_insights.extend(result["insights"])
        all_learnings.extend(result["learnings"])

    def dedup(items):
        seen, out = set(), []
        for x in items:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    return {"insights": dedup(all_insights), "learnings": dedup(all_learnings)}


# ── Step 2: Aggregation ───────────────────────────────────────────────────────
YOUTUBE_SYSTEM = (
    "You are a learning agent for a strategy consultant with an engineering "
    "background. You will receive two lists: strategic INSIGHTS and concrete LEARNINGS "
    "extracted from a YouTube video. Synthesise them into a structured knowledge entry. "
    "Return ONLY valid JSON — no markdown fences, no commentary:\n"
    "{\n"
    '  "title": "short topic label (3-6 words) — name the subject, not the insight. E.g. \'Claude Code Skills Feature\', \'Agentic AI Architecture\', \'LLM Context Engineering\'",\n'
    '  "core_argument": "2-3 sentences — the central thesis of the content",\n'
    '  "key_claims": ["4-6 specific, non-obvious strategic claims worth remembering"],\n'
    '  "concrete_learnings": ["every distinct tool, feature, capability, named concept, or technique introduced — be specific, e.g. \'Claude Code skills feature expands slash commands into full reusable prompts\'. Include ALL of them, do not summarise or merge."],\n'
    '  "so_what": "1-2 sentences — what a strategy consultant would do differently knowing this",\n'
    '  "key_concepts": ["3-5 short topic tags"]\n'
    "}"
)

OTHER_SYSTEM = (
    "You are a learning agent for a strategy consultant with an engineering "
    "background. You will receive two lists: strategic INSIGHTS and concrete LEARNINGS "
    "extracted from a piece of content. Synthesise them into a structured knowledge entry. "
    "Return ONLY valid JSON — no markdown fences, no commentary:\n"
    "{\n"
    '  "title": "short topic label (3-6 words) — name the subject, not the insight",\n'
    '  "core_argument": "1-2 sentences",\n'
    '  "key_claims": ["2-3 strategic claims"],\n'
    '  "concrete_learnings": ["every distinct tool, feature, capability, named concept, or technique introduced — be specific. Include ALL of them."],\n'
    '  "so_what": "1 sentence",\n'
    '  "key_concepts": ["3-5 short topic tags"]\n'
    "}"
)


def aggregate_claims(claims: dict[str, list[str]], source_type: str, model: str) -> dict:
    system = YOUTUBE_SYSTEM if source_type == "YouTube" else OTHER_SYSTEM

    insights_block  = "\n".join(f"- {c}" for c in claims["insights"])
    learnings_block = "\n".join(f"- {c}" for c in claims["learnings"])
    user_msg = (
        f"STRATEGIC INSIGHTS:\n{insights_block}\n\n"
        f"CONCRETE LEARNINGS (tools / features / concepts):\n{learnings_block}"
    )

    response = ai.messages.create(
        model=model,
        max_tokens=4096,
        messages=[{"role": "user", "content": user_msg}],
        system=system,
    )
    raw = response.content[0].text.strip()

    # Strip markdown fences if model added them despite instructions
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    return json.loads(raw)


# ── Step 3: Notion write ──────────────────────────────────────────────────────
def _bullet_blocks(items: list[str]) -> list[dict]:
    """Convert a list of strings into Notion bulleted_list_item blocks."""
    return [
        {
            "object": "block",
            "type": "bulleted_list_item",
            "bulleted_list_item": {
                "rich_text": [{"type": "text", "text": {"content": item}}]
            },
        }
        for item in items
    ]


def write_to_notion(summary: dict, source_url: str, source_type: str) -> str:
    """Returns the created page ID."""
    today = str(date.today())
    summary_text = summary["core_argument"] + "\n\n" + summary["so_what"]

    concrete = summary.get("concrete_learnings", [])
    key_claims = summary.get("key_claims", [])

    # Page body: two sections so the page is useful when opened
    body_blocks = [
        {
            "object": "block",
            "type": "heading_2",
            "heading_2": {
                "rich_text": [{"type": "text", "text": {"content": "What I Learned"}}]
            },
        },
        *_bullet_blocks(concrete),
        {
            "object": "block",
            "type": "heading_2",
            "heading_2": {
                "rich_text": [{"type": "text", "text": {"content": "Key Claims"}}]
            },
        },
        *_bullet_blocks(key_claims),
    ]

    page = notion.pages.create(
        parent={"database_id": NOTION_DB_ID},
        properties={
            "Title": {
                "title": [{"type": "text", "text": {"content": summary["title"]}}]
            },
            "Source": {
                "rich_text": [{"type": "text", "text": {"content": source_url}}]
            },
            "Source_Type": {"select": {"name": source_type}},
            "Date_Added": {"date": {"start": today}},
            "Summary": {
                "rich_text": [{"type": "text", "text": {"content": summary_text}}]
            },
            "Key_Concepts": {
                "multi_select": [{"name": tag} for tag in summary.get("key_concepts", [])]
            },
            "Understanding_Level": {"select": {"name": "New"}},
        },
        children=body_blocks,
    )
    return page["id"]


# ── Step 4: Self-test ─────────────────────────────────────────────────────────
def run_test(summary: dict, page_id: str, source_type: str):
    failures = []

    # title must be substantive
    if len(summary.get("title", "")) < 10:
        failures.append("title is too short / empty")

    # key_claims count
    claims = summary.get("key_claims", [])
    if source_type == "YouTube" and not (4 <= len(claims) <= 6):
        failures.append(f"key_claims count {len(claims)} — expected 4-6 for YouTube")

    # so_what present and non-trivial
    if len(summary.get("so_what", "")) < 20:
        failures.append("so_what is missing or too short")

    # concrete_learnings must be present
    learnings = summary.get("concrete_learnings", [])
    if not learnings:
        failures.append("concrete_learnings is empty — specific features/concepts not captured")

    # Notion entry exists with correct fields and page body
    try:
        page = notion.pages.retrieve(page_id)
        notion_title = page["properties"]["Title"]["title"][0]["text"]["content"]
        if notion_title != summary["title"]:
            failures.append(f"Notion title mismatch: '{notion_title}'")
        # Verify page has body blocks
        blocks = notion.blocks.children.list(page_id)
        if not blocks.get("results"):
            failures.append("Notion page body is empty — concrete_learnings blocks missing")
    except Exception as e:
        failures.append(f"Could not retrieve Notion entry: {e}")

    if failures:
        print("FAIL: " + "; ".join(failures))
    else:
        print(f"PASS — {len(learnings)} concrete learnings + {len(claims)} key claims written to Notion")


# ── Extraction only (no Notion write) ────────────────────────────────────────
def extract_only(json_path) -> dict:
    """
    Run transcript extraction + aggregation but do NOT write to Notion.
    Returns {"summary", "video_id", "source_url", "source_type", "model", "n_chunks"}.
    Used by ingest_queue when writing to Reading List first.
    """
    json_path = Path(json_path)
    with open(json_path, encoding="utf-8") as f:
        transcript = json.load(f)

    video_id    = transcript["video_id"]
    chunks      = transcript["chunks"]
    source_url  = transcript["url"]
    source_type = transcript["source_type"]
    n_chunks    = len(chunks)
    model       = pick_aggregation_model(n_chunks)

    print(f"Source: {source_type} | Chunks: {n_chunks} | Model: {model}")
    print("\n[Step 1] Extracting claims...")
    claims = extract_all_claims(chunks)
    print(f"  {len(claims['insights'])} insights, {len(claims['learnings'])} learnings")
    if not claims["insights"] and not claims["learnings"]:
        raise ValueError(
            "No extractable content — video may be music-only, too short, or in an unsupported language"
        )

    print("\n[Step 2] Aggregating...")
    summary = aggregate_claims(claims, source_type, model)
    print(f"  Title: {summary['title']}")

    return {
        "summary":     summary,
        "video_id":    video_id,
        "source_url":  source_url,
        "source_type": source_type,
        "model":       model,
        "n_chunks":    n_chunks,
    }


# ── Importable pipeline entry point ──────────────────────────────────────────
def process_transcript_file(json_path, notion_only: bool = False) -> dict:
    """
    Full pipeline for a transcript JSON file.
    Can be imported and called directly (e.g. from ingest_queue.py).
    Returns {"page_id", "summary", "dedup", "video_id", "model", "n_chunks"}.
    """
    json_path = Path(json_path)
    with open(json_path, encoding="utf-8") as f:
        transcript = json.load(f)

    video_id    = transcript["video_id"]
    chunks      = transcript["chunks"]
    source_url  = transcript["url"]
    source_type = transcript["source_type"]
    n_chunks    = len(chunks)
    model       = pick_aggregation_model(n_chunks)

    if notion_only:
        export_path = EXPORTS_DIR / f"{video_id}.txt"
        if not export_path.exists():
            raise FileNotFoundError(f"No cached export at {export_path}. Run without --notion-only first.")
        with open(export_path, encoding="utf-8") as f:
            summary = json.load(f)
        print(f"Loaded cached summary: {summary['title']}")
    else:
        print(f"Source: {source_type} | Chunks: {n_chunks} | Model: {model}")
        print("\n[Step 1] Extracting claims...")
        claims = extract_all_claims(chunks)
        print(f"  {len(claims['insights'])} insights, {len(claims['learnings'])} learnings")
        if not claims["insights"] and not claims["learnings"]:
            raise ValueError(
                "No extractable content — video may be music-only, too short, or in an unsupported language"
            )

        print("\n[Step 2] Aggregating...")
        summary = aggregate_claims(claims, source_type, model)
        print(f"  Title: {summary['title']}")

    print("\n[Step 3] Writing to Notion KB...")
    page_id = write_to_notion(summary, source_url, source_type)
    print(f"  Written: {summary['title']}")

    import dedup_engine
    print("\n[Step 3b] Semantic dedup + concept novelty check...")
    dedup_result = dedup_engine.check_novelty(
        page_id,
        claims=summary.get("key_claims", []),
        learnings=summary.get("concrete_learnings", []),
    )

    # Fallback export in case caller wants it
    export_path = EXPORTS_DIR / f"{video_id}.txt"
    export_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    return {
        "page_id":  page_id,
        "summary":  summary,
        "dedup":    dedup_result,
        "video_id": video_id,
        "model":    model,
        "n_chunks": n_chunks,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = sys.argv[1:]
    notion_only = "--notion-only" in args
    positional = [a for a in args if not a.startswith("--")]

    if not positional:
        print("Usage: python summariser.py <path_to_transcript_json> [--notion-only]")
        sys.exit(1)

    json_path = Path(positional[0])
    if not json_path.exists():
        print(f"File not found: {json_path}")
        sys.exit(1)

    try:
        result = process_transcript_file(json_path, notion_only=notion_only)
    except Exception as e:
        print(f"Pipeline failed: {e}")
        sys.exit(1)

    # Step 4: self-test
    print("\n[Step 4] Running self-test...")
    run_test(result["summary"], result["page_id"], result["summary"].get("source_type", "YouTube"))


if __name__ == "__main__":
    main()
