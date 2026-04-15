"""
db_seeder.py
Usage: python db_seeder.py <path_to_folder>

Ingests .txt and .md files into the Notion KB.
Skips files under 100 words, skips already-imported files (dedup by Source=filename
and by content hash to catch same content under a different filename).

Pipeline: chunk full content → Haiku extracts claims per chunk → aggregate model
chosen by chunk count (same pattern as summariser.py). No truncation at ingestion.
"""

import sys
import os
import re
import time
import hashlib
import json
from datetime import date
from pathlib import Path

from dotenv import load_dotenv, find_dotenv
import anthropic
from notion_client import Client

# Load .env with override=True so sandbox env vars don't block real values
load_dotenv(find_dotenv(usecwd=True), override=True)

# ── Clients ───────────────────────────────────────────────────────────────────
ai = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
notion = Client(auth=os.getenv("NOTION_TOKEN"))
NOTION_DB_ID = os.getenv("NOTION_DB_ID")

HAIKU = "claude-haiku-4-5-20251001"
RATE_LIMIT_WAIT = 13  # seconds between API calls

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)
FAIL_LOG = LOGS_DIR / "failed_imports.txt"
HASH_STORE = LOGS_DIR / "seen_hashes.json"


# ── Content hash store ────────────────────────────────────────────────────────
def load_seen_hashes() -> dict[str, str]:
    """Returns {hash: filename} for all previously imported files."""
    if HASH_STORE.exists():
        return json.loads(HASH_STORE.read_text())
    return {}


def save_seen_hashes(hashes: dict[str, str]):
    HASH_STORE.write_text(json.dumps(hashes, indent=2))


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# ── Notion: check if file already imported ────────────────────────────────────
import requests as _requests

def already_in_notion(filename: str) -> bool:
    """Returns True if a page with Source = filename exists in KB or Reading List."""
    token = os.getenv("NOTION_TOKEN")
    headers = {
        "Authorization": f"Bearer {token}",
        "Notion-Version": "2022-06-28",
        "Content-Type": "application/json",
    }
    filter_body = {"filter": {"property": "Source", "rich_text": {"equals": filename}}}
    try:
        for db_id in [NOTION_DB_ID, os.getenv("NOTION_READING_LIST_ID", "")]:
            if not db_id:
                continue
            resp = _requests.post(
                f"https://api.notion.com/v1/databases/{db_id}/query",
                headers=headers, json=filter_body,
            )
            if resp.json().get("results"):
                return True
        return False
    except Exception as e:
        print(f"  [Warning] Notion dedup check failed for {filename}: {e}")
        return False


# ── Model selection ───────────────────────────────────────────────────────────
SONNET = "claude-sonnet-4-6"

def pick_model(chunk_count: int) -> str:
    if chunk_count <= 3:
        return HAIKU
    elif chunk_count <= 8:
        return SONNET
    else:
        return "claude-opus-4-6"


# ── Chunk text ────────────────────────────────────────────────────────────────
def chunk_text(text: str, max_words: int = 500) -> list[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks, current, count = [], [], 0
    for sentence in sentences:
        words = len(sentence.split())
        if count + words > max_words and current:
            chunks.append(" ".join(current))
            current, count = [], 0
        current.append(sentence)
        count += words
    if current:
        chunks.append(" ".join(current))
    return chunks


# ── Chunk-level extraction (Haiku) ────────────────────────────────────────────
CHUNK_SYSTEM = (
    "You are a knowledge extraction engine for a strategy consultant with an engineering background. "
    "From this content chunk, extract two types of knowledge:\n\n"
    "INSIGHT: a genuinely non-obvious strategic idea, implication, or framework — something that changes how you think.\n"
    "LEARN: a specific tool, feature, capability, named concept, technique, or fact — something concrete and referenceable.\n\n"
    "Output each item on its own line prefixed exactly with its type:\n"
    "INSIGHT: <single sentence>\n"
    "LEARN: <single sentence>\n\n"
    "Skip obvious statements. Be precise."
)


def extract_claims_from_chunk(chunk: str) -> dict[str, list[str]]:
    response = ai.messages.create(
        model=HAIKU,
        max_tokens=1024,
        system=CHUNK_SYSTEM,
        messages=[{"role": "user", "content": chunk}],
    )
    raw = response.content[0].text.strip()
    insights, learnings = [], []
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("INSIGHT:"):
            t = line[len("INSIGHT:"):].strip()
            if t:
                insights.append(t)
        elif line.startswith("LEARN:"):
            t = line[len("LEARN:"):].strip()
            if t:
                learnings.append(t)
    return {"insights": insights, "learnings": learnings}


# ── Aggregation ───────────────────────────────────────────────────────────────
AGG_SYSTEM = (
    "You are a knowledge synthesis assistant for a strategy consultant with an engineering background. "
    "You will receive INSIGHTS and LEARNINGS extracted from a document. "
    "Synthesise them into a structured knowledge entry. "
    "Return ONLY valid JSON, no markdown fences:\n"
    "{\n"
    '  "title": "short topic label (3-6 words) — name the subject, not the insight. E.g. \'Edge AI Economics\', \'Cursor GTM Strategy\', \'LLM Persuasion Tactics\'",\n'
    '  "summary": "2-3 sentences — core argument and so-what for a strategy consultant",\n'
    '  "key_claims": ["3-5 specific, non-obvious claims worth remembering"],\n'
    '  "concrete_learnings": ["every distinct tool, concept, framework, or fact — be specific, include ALL of them"],\n'
    '  "key_concepts": ["3-5 short topic tags"]\n'
    "}"
)


def extract_metadata(content: str) -> dict:
    """Chunk full content → Haiku per chunk → aggregate. No truncation."""
    chunks = chunk_text(content)
    all_insights, all_learnings = [], []

    for chunk in chunks:
        result = extract_claims_from_chunk(chunk)
        all_insights.extend(result["insights"])
        all_learnings.extend(result["learnings"])

    # Deduplicate
    all_insights = list(dict.fromkeys(all_insights))
    all_learnings = list(dict.fromkeys(all_learnings))

    model = pick_model(len(chunks))

    insights_block = "\n".join(f"- {c}" for c in all_insights)
    learnings_block = "\n".join(f"- {c}" for c in all_learnings)
    user_msg = f"INSIGHTS:\n{insights_block}\n\nLEARNINGS:\n{learnings_block}"

    response = ai.messages.create(
        model=model,
        max_tokens=4096,
        system=AGG_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )
    raw = response.content[0].text.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


# ── Notion: write entry ───────────────────────────────────────────────────────
def _bullet_blocks(items: list[str]) -> list[dict]:
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


def write_to_notion(metadata: dict, filename: str, source_type: str = "Notes") -> str:
    """Returns the created page ID."""
    today = str(date.today())

    body_blocks = [
        {
            "object": "block", "type": "heading_2",
            "heading_2": {"rich_text": [{"type": "text", "text": {"content": "What I Learned"}}]},
        },
        *_bullet_blocks(metadata.get("concrete_learnings", [])),
        {
            "object": "block", "type": "heading_2",
            "heading_2": {"rich_text": [{"type": "text", "text": {"content": "Key Claims"}}]},
        },
        *_bullet_blocks(metadata.get("key_claims", [])),
    ]

    page = notion.pages.create(
        parent={"database_id": NOTION_DB_ID},
        properties={
            "Title": {
                "title": [{"type": "text", "text": {"content": metadata["title"]}}]
            },
            "Source": {
                "rich_text": [{"type": "text", "text": {"content": filename}}]
            },
            "Source_Type": {"select": {"name": source_type}},
            "Date_Added": {"date": {"start": today}},
            "Summary": {
                "rich_text": [{"type": "text", "text": {"content": metadata["summary"]}}]
            },
            "Key_Concepts": {
                "multi_select": [{"name": tag} for tag in metadata.get("key_concepts", [])]
            },
            "Understanding_Level": {"select": {"name": "Understood"}},
        },
        children=body_blocks,
    )
    return page["id"]


# ── Log failure ───────────────────────────────────────────────────────────────
def log_failure(filename: str, reason: str):
    with open(FAIL_LOG, "a", encoding="utf-8") as f:
        f.write(f"{date.today()} | {filename} | {reason}\n")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="Folder of .txt/.md files to ingest")
    parser.add_argument(
        "--source-type",
        default=None,
        help="Notion Source_Type label (e.g. Granola, Case, Link, Newsletter, YouTube). "
             "If not set, inferred from folder name."
    )
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"Not a directory: {folder}")
        sys.exit(1)

    # Infer source type from folder name if not provided
    folder_lower = folder.name.lower()
    if args.source_type:
        source_type = args.source_type
    elif "granola" in folder_lower or "talks" in folder_lower:
        source_type = "Granola"
    elif "case" in folder_lower:
        source_type = "Case"
    elif "link" in folder_lower:
        source_type = "Link"
    elif "newsletter" in folder_lower:
        source_type = "Newsletter"
    else:
        source_type = "Notes"

    print(f"Source type: {source_type}")

    # Collect all .txt and .md files
    files = sorted(
        [f for f in folder.iterdir() if f.suffix.lower() in (".txt", ".md") and f.is_file()]
    )
    total = len(files)

    if total == 0:
        print("No .txt or .md files found in folder.")
        sys.exit(0)

    seen_hashes = load_seen_hashes()

    added = 0
    skipped_exist = 0
    skipped_dupe = 0
    skipped_short = 0
    failed = 0
    api_calls_made = 0

    for i, filepath in enumerate(files, start=1):
        filename = filepath.name
        print(f"Processing {i}/{total}: {filename}", end=" ... ", flush=True)

        # a. Word count check
        try:
            content = filepath.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            print(f"SKIP (read error: {e})")
            log_failure(filename, f"read error: {e}")
            failed += 1
            continue

        word_count = len(content.split())
        if word_count < 100:
            print(f"SKIP (only {word_count} words — minimum 100)")
            skipped_short += 1
            continue

        # b. Content hash check — catches same content under a different filename
        h = content_hash(content)
        if h in seen_hashes:
            print(f"SKIP (duplicate of '{seen_hashes[h]}')")
            skipped_dupe += 1
            continue

        # c. Notion filename dedup check
        if already_in_notion(filename):
            print("SKIP (already in Notion)")
            seen_hashes[h] = filename  # backfill hash so future runs are faster
            save_seen_hashes(seen_hashes)
            skipped_exist += 1
            continue

        # d. Rate-limit wait (except before the very first call)
        if api_calls_made > 0:
            time.sleep(RATE_LIMIT_WAIT)

        # e. Claude extraction
        try:
            metadata = extract_metadata(content)
            api_calls_made += 1
        except Exception as e:
            print(f"FAIL (Claude error: {e})")
            log_failure(filename, f"Claude extraction failed: {e}")
            failed += 1
            continue

        # f. Novelty check + write to Reading List (not KB)
        try:
            import dedup_engine, reading_list, re as _re, json as _json
            from pathlib import Path as _Path

            key_concepts = metadata.get("key_concepts", [])
            body_text    = "\n".join(metadata.get("concrete_learnings", []) + metadata.get("key_claims", []))
            dedup = dedup_engine.check_novelty_standalone(
                key_concepts=key_concepts,
                body_text=body_text,
                claims=metadata.get("key_claims", []),
                learnings=metadata.get("concrete_learnings", []),
            )

            rl_page_id = reading_list.write_to_reading_list(metadata, dedup, filename, source_type)

            # Cache for promote step
            safe = _re.sub(r"[^\w\-.]", "_", filename)[:60]
            cache_key = f"file_{safe}"
            exports_dir = _Path("exports"); exports_dir.mkdir(exist_ok=True)
            (exports_dir / f"{cache_key}.json").write_text(
                _json.dumps({**metadata, "_meta": {"cache_key": cache_key, "source": filename,
                                                    "source_type": source_type, "rl_page_id": rl_page_id}},
                            indent=2, ensure_ascii=False)
            )

            seen_hashes[h] = filename
            save_seen_hashes(seen_hashes)
            verdict = dedup.get("verdict", "?")
            n_new   = len(dedup.get("concept_report", {}).get("new", []))
            print(f"ADDED → Reading List  \"{metadata['title']}\"  [{verdict} | {n_new} new concepts]")
            added += 1
        except Exception as e:
            print(f"FAIL (Reading List write: {e})")
            log_failure(filename, f"Reading List write failed: {e}")
            failed += 1
            continue

    print(
        f"\nSeeding complete. {added} added. "
        f"{skipped_exist} skipped (exist). "
        f"{skipped_dupe} skipped (duplicate content). "
        f"{skipped_short} skipped (too short). "
        f"{failed} failed."
    )

    if failed > 0:
        print(f"See {FAIL_LOG} for details.")


if __name__ == "__main__":
    main()
