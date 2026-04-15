"""
reading_list.py
Shared helpers for writing to the Notion Reading List and KB databases.
Imported by: ingest_queue.py, db_seeder.py

Two public functions:
  write_to_reading_list(summary, dedup, source, source_type) → rl_page_id
  write_to_kb(summary, source, source_type)                  → kb_page_id
"""

import os
from datetime import date
from dotenv import load_dotenv, find_dotenv
from notion_client import Client

load_dotenv(find_dotenv(usecwd=True), override=True)

notion = Client(auth=os.getenv("NOTION_TOKEN"))


def _get_rl_id() -> str:
    db_id = os.getenv("NOTION_READING_LIST_ID", "")
    if not db_id:
        raise SystemExit("NOTION_READING_LIST_ID not set — run: python notion_setup.py")
    return db_id


def _get_kb_id() -> str:
    db_id = os.getenv("NOTION_DB_ID", "")
    if not db_id:
        raise SystemExit("NOTION_DB_ID not set — run: python notion_setup.py")
    return db_id


# ── Block builders ────────────────────────────────────────────────────────────

def _h2(text: str) -> dict:
    return {
        "object": "block", "type": "heading_2",
        "heading_2": {"rich_text": [{"type": "text", "text": {"content": text}}]},
    }


def _bullet(text: str) -> dict:
    return {
        "object": "block", "type": "bulleted_list_item",
        "bulleted_list_item": {"rich_text": [{"type": "text", "text": {"content": text[:2000]}}]},
    }


def _bullet_blocks(items: list[str]) -> list[dict]:
    return [_bullet(item) for item in items]


def reading_list_body(concept_report: dict) -> list[dict]:
    """
    Three sections organised by novelty + Test placeholder.
    All key_claims and concrete_learnings are distributed across sections
    (concept_novelty_report classified them when called with claims + learnings).
    """
    new_items     = concept_report.get("new", [])
    related_items = concept_report.get("related", [])
    covered_items = concept_report.get("covered", [])

    blocks = [_h2("🟢 New — Learn These")]
    if new_items:
        blocks += [_bullet(x["text"]) for x in new_items]
    else:
        blocks.append(_bullet("No new concepts detected in this content."))

    blocks.append(_h2("🟡 New Angle — Connect to What You Know"))
    if related_items:
        blocks += [
            _bullet(f"{x['text']}  →  extends: \"{x['match_title'][:55]}\" ({x['score']:.2f})")
            for x in related_items
        ]
    else:
        blocks.append(_bullet("No related concepts detected."))

    blocks.append(_h2("🔁 Already Know — Validate"))
    if covered_items:
        blocks += [
            _bullet(f"{x['text']}  →  seen in: \"{x['match_title'][:55]}\" ({x['score']:.2f})")
            for x in covered_items
        ]
    else:
        blocks.append(_bullet("Nothing overlaps with your existing KB."))

    blocks.append(_h2("📝 Test"))
    blocks.append(_bullet("Test questions will be generated here in a future update."))

    return blocks


def kb_body(summary: dict) -> list[dict]:
    """Clean KB page body — no novelty labels."""
    return [
        _h2("What I Learned"),
        *_bullet_blocks(summary.get("concrete_learnings", [])),
        _h2("Key Claims"),
        *_bullet_blocks(summary.get("key_claims", [])),
    ]


# ── Reading List write ────────────────────────────────────────────────────────

def write_to_reading_list(
    summary: dict,
    dedup: dict,
    source: str,
    source_type: str,
) -> str:
    """
    Write extracted + novelty-checked content to Reading List.
    Handles both YouTube schema (core_argument / so_what) and file schema (summary field).
    Returns the created Notion page_id.
    """
    today          = str(date.today())
    verdict        = dedup.get("verdict", "NEW")
    score          = dedup.get("score", 0.0)
    concept_report = dedup.get("concept_report", {})

    n_new = len(concept_report.get("new", []))
    n_rel = len(concept_report.get("related", []))
    n_cov = len(concept_report.get("covered", []))
    new_chunks_text = f"{n_new} new · {n_rel} related · {n_cov} covered"

    core    = summary.get("core_argument") or summary.get("summary", "")
    so_what = summary.get("so_what", "")
    summary_text = (core + ("\n\n" + so_what if so_what else ""))[:2000]

    page = notion.pages.create(
        parent={"database_id": _get_rl_id()},
        properties={
            "Title":      {"title":     [{"type": "text", "text": {"content": summary.get("title", "Untitled")}}]},
            "Source":     {"rich_text": [{"type": "text", "text": {"content": source}}]},
            "Source_Type": {"select":   {"name": source_type}},
            "Date_Added": {"date":      {"start": today}},
            "Summary":    {"rich_text": [{"type": "text", "text": {"content": summary_text}}]},
            "Verdict":    {"select":    {"name": verdict}},
            "Similarity_Score": {"number": round(score, 4)},
            "New_Chunks": {"rich_text": [{"type": "text", "text": {"content": new_chunks_text}}]},
            "Status":     {"select":    {"name": "To Read"}},
        },
        children=reading_list_body(concept_report),
    )
    return page["id"]


# ── KB write (used by promote) ────────────────────────────────────────────────

def write_to_kb(summary: dict, source: str, source_type: str) -> str:
    """
    Write a clean entry to the KB after user marks Reading List entry as Read.
    No novelty labels — just knowledge.
    Returns the created KB page_id.
    """
    today   = str(date.today())
    core    = summary.get("core_argument") or summary.get("summary", "")
    so_what = summary.get("so_what", "")
    summary_text = (core + ("\n\n" + so_what if so_what else ""))[:2000]

    page = notion.pages.create(
        parent={"database_id": _get_kb_id()},
        properties={
            "Title":      {"title":     [{"type": "text", "text": {"content": summary.get("title", "Untitled")}}]},
            "Source":     {"rich_text": [{"type": "text", "text": {"content": source}}]},
            "Source_Type": {"select":   {"name": source_type}},
            "Date_Added": {"date":      {"start": today}},
            "Summary":    {"rich_text": [{"type": "text", "text": {"content": summary_text}}]},
            "Key_Concepts": {
                "multi_select": [{"name": t} for t in summary.get("key_concepts", [])]
            },
            "Understanding_Level": {"select": {"name": "New"}},
        },
        children=kb_body(summary),
    )
    return page["id"]
