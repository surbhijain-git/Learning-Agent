"""
eval/eval_agent.py
Judge LLM that evaluates Content Intelligence Agent outputs against the rubric.

Modes:
  spot    — score the last N entries (default 10) after a pipeline run
  weekly  — score all entries added in the last 7 days
  full    — score the entire KB (use sparingly)

Usage:
    python eval/eval_agent.py spot              # last 10 entries
    python eval/eval_agent.py spot --n 20
    python eval/eval_agent.py weekly
    python eval/eval_agent.py full
    python eval/eval_agent.py spot --dry-run    # print scores, don't write to Notion
"""

import os
import sys
import json
import time
import httpx
import argparse
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Allow imports from parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv, find_dotenv
import anthropic
from notion_client import Client

from eval.rubric import build_judge_prompt, WEIGHTS, PASS_THRESHOLD, REVIEW_THRESHOLD

load_dotenv(find_dotenv(usecwd=True), override=True)

ai     = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
notion = Client(auth=os.getenv("NOTION_TOKEN"))

NOTION_HEADERS = {
    "Authorization": f"Bearer {os.getenv('NOTION_TOKEN', '')}",
    "Notion-Version": "2022-06-28",
    "Content-Type": "application/json",
}

NOTION_DB_ID      = os.getenv("NOTION_DB_ID", "").strip("'\"")
NOTION_EVAL_DB_ID = os.getenv("NOTION_EVAL_DB_ID", "").strip("'\"")
SONNET            = "claude-sonnet-4-6"
RATE_WAIT         = 3  # seconds between Judge calls

NOTION_HEADERS = {
    "Authorization": f"Bearer {os.getenv('NOTION_TOKEN', '')}",
    "Notion-Version": "2022-06-28",
    "Content-Type": "application/json",
}

LOGS_DIR = Path(__file__).parent.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)


# ── Notion helpers ────────────────────────────────────────────────────────────
def _text(prop) -> str:
    items = prop.get("rich_text") or prop.get("title") or []
    return "".join(t.get("plain_text", "") for t in items)

def _select(prop) -> str:
    s = prop.get("select")
    return s["name"] if s else ""

def _multiselect(prop) -> list[str]:
    return [o["name"] for o in prop.get("multi_select", [])]

def _date(prop) -> str:
    d = prop.get("date")
    return d["start"] if d else ""


def _ensure_eval_db() -> str:
    """Return the Eval Results DB ID, searching Notion first before creating."""
    global NOTION_EVAL_DB_ID

    if NOTION_EVAL_DB_ID:
        try:
            notion.databases.retrieve(NOTION_EVAL_DB_ID)
            return NOTION_EVAL_DB_ID
        except Exception:
            NOTION_EVAL_DB_ID = ""  # reset — fall through to search/create

    # Search for an existing "Eval Results" DB before creating a new one
    # (prevents duplicate DBs across CI runs where NOTION_EVAL_DB_ID isn't set)
    search_resp = httpx.post(
        "https://api.notion.com/v1/search",
        headers=NOTION_HEADERS,
        json={"query": "Eval Results", "filter": {"value": "database", "property": "object"}},
    ).json()
    for result in search_resp.get("results", []):
        title_items = result.get("title", [])
        title = "".join(t.get("plain_text", "") for t in title_items)
        if title == "Eval Results":
            NOTION_EVAL_DB_ID = result["id"]
            print(f"  Found existing Eval Results DB: {NOTION_EVAL_DB_ID}")
            print(f"  Tip: add NOTION_EVAL_DB_ID={NOTION_EVAL_DB_ID} as a GitHub secret to skip this search.")
            return NOTION_EVAL_DB_ID

    # Not found — create it
    kb_db = notion.databases.retrieve(NOTION_DB_ID)
    parent = kb_db.get("parent", {})

    db = notion.databases.create(
        parent=parent,
        title=[{"type": "text", "text": {"content": "Eval Results"}}],
        properties={
            "KB_Title":     {"title": {}},
            "Eval_Score":   {"number": {"format": "number"}},
            "Eval_Verdict": {
                "select": {"options": [
                    {"name": "PASS",   "color": "green"},
                    {"name": "REVIEW", "color": "yellow"},
                    {"name": "FAIL",   "color": "red"},
                ]}
            },
            "Eval_Date":    {"date": {}},
            "Source_Type":  {"select": {"options": []}},
            "extraction_fidelity":  {"number": {"format": "number"}},
            "insight_depth":        {"number": {"format": "number"}},
            "novelty_calibration":  {"number": {"format": "number"}},
            "pipeline_integrity":   {"number": {"format": "number"}},
            "structural_quality":   {"number": {"format": "number"}},
            "strategic_relevance":  {"number": {"format": "number"}},
            "Summary":      {"rich_text": {}},
            "Notes":        {"rich_text": {}},
            "KB_Page_ID":   {"rich_text": {}},
        },
    )
    NOTION_EVAL_DB_ID = db["id"]

    # Save to .env
    from dotenv import set_key
    env_path = Path(__file__).parent.parent / ".env"
    set_key(str(env_path), "NOTION_EVAL_DB_ID", NOTION_EVAL_DB_ID)
    print(f"  Created Eval Results DB: {NOTION_EVAL_DB_ID}")
    return NOTION_EVAL_DB_ID


def _query_db(database_id: str, **kwargs) -> list[dict]:
    """Paginate a Notion database query."""
    results, cursor = [], None
    while True:
        body = {k: v for k, v in kwargs.items() if v is not None}
        if cursor:
            body["start_cursor"] = cursor
        resp = httpx.post(
            f"https://api.notion.com/v1/databases/{database_id}/query",
            headers=NOTION_HEADERS,
            json=body,
        ).json()
        results.extend(resp.get("results", []))
        if not resp.get("has_more"):
            break
        cursor = resp.get("next_cursor")
    return results


def fetch_entries(mode: str, n: int = 10) -> list[dict]:
    """Fetch KB entries based on mode."""
    kwargs = dict(sorts=[{"property": "Date_Added", "direction": "descending"}])

    if mode == "spot":
        kwargs["page_size"] = n
    elif mode == "weekly":
        week_ago = (datetime.now(timezone.utc) - timedelta(days=7)).date().isoformat()
        kwargs["filter"] = {"property": "Date_Added", "date": {"on_or_after": week_ago}}

    pages = _query_db(NOTION_DB_ID, **kwargs)
    return pages[:n] if mode == "spot" else pages


# ── Page body reader (for key claims / learnings stored as blocks) ─────────────
def fetch_page_blocks(page_id: str) -> str:
    """Fetch page body blocks and return as plain text."""
    try:
        resp = notion.blocks.children.list(block_id=page_id)
        lines = []
        for block in resp.get("results", []):
            bt = block.get("type", "")
            rich = block.get(bt, {}).get("rich_text", [])
            text = "".join(t.get("plain_text", "") for t in rich)
            if text:
                lines.append(text)
        return "\n".join(lines)
    except Exception:
        return ""


# ── Judge LLM ─────────────────────────────────────────────────────────────────
JUDGE_SYSTEM = build_judge_prompt()


def judge_entry(entry: dict) -> dict:
    """Run the Judge LLM on a single KB entry. Returns score dict."""
    props = entry["properties"]

    title     = _text(props.get("Title", {}))
    summary   = _text(props.get("Summary", {}))
    concepts  = _multiselect(props.get("Key_Concepts", {}))
    verdict   = _select(props.get("Verdict", {})) or _select(props.get("Understanding_Level", {}))
    source_type = _select(props.get("Source_Type", {}))

    # Fetch body for claims and learnings
    body = fetch_page_blocks(entry["id"])

    user_msg = f"""Evaluate this KB entry:

Title: {title}
Source Type: {source_type}
Summary: {summary}
Key Concepts: {concepts}
Novelty Verdict: {verdict}

Page body (claims + learnings):
{body[:3000] if body else "(no body content)"}"""

    resp = ai.messages.create(
        model=SONNET,
        max_tokens=1024,
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )
    raw = resp.content[0].text.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    result = json.loads(raw)

    # Recalculate weighted score from returned scores (don't trust model's arithmetic)
    scores = result["scores"]
    weighted = sum(scores[k] * WEIGHTS[k] for k in WEIGHTS)
    result["weighted_score"] = round(weighted, 2)
    if weighted >= PASS_THRESHOLD:
        result["verdict"] = "PASS"
    elif weighted >= REVIEW_THRESHOLD:
        result["verdict"] = "REVIEW"
    else:
        result["verdict"] = "FAIL"

    return result


# ── Write results to separate Eval Results DB ─────────────────────────────────
def write_eval_to_notion(eval_db_id: str, title: str, page_id: str, source_type: str, result: dict):
    """Creates a new row in the Eval Results database — never touches the main KB."""
    scores    = result.get("scores", {})
    reasoning = result.get("reasoning", {})
    notes     = " | ".join(f"{k}: {v}" for k, v in reasoning.items())[:2000]
    today     = datetime.now(timezone.utc).date().isoformat()

    notion.pages.create(
        parent={"database_id": eval_db_id},
        properties={
            "KB_Title":    {"title": [{"type": "text", "text": {"content": title[:200]}}]},
            "Eval_Score":  {"number": result["weighted_score"]},
            "Eval_Verdict":{"select": {"name": result["verdict"]}},
            "Eval_Date":   {"date": {"start": today}},
            "Source_Type": {"select": {"name": source_type}} if source_type else {},
            "extraction_fidelity": {"number": scores.get("extraction_fidelity")},
            "insight_depth":       {"number": scores.get("insight_depth")},
            "novelty_calibration": {"number": scores.get("novelty_calibration")},
            "pipeline_integrity":  {"number": scores.get("pipeline_integrity")},
            "structural_quality":  {"number": scores.get("structural_quality")},
            "strategic_relevance": {"number": scores.get("strategic_relevance")},
            "Summary":    {"rich_text": [{"type": "text", "text": {"content": result.get("summary", "")[:2000]}}]},
            "Notes":      {"rich_text": [{"type": "text", "text": {"content": notes}}]},
            "KB_Page_ID": {"rich_text": [{"type": "text", "text": {"content": page_id}}]},
        },
    )


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Eval agent — Judge LLM for KB quality")
    parser.add_argument("mode", choices=["spot", "weekly", "full"], help="Evaluation mode")
    parser.add_argument("--n",       type=int, default=10, help="Number of entries for spot mode")
    parser.add_argument("--dry-run", action="store_true",  help="Print scores, don't write to Notion")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Eval Agent — mode: {args.mode}{' (dry run)' if args.dry_run else ''}")
    print(f"{'='*60}\n")

    eval_db_id = None
    if not args.dry_run:
        eval_db_id = _ensure_eval_db()
        print(f"  Writing results to Eval Results DB (separate from KB)\n")

    entries = fetch_entries(args.mode, n=args.n)
    print(f"Evaluating {len(entries)} entries...\n")

    results = []
    pass_count = review_count = fail_count = 0

    for i, entry in enumerate(entries, 1):
        props       = entry["properties"]
        title       = _text(props.get("Title", {}))
        source_type = _select(props.get("Source_Type", {}))
        print(f"[{i:02d}/{len(entries):02d}] {title[:65]}")

        try:
            result = judge_entry(entry)
            results.append({"title": title, "page_id": entry["id"], **result})

            score   = result["weighted_score"]
            verdict = result["verdict"]
            summary = result.get("summary", "")

            verdict_symbol = {"PASS": "✅", "REVIEW": "🟡", "FAIL": "❌"}.get(verdict, "?")
            print(f"       {verdict_symbol} {verdict}  score: {score:.2f}  — {summary}")

            for dim, s in result["scores"].items():
                bar = "█" * s + "░" * (5 - s)
                print(f"         {dim:25} {bar} {s}/5")

            if verdict == "PASS":     pass_count += 1
            elif verdict == "REVIEW": review_count += 1
            else:                     fail_count += 1

            if not args.dry_run:
                write_eval_to_notion(eval_db_id, title, entry["id"], source_type, result)

            if i < len(entries):
                time.sleep(RATE_WAIT)

        except Exception as e:
            print(f"       ⚠️  Error: {e}")

        print()

    # Summary
    total = len(entries)
    print(f"{'='*60}")
    print(f"Results: {total} evaluated")
    print(f"  ✅ PASS:   {pass_count}  ({pass_count/total*100:.0f}%)")
    print(f"  🟡 REVIEW: {review_count}  ({review_count/total*100:.0f}%)")
    print(f"  ❌ FAIL:   {fail_count}  ({fail_count/total*100:.0f}%)")

    avg_score = sum(r["weighted_score"] for r in results) / len(results) if results else 0
    print(f"  Average score: {avg_score:.2f}")
    print(f"{'='*60}\n")

    # Log results
    log_path = LOGS_DIR / f"eval_{args.mode}_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    log_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Full results logged to: {log_path}")

    if not args.dry_run:
        print("Scores written to Notion KB (Eval_Score, Eval_Verdict, Eval_Notes fields).")


if __name__ == "__main__":
    main()
