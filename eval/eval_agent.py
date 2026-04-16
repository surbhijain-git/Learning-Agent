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

NOTION_DB_ID = os.getenv("NOTION_DB_ID", "").strip("'\"")
SONNET       = "claude-sonnet-4-6"
RATE_WAIT    = 3  # seconds between Judge calls

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


def _ensure_eval_fields():
    """Add Eval_Score, Eval_Verdict, Eval_Notes fields to KB if missing."""
    db = notion.databases.retrieve(NOTION_DB_ID)
    existing = set(db["properties"].keys())
    updates = {}
    if "Eval_Score" not in existing:
        updates["Eval_Score"] = {"number": {"format": "number"}}
    if "Eval_Verdict" not in existing:
        updates["Eval_Verdict"] = {
            "select": {
                "options": [
                    {"name": "PASS",   "color": "green"},
                    {"name": "REVIEW", "color": "yellow"},
                    {"name": "FAIL",   "color": "red"},
                ]
            }
        }
    if "Eval_Notes" not in existing:
        updates["Eval_Notes"] = {"rich_text": {}}
    if updates:
        notion.databases.update(database_id=NOTION_DB_ID, properties=updates)
        print(f"  Added eval fields to KB: {list(updates.keys())}")


def fetch_entries(mode: str, n: int = 10) -> list[dict]:
    """Fetch KB entries based on mode."""
    kwargs = dict(
        database_id=NOTION_DB_ID,
        sorts=[{"property": "Date_Added", "direction": "descending"}],
    )

    if mode == "spot":
        kwargs["page_size"] = n
    elif mode == "weekly":
        week_ago = (datetime.now(timezone.utc) - timedelta(days=7)).date().isoformat()
        kwargs["filter"] = {"property": "Date_Added", "date": {"on_or_after": week_ago}}

    pages, cursor = [], None
    while True:
        resp = notion.databases.query(**kwargs, start_cursor=cursor) if cursor \
              else notion.databases.query(**kwargs)
        pages.extend(resp.get("results", []))
        if not resp.get("has_more") or (mode == "spot" and len(pages) >= n):
            break
        cursor = resp.get("next_cursor")

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


# ── Write results back to Notion ──────────────────────────────────────────────
def write_eval_to_notion(page_id: str, result: dict):
    reasoning = result.get("reasoning", {})
    notes = " | ".join(f"{k}: {v}" for k, v in reasoning.items())
    summary_line = result.get("summary", "")
    full_notes = f"{summary_line} | {notes}"[:2000]

    notion.pages.update(
        page_id=page_id,
        properties={
            "Eval_Score":   {"number": result["weighted_score"]},
            "Eval_Verdict": {"select": {"name": result["verdict"]}},
            "Eval_Notes":   {"rich_text": [{"type": "text", "text": {"content": full_notes}}]},
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

    if not args.dry_run:
        _ensure_eval_fields()

    entries = fetch_entries(args.mode, n=args.n)
    print(f"Evaluating {len(entries)} entries...\n")

    results = []
    pass_count = review_count = fail_count = 0

    for i, entry in enumerate(entries, 1):
        title = _text(entry["properties"].get("Title", {}))
        print(f"[{i:02d}/{len(entries):02d}] {title[:65]}")

        try:
            result = judge_entry(entry)
            results.append({"title": title, "page_id": entry["id"], **result})

            score   = result["weighted_score"]
            verdict = result["verdict"]
            summary = result.get("summary", "")

            verdict_symbol = {"PASS": "✅", "REVIEW": "🟡", "FAIL": "❌"}.get(verdict, "?")
            print(f"       {verdict_symbol} {verdict}  score: {score:.2f}  — {summary}")

            # Per-dimension breakdown
            for dim, s in result["scores"].items():
                bar = "█" * s + "░" * (5 - s)
                print(f"         {dim:25} {bar} {s}/5")

            if verdict == "PASS":   pass_count += 1
            elif verdict == "REVIEW": review_count += 1
            else:                    fail_count += 1

            if not args.dry_run:
                write_eval_to_notion(entry["id"], result)

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
