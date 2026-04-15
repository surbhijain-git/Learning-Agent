"""
retitle_kb.py
One-time script to retitle existing KB entries from insight-sentences to short topic labels.
Uses the existing Summary field as context for Claude to generate a clean title.

Usage:
    python retitle_kb.py            # preview only (dry run)
    python retitle_kb.py --apply    # actually update Notion
"""

import os
import sys
import time
import argparse
from dotenv import load_dotenv, find_dotenv
import anthropic
from notion_client import Client

load_dotenv(find_dotenv(usecwd=True), override=True)
notion = Client(auth=os.getenv("NOTION_TOKEN"))
ai     = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

NOTION_DB_ID = os.getenv("NOTION_DB_ID").strip("'\"")
HAIKU        = "claude-haiku-4-5-20251001"


def _text(prop) -> str:
    items = prop.get("rich_text") or prop.get("title") or []
    return "".join(t.get("plain_text", "") for t in items)


def generate_short_title(current_title: str, summary: str, source: str) -> str:
    prompt = (
        f"Current title: {current_title}\n"
        f"Summary: {summary}\n"
        f"Source: {source}\n\n"
        "Generate a short topic label (3-6 words) that names the subject of this content. "
        "Do NOT write an insight or implication sentence. "
        "Examples of good labels: 'Edge AI Economics', 'Cursor GTM Strategy', "
        "'LLM Context Engineering', 'AI Regulation Week', 'Agentic Architecture Patterns'. "
        "Return ONLY the label, nothing else."
    )
    resp = ai.messages.create(
        model=HAIKU,
        max_tokens=30,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text.strip().strip('"').strip("'")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Actually update Notion (default is dry run)")
    args = parser.parse_args()

    dry_run = not args.apply
    if dry_run:
        print("DRY RUN — pass --apply to update Notion\n")

    # Fetch all KB entries
    pages, cursor = [], None
    while True:
        resp = notion.databases.query(database_id=NOTION_DB_ID, start_cursor=cursor) if cursor \
              else notion.databases.query(database_id=NOTION_DB_ID)
        pages.extend(resp.get("results", []))
        if not resp.get("has_more"):
            break
        cursor = resp.get("next_cursor")

    print(f"Found {len(pages)} KB entries\n")

    updated = 0
    for i, page in enumerate(pages, 1):
        props         = page["properties"]
        current_title = _text(props.get("Title", {}))
        summary       = _text(props.get("Summary", {}))
        source        = _text(props.get("Source", {}))
        page_id       = page["id"]

        new_title = generate_short_title(current_title, summary, source)

        print(f"[{i:02d}] {current_title[:70]}")
        print(f"   → {new_title}")

        if not dry_run:
            notion.pages.update(
                page_id=page_id,
                properties={
                    "Title": {"title": [{"type": "text", "text": {"content": new_title}}]}
                },
            )
            updated += 1
            time.sleep(0.4)  # avoid rate limiting

        print()

    if dry_run:
        print(f"Dry run complete. Run with --apply to update all {len(pages)} entries.")
    else:
        print(f"Done. Updated {updated}/{len(pages)} entries.")


if __name__ == "__main__":
    main()
