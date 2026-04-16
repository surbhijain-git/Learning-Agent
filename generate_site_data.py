"""
generate_site_data.py
Queries Notion KB + Reading List and writes docs/data.json for the dashboard.
Run manually or automatically by GitHub Actions after each pipeline run.

Usage:
    python generate_site_data.py
"""

import os
import json
import httpx
from datetime import datetime, timezone, timedelta
from pathlib import Path
from collections import Counter

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(usecwd=True), override=True)

NOTION_TOKEN           = os.getenv("NOTION_TOKEN", "")
NOTION_DB_ID           = os.getenv("NOTION_DB_ID", "").strip("'\"")
NOTION_READING_LIST_ID = os.getenv("NOTION_READING_LIST_ID", "").strip("'\"")
OUT_PATH               = Path("docs/data.json")
OUT_PATH.parent.mkdir(exist_ok=True)

HEADERS = {
    "Authorization": f"Bearer {NOTION_TOKEN}",
    "Notion-Version": "2022-06-28",
    "Content-Type": "application/json",
}


def _query_db(database_id: str, **kwargs) -> list[dict]:
    """Paginate a Notion database query."""
    results, cursor = [], None
    while True:
        body = {k: v for k, v in kwargs.items() if v is not None}
        if cursor:
            body["start_cursor"] = cursor
        resp = httpx.post(
            f"https://api.notion.com/v1/databases/{database_id}/query",
            headers=HEADERS,
            json=body,
        )
        data = resp.json()
        results.extend(data.get("results", []))
        if not data.get("has_more"):
            break
        cursor = data.get("next_cursor")
    return results


def _text(prop) -> str:
    items = prop.get("rich_text") or prop.get("title") or []
    return "".join(t.get("plain_text", "") for t in items)


def _select(prop) -> str:
    s = prop.get("select")
    return s["name"] if s else ""


def _date(prop) -> str:
    d = prop.get("date")
    return d["start"] if d else ""


def _multiselect(prop) -> list[str]:
    return [o["name"] for o in prop.get("multi_select", [])]


def main():
    print("Fetching KB entries...")
    kb_pages = _query_db(
        NOTION_DB_ID,
        sorts=[{"property": "Date_Added", "direction": "descending"}],
    )
    print(f"  {len(kb_pages)} total entries")

    week_ago = (datetime.now(timezone.utc) - timedelta(days=7)).date().isoformat()
    added_this_week = sum(
        1 for p in kb_pages
        if _date(p["properties"].get("Date_Added", {})) >= week_ago
    )

    all_concepts = []
    source_counts = Counter()
    for p in kb_pages:
        props = p["properties"]
        all_concepts.extend(_multiselect(props.get("Key_Concepts", {})))
        st = _select(props.get("Source_Type", {}))
        if st:
            source_counts[st] += 1

    top_concepts   = [c for c, _ in Counter(all_concepts).most_common(20)]
    unique_concepts = len(set(all_concepts))

    recent_entries = []
    for p in kb_pages[:10]:
        props = p["properties"]
        recent_entries.append({
            "title":       _text(props.get("Title", {})),
            "summary":     _text(props.get("Summary", {})),
            "source_type": _select(props.get("Source_Type", {})),
            "verdict":     _select(props.get("Verdict", {})) or ("NEW" if props.get("Is_New_Info", {}).get("checkbox") else ""),
            "date_added":  _date(props.get("Date_Added", {})),
        })

    print("Fetching Reading List...")
    rl_pages = _query_db(
        NOTION_READING_LIST_ID,
        filter={"property": "Status", "select": {"equals": "To Read"}},
    )
    reading_list_queue = len(rl_pages)
    print(f"  {reading_list_queue} items pending")

    data = {
        "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "stats": {
            "total_kb_entries":   len(kb_pages),
            "added_this_week":    added_this_week,
            "reading_list_queue": reading_list_queue,
            "unique_concepts":    unique_concepts,
        },
        "recent_entries":   recent_entries,
        "top_concepts":     top_concepts,
        "source_breakdown": dict(source_counts),
    }

    OUT_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"Written to {OUT_PATH}")
    print(f"  KB: {len(kb_pages)} entries | This week: {added_this_week} | Queue: {reading_list_queue}")


if __name__ == "__main__":
    main()
