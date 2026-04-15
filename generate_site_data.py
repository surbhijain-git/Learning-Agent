"""
generate_site_data.py
Queries Notion KB + Reading List and writes docs/data.json for the GitHub Pages dashboard.
Run manually or automatically by GitHub Actions after each pipeline run.

Usage:
    python generate_site_data.py
"""

import os
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from collections import Counter

from dotenv import load_dotenv, find_dotenv
from notion_client import Client

load_dotenv(find_dotenv(usecwd=True), override=True)
notion = Client(auth=os.getenv("NOTION_TOKEN"))

NOTION_DB_ID            = os.getenv("NOTION_DB_ID", "").strip("'\"")
NOTION_READING_LIST_ID  = os.getenv("NOTION_READING_LIST_ID", "").strip("'\"")
OUT_PATH                = Path("docs/data.json")
OUT_PATH.parent.mkdir(exist_ok=True)


def _paginate(fn, **kwargs) -> list[dict]:
    results, cursor = [], None
    while True:
        resp = fn(**kwargs, start_cursor=cursor) if cursor else fn(**kwargs)
        results.extend(resp.get("results", []))
        if not resp.get("has_more"):
            break
        cursor = resp.get("next_cursor")
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
    kb_pages = _paginate(
        notion.databases.query,
        database_id=NOTION_DB_ID,
        sorts=[{"property": "Date_Added", "direction": "descending"}],
    )

    print(f"  {len(kb_pages)} total entries")

    # Count added this week
    week_ago = (datetime.now(timezone.utc) - timedelta(days=7)).date().isoformat()
    added_this_week = sum(
        1 for p in kb_pages
        if _date(p["properties"].get("Date_Added", {})) >= week_ago
    )

    # Top concepts
    all_concepts = []
    source_counts = Counter()
    for p in kb_pages:
        props = p["properties"]
        concepts = _multiselect(props.get("Key_Concepts", {}))
        all_concepts.extend(concepts)
        st = _select(props.get("Source_Type", {}))
        if st:
            source_counts[st] += 1

    top_concepts = [c for c, _ in Counter(all_concepts).most_common(20)]
    unique_concepts = len(set(all_concepts))

    # Recent entries (last 10)
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

    # Reading list queue count
    print("Fetching Reading List...")
    rl_pages = _paginate(
        notion.databases.query,
        database_id=NOTION_READING_LIST_ID,
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
