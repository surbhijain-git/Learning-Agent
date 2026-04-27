"""
topic_concentration.py
Weekly topic concentration report for the Content Intelligence Agent.

What it does:
  1. Reads all KB entries from Notion (title + summary + Key_Concepts)
  2. Classifies each into one of 5 learning-goal categories using Haiku
  3. Computes % actual vs % target for each category
  4. Saves logs/topic_YYYY-MM-DD.json  (local + committed to GitHub)
  5. Writes docs/topic_data.json       (read by the HTML dashboard)
  6. Updates the NYW Project Notion page with a formatted callout section

Run:
    python topic_concentration.py          # full run
    python topic_concentration.py --dry-run  # print report, skip Notion write
"""

import os
import sys
import json
import time
import httpx
import argparse
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
import anthropic

load_dotenv(find_dotenv(usecwd=True), override=True)

# ── Constants ────────────────────────────────────────────────────────────────

ANTHROPIC_API_KEY  = os.getenv("ANTHROPIC_API_KEY", "")
NOTION_TOKEN       = os.getenv("NOTION_TOKEN", "").strip("'\"")
NOTION_DB_ID       = os.getenv("NOTION_DB_ID", "").strip("'\"")
NOTION_PARENT_PAGE_ID = os.getenv("NOTION_PARENT_PAGE_ID", "").strip("'\"")
# NYW Project page — format with hyphens for API calls
_raw = NOTION_PARENT_PAGE_ID.replace("-", "")
NYW_PAGE_ID = f"{_raw[:8]}-{_raw[8:12]}-{_raw[12:16]}-{_raw[16:20]}-{_raw[20:]}" if len(_raw) == 32 else NOTION_PARENT_PAGE_ID

HAIKU   = "claude-haiku-4-5-20251001"
BATCH   = 15          # entries per Haiku classification call
RATE_S  = 2.0         # seconds between API calls (free tier guard)

NOTION_HEADERS = {
    "Authorization": f"Bearer {NOTION_TOKEN}",
    "Notion-Version": "2022-06-28",
    "Content-Type": "application/json",
}

# ── Learning Goal Categories ─────────────────────────────────────────────────

CATEGORIES = [
    {
        "id":       "competitive",
        "question": "Who wins?",
        "label":    "Competitive Landscape",
        "target":   25,
        "desc":     "Market dynamics, vendor battles, adoption curves, moats, business model comparisons",
        "color":    "#2563eb",   # blue
    },
    {
        "id":       "practice",
        "question": "How does work change?",
        "label":    "AI in Practice",
        "target":   35,
        "desc":     "Workflows, agent architecture, prompt engineering, tool use, implementation case studies",
        "color":    "#16a34a",   # green
    },
    {
        "id":       "human",
        "question": "Who does what?",
        "label":    "Human & Org Impact",
        "target":   20,
        "desc":     "Role displacement, upskilling, org design, leadership, team structure with AI",
        "color":    "#7c3aed",   # purple
    },
    {
        "id":       "emerging",
        "question": "What's being built?",
        "label":    "Emerging Space",
        "target":   15,
        "desc":     "New products, research breakthroughs, capability announcements, early-stage ideas",
        "color":    "#d97706",   # amber
    },
    {
        "id":       "risk",
        "question": "What could go wrong?",
        "label":    "Risk & Governance",
        "target":    5,
        "desc":     "Regulation, safety, bias, hallucination risks, alignment, policy",
        "color":    "#dc2626",   # red
    },
]

CAT_IDS = [c["id"] for c in CATEGORIES]
CAT_BY_ID = {c["id"]: c for c in CATEGORIES}

# ── Notion helpers ────────────────────────────────────────────────────────────

def _notion_post(path: str, payload: dict) -> dict:
    r = httpx.post(
        f"https://api.notion.com/v1/{path}",
        headers=NOTION_HEADERS,
        json=payload,
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def _notion_get(path: str) -> dict:
    r = httpx.get(
        f"https://api.notion.com/v1/{path}",
        headers=NOTION_HEADERS,
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def _notion_patch(path: str, payload: dict) -> dict:
    r = httpx.patch(
        f"https://api.notion.com/v1/{path}",
        headers=NOTION_HEADERS,
        json=payload,
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def _notion_delete(path: str) -> dict:
    r = httpx.delete(
        f"https://api.notion.com/v1/{path}",
        headers=NOTION_HEADERS,
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def fetch_all_kb_entries() -> list[dict]:
    """Return list of {title, summary, concepts} for every KB entry."""
    entries = []
    cursor = None
    while True:
        payload: dict = {"page_size": 100}
        if cursor:
            payload["start_cursor"] = cursor
        data = _notion_post(f"databases/{NOTION_DB_ID}/query", payload)
        for page in data.get("results", []):
            props = page.get("properties", {})
            # Title
            title_arr = props.get("Title", {}).get("title", [])
            title = "".join(t.get("plain_text", "") for t in title_arr).strip()
            # Summary — rich_text
            summary_arr = props.get("Summary", {}).get("rich_text", [])
            summary = "".join(t.get("plain_text", "") for t in summary_arr).strip()[:400]
            # Key_Concepts — multi_select
            concepts_arr = props.get("Key_Concepts", {}).get("multi_select", [])
            concepts = ", ".join(c["name"] for c in concepts_arr)
            if title:
                entries.append({
                    "id":       page["id"],
                    "title":    title,
                    "summary":  summary,
                    "concepts": concepts,
                })
        if not data.get("has_more"):
            break
        cursor = data.get("next_cursor")
    return entries


# ── Classification ────────────────────────────────────────────────────────────

CLASSIFY_SYSTEM = """You are a topic classifier for a personal AI learning KB.
Classify each entry into exactly one of these 5 categories based on its title, summary, and concepts:

competitive  — Who wins? Market dynamics, vendor battles, adoption curves, business models, moats
practice     — How does work change? Workflows, agent architecture, implementation, tool use, case studies
human        — Who does what? Role displacement, upskilling, org design, leadership with AI
emerging     — What's being built? New products, research, capability announcements, early-stage
risk         — What could go wrong? Regulation, safety, bias, alignment, policy

Return a JSON array of objects, one per entry, in the same order as input:
[{"idx": 0, "id": "...", "cat": "practice"}, ...]

Use only the category IDs above. No explanation. No markdown fences."""


def classify_batch(entries: list[dict], ai_client: anthropic.Anthropic) -> list[dict]:
    """Classify a batch of entries. Returns list of {idx, id, cat}."""
    items_text = "\n".join(
        f'[{i}] id={e["id"]}\n  title: {e["title"]}\n  summary: {e["summary"][:200]}\n  concepts: {e["concepts"]}'
        for i, e in enumerate(entries)
    )
    msg = ai_client.messages.create(
        model=HAIKU,
        max_tokens=1024,
        system=CLASSIFY_SYSTEM,
        messages=[{"role": "user", "content": f"Classify these {len(entries)} entries:\n\n{items_text}"}],
    )
    raw = msg.content[0].text.strip()
    # Strip any accidental markdown fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw)


def classify_all(entries: list[dict], ai_client: anthropic.Anthropic) -> dict[str, str]:
    """Returns {entry_id: category_id} for all entries."""
    id_to_cat: dict[str, str] = {}
    total = len(entries)
    for start in range(0, total, BATCH):
        batch = entries[start: start + BATCH]
        print(f"  Classifying entries {start+1}–{min(start+BATCH, total)} of {total}…")
        try:
            results = classify_batch(batch, ai_client)
            for r in results:
                cat = r.get("cat", "").strip().lower()
                if cat not in CAT_IDS:
                    cat = "practice"   # safe fallback
                id_to_cat[r["id"]] = cat
        except Exception as e:
            print(f"  ⚠ Batch classification failed: {e} — falling back to 'practice' for batch")
            for entry in batch:
                id_to_cat[entry["id"]] = "practice"
        if start + BATCH < total:
            time.sleep(RATE_S)
    return id_to_cat


# ── Report builder ────────────────────────────────────────────────────────────

def build_report(entries: list[dict], id_to_cat: dict[str, str]) -> dict:
    """Build the full concentration report dict."""
    total = len(entries)
    # Group entry titles by category
    cat_entries: dict[str, list[str]] = {c: [] for c in CAT_IDS}
    for e in entries:
        cat = id_to_cat.get(e["id"], "practice")
        cat_entries[cat].append(e["title"])

    categories_out = []
    for c in CATEGORIES:
        count = len(cat_entries[c["id"]])
        pct   = round(count / total * 100, 1) if total else 0
        delta = round(pct - c["target"], 1)
        categories_out.append({
            "id":       c["id"],
            "question": c["question"],
            "label":    c["label"],
            "target":   c["target"],
            "count":    count,
            "pct":      pct,
            "delta":    delta,     # positive = over-indexed, negative = under-indexed
            "color":    c["color"],
            "entries":  cat_entries[c["id"]][:10],  # top 10 titles for reference
        })

    return {
        "generated":   datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "total_entries": total,
        "categories":  categories_out,
    }


def print_report(report: dict) -> None:
    print(f"\n{'─'*60}")
    print(f"  TOPIC CONCENTRATION REPORT — {report['generated']}")
    print(f"  Total KB entries: {report['total_entries']}")
    print(f"{'─'*60}")
    for c in report["categories"]:
        bar_filled = int(c["pct"] / 2)
        bar_target = int(c["target"] / 2)
        arrow = "▲" if c["delta"] > 2 else ("▼" if c["delta"] < -2 else "~")
        print(f"  {c['question']:<28} {c['pct']:>5.1f}%  (target {c['target']}%  {arrow}{abs(c['delta']):.1f}pp)  [{c['count']} entries]")
    print(f"{'─'*60}\n")


# ── Save locally ──────────────────────────────────────────────────────────────

def save_logs(report: dict) -> Path:
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    path = logs_dir / f"topic_{report['generated']}.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"  ✓ Saved {path}")
    return path


def save_topic_data(report: dict) -> Path:
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)
    path = docs_dir / "topic_data.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"  ✓ Saved {path}")
    return path


# ── Notion page update ────────────────────────────────────────────────────────

SECTION_MARKER = "📊 Topic Concentration"

def _rich(text: str, bold: bool = False, color: str | None = None) -> dict:
    t: dict = {"type": "text", "text": {"content": text}}
    if bold or color:
        t["annotations"] = {}
        if bold:
            t["annotations"]["bold"] = True
        if color:
            t["annotations"]["color"] = color
    return t


def _bar(pct: float, target: int) -> str:
    filled = round(pct / 2)
    empty  = 50 - filled
    arrow  = "▲" if (pct - target) > 2 else ("▼" if (pct - target) < -2 else "≈")
    return f"{'█' * filled}{'░' * empty}  {pct:.1f}% (target {target}%  {arrow})"


def build_notion_blocks(report: dict) -> list[dict]:
    """Build Notion block list for the concentration section."""
    blocks = []

    # Heading
    blocks.append({
        "object": "block", "type": "heading_2",
        "heading_2": {"rich_text": [_rich(SECTION_MARKER)]}
    })

    # Subtitle callout
    blocks.append({
        "object": "block", "type": "callout",
        "callout": {
            "icon": {"type": "emoji", "emoji": "🎯"},
            "color": "gray_background",
            "rich_text": [_rich(
                f"Generated {report['generated']} · {report['total_entries']} KB entries classified · "
                "Targets: Who wins 25% · Work change 35% · Who does what 20% · Being built 15% · Risks 5%"
            )],
        }
    })

    # One paragraph per category
    for c in report["categories"]:
        delta_str = f"+{c['delta']:.1f}" if c['delta'] >= 0 else f"{c['delta']:.1f}"
        status = "on target ✅" if abs(c["delta"]) <= 3 else (
            f"over-indexed by {abs(c['delta']):.0f}pp ⚠️" if c["delta"] > 0
            else f"under-indexed by {abs(c['delta']):.0f}pp 📉"
        )
        bar = _bar(c["pct"], c["target"])
        blocks.append({
            "object": "block", "type": "paragraph",
            "paragraph": {"rich_text": [
                _rich(f"{c['question']}  ({c['label']})\n", bold=True),
                _rich(bar + f"\n  {c['count']} entries · {status}"),
            ]}
        })

    # Recommendation paragraph
    over  = [c for c in report["categories"] if c["delta"] > 3]
    under = [c for c in report["categories"] if c["delta"] < -3]
    recs = []
    if over:
        recs.append("Reduce: " + ", ".join('"' + c['question'] + '"' for c in over))
    if under:
        recs.append("Seek more: " + ", ".join('"' + c['question'] + '"' for c in under))
    if recs:
        blocks.append({
            "object": "block", "type": "callout",
            "callout": {
                "icon": {"type": "emoji", "emoji": "\U0001f4a1"},
                "color": "yellow_background",
                "rich_text": [_rich("Rebalancing nudge: " + " \u00b7 ".join(recs))],
            }
        })
    else:
        blocks.append({
            "object": "block", "type": "callout",
            "callout": {
                "icon": {"type": "emoji", "emoji": "\u2705"},
                "color": "green_background",
                "rich_text": [_rich("Learning mix is on target across all categories.")],
            }
        })

    # Divider
    blocks.append({"object": "block", "type": "divider", "divider": {}})
    return blocks


def _find_existing_section_block_id(page_id: str) -> str | None:
    """Find the heading_2 block with SECTION_MARKER text, if it exists."""
    cursor = None
    while True:
        url = f"blocks/{page_id}/children"
        if cursor:
            url += f"?start_cursor={cursor}"
        data = _notion_get(url)
        for block in data.get("results", []):
            btype = block.get("type", "")
            texts = block.get(btype, {}).get("rich_text", [])
            plain = "".join(t.get("plain_text", "") for t in texts)
            if SECTION_MARKER in plain and btype == "heading_2":
                return block["id"]
        if not data.get("has_more"):
            break
        cursor = data.get("next_cursor")
    return None


def _delete_block_and_children(block_id: str) -> None:
    """Delete a block (and all its children, which Notion handles automatically)."""
    try:
        _notion_delete(f"blocks/{block_id}")
    except Exception as e:
        print(f"  ⚠ Could not delete block {block_id}: {e}")


def _get_blocks_after(page_id: str, heading_id: str) -> list[str]:
    """Get IDs of blocks that belong to the existing section (heading + content until next h2 or divider-after-h2)."""
    block_ids = [heading_id]
    cursor = None
    found_heading = False
    while True:
        url = f"blocks/{page_id}/children"
        if cursor:
            url += f"?start_cursor={cursor}"
        data = _notion_get(url)
        for block in data.get("results", []):
            if block["id"] == heading_id:
                found_heading = True
                continue
            if not found_heading:
                continue
            btype = block.get("type", "")
            # Stop at the next heading_2 (different section)
            if btype == "heading_2":
                texts = block.get(btype, {}).get("rich_text", [])
                plain = "".join(t.get("plain_text", "") for t in texts)
                if SECTION_MARKER not in plain:
                    return block_ids
            block_ids.append(block["id"])
            # Stop after a divider (section terminator)
            if btype == "divider":
                return block_ids
        if not data.get("has_more"):
            break
        cursor = data.get("next_cursor")
    return block_ids


def update_notion_page(report: dict) -> None:
    """Replace the Topic Concentration section on the NYW Project Notion page."""
    if not NYW_PAGE_ID or NYW_PAGE_ID.replace("-", "") == "":
        print("  ⚠ NOTION_PARENT_PAGE_ID not set — skipping Notion update")
        return

    print(f"  Updating Notion page {NYW_PAGE_ID}…")

    # Find and delete existing section
    existing_id = _find_existing_section_block_id(NYW_PAGE_ID)
    if existing_id:
        print(f"  Found existing section (block {existing_id}) — removing…")
        old_ids = _get_blocks_after(NYW_PAGE_ID, existing_id)
        for bid in old_ids:
            _delete_block_and_children(bid)
            time.sleep(0.3)

    # Append new blocks
    new_blocks = build_notion_blocks(report)
    # Notion API allows max 100 blocks per append; our section is ~10
    _notion_patch(f"blocks/{NYW_PAGE_ID}/children", {"children": new_blocks})
    print(f"  ✓ Notion page updated ({len(new_blocks)} blocks)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Topic Concentration weekly report")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print report but skip Notion write and file saves")
    args = parser.parse_args()

    ai_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    print("Topic Concentration Report")
    print("══════════════════════════")

    # 1. Fetch KB entries
    print("1/4  Fetching KB entries from Notion…")
    entries = fetch_all_kb_entries()
    print(f"     Found {len(entries)} entries")
    if not entries:
        print("     Nothing to classify. Exiting.")
        return

    # 2. Classify
    print("2/4  Classifying with Haiku…")
    id_to_cat = classify_all(entries, ai_client)

    # 3. Build report
    print("3/4  Building report…")
    report = build_report(entries, id_to_cat)
    print_report(report)

    if args.dry_run:
        print("  [dry-run] Skipping file saves and Notion update.")
        return

    # 4. Save outputs
    print("4/4  Saving outputs…")
    save_logs(report)
    save_topic_data(report)
    update_notion_page(report)

    print("\nDone ✓")


if __name__ == "__main__":
    main()
