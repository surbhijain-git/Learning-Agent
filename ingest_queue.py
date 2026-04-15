"""
ingest_queue.py
Single entry point for all ingestion sources.
Paste a YouTube URL or web link, or let newsletter_sync / granola_sync add file paths.
The poller picks up every queued item, extracts content, runs novelty check, writes to Reading List.
When you mark a Reading List entry as "Read", the promote poller moves it to KB.

Supported in Ingest Queue:
  - YouTube URLs
  - Web links
  - Newsletter / Granola / PDF file paths (added programmatically by sync scripts)

Usage:
    python ingest_queue.py poll             # poll Ingest Queue → Reading List (every 30s, loops)
    python ingest_queue.py poll --once      # process all pending items once, then exit (used by launchd)
    python ingest_queue.py poll --interval 60
    python ingest_queue.py promote          # poll Reading List (Read) → KB (every 60s)
    python ingest_queue.py promote --interval 60
"""

import os
import re
import sys
import json
import time
import hashlib
import logging
import subprocess
from pathlib import Path

import requests as _requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv, find_dotenv
from notion_client import Client

import reading_list as rl

load_dotenv(find_dotenv(usecwd=True), override=True)

notion      = Client(auth=os.getenv("NOTION_TOKEN"))
LOGS_DIR    = Path("logs");    LOGS_DIR.mkdir(exist_ok=True)
EXPORTS_DIR = Path("exports"); EXPORTS_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "ingest_queue.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ── DB IDs ────────────────────────────────────────────────────────────────────
def _queue_id() -> str:
    v = os.getenv("NOTION_INGEST_QUEUE_ID", "")
    if not v:
        raise SystemExit("NOTION_INGEST_QUEUE_ID not set — run: python notion_setup.py")
    return v

def _rl_id() -> str:
    v = os.getenv("NOTION_READING_LIST_ID", "")
    if not v:
        raise SystemExit("NOTION_READING_LIST_ID not set — run: python notion_setup.py")
    return v

def _kb_id() -> str:
    v = os.getenv("NOTION_DB_ID", "")
    if not v:
        raise SystemExit("NOTION_DB_ID not set — run: python notion_setup.py")
    return v


# ── URL helpers ───────────────────────────────────────────────────────────────
def _is_youtube(url: str) -> bool:
    return bool(re.search(r"(youtube\.com|youtu\.be)", url))

def _extract_video_id(url: str) -> str:
    m = re.search(r"(?:v=|youtu\.be/|embed/|shorts/)([A-Za-z0-9_-]{11})", url)
    if not m:
        raise ValueError(f"Cannot extract video ID from: {url}")
    return m.group(1)

def _url_hash(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()[:12]


# ── Dedup checks ──────────────────────────────────────────────────────────────
def _in_kb(source_contains: str) -> bool:
    resp = notion.databases.query(
        database_id=_kb_id(),
        filter={"property": "Source", "rich_text": {"contains": source_contains}},
    )
    return bool(resp.get("results"))

def _in_reading_list(source_contains: str) -> bool:
    resp = notion.databases.query(
        database_id=_rl_id(),
        filter={"property": "Source", "rich_text": {"contains": source_contains}},
    )
    return bool(resp.get("results"))


# ── Ingest Queue Notion helpers ───────────────────────────────────────────────
def _in_queue(identifier: str) -> bool:
    """Returns True if this URL or file path is already in the Ingest Queue (pending or failed)."""
    resp = notion.databases.query(
        database_id=_queue_id(),
        filter={"property": "URL", "title": {"contains": identifier[-60:]}},
    )
    return bool(resp.get("results"))

def add_to_queue(identifier: str, source_type: str) -> str | None:
    """
    Add a URL or file path to the Ingest Queue.
    Skips if already in KB, Reading List, or Queue.
    Returns the new page ID, or None if skipped.
    Called by newsletter_sync.py and granola_sync.py.
    """
    short_key = identifier.split("/")[-1] if "/" in identifier else identifier
    if _in_kb(short_key) or _in_reading_list(short_key) or _in_queue(short_key):
        log.info(f"  Queue skip (already exists): {short_key}")
        return None
    page = notion.pages.create(
        parent={"database_id": _queue_id()},
        properties={
            "URL": {"title": [{"type": "text", "text": {"content": identifier}}]},
            "Source_Type": {"select": {"name": source_type}},
        },
    )
    log.info(f"  Queued [{source_type}]: {short_key}")
    return page["id"]

def _query_new_queue() -> list[dict]:
    resp = notion.databases.query(
        database_id=_queue_id(),
        filter={"property": "Status", "select": {"is_empty": True}},
    )
    return resp.get("results", [])

def _set_queue_status(page_id: str, status: str):
    notion.pages.update(
        page_id=page_id,
        properties={"Status": {"select": {"name": status}}},
    )

def _write_error(page_id: str, error: str):
    notion.pages.update(
        page_id=page_id,
        properties={
            "Status": {"select": {"name": "Failed"}},
            "Error":  {"rich_text": [{"type": "text", "text": {"content": error[:800]}}]},
        },
    )
    chunks = [error[i:i+2000] for i in range(0, len(error), 2000)]
    notion.blocks.children.append(
        block_id=page_id,
        children=[{"object": "block", "type": "paragraph",
                   "paragraph": {"rich_text": [{"type": "text", "text": {"content": c}}]}}
                  for c in chunks],
    )


# ── Web link extractor ────────────────────────────────────────────────────────
def _fetch_web_content(url: str) -> dict:
    """Fetch a web page and return {title, text, url}."""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; LearningAgent/1.0)"}
    resp = _requests.get(url, headers=headers, timeout=20)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    title_tag = soup.find("title")
    page_title = title_tag.get_text(strip=True) if title_tag else url

    main = soup.find("main") or soup.find("article") or soup.find("body")
    body_text = (main or soup).get_text(separator=" ", strip=True)
    body_text = re.sub(r"\s{2,}", " ", body_text)[:15000]

    return {"title": page_title, "text": body_text, "url": url}


# ── Shared novelty check ──────────────────────────────────────────────────────
def _run_novelty(key_concepts: list, claims: list, learnings: list) -> dict:
    import dedup_engine
    body_text = "\n".join(learnings + claims)
    return dedup_engine.check_novelty_standalone(
        key_concepts=key_concepts,
        body_text=body_text,
        claims=claims,
        learnings=learnings,
    )


# ── Cache helpers ─────────────────────────────────────────────────────────────
def _save_export(cache_key: str, summary: dict, source: str, source_type: str, rl_page_id: str):
    export = {**summary, "_meta": {"cache_key": cache_key, "source": source,
                                   "source_type": source_type, "rl_page_id": rl_page_id}}
    (EXPORTS_DIR / f"{cache_key}.json").write_text(
        json.dumps(export, indent=2, ensure_ascii=False)
    )

def _load_export(cache_key: str) -> dict | None:
    path = EXPORTS_DIR / f"{cache_key}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── Process: YouTube ──────────────────────────────────────────────────────────
def _process_youtube(url: str, queue_page_id: str):
    video_id  = _extract_video_id(url)
    json_path = Path("raw_transcripts") / f"{video_id}.json"

    dedup_key = video_id  # for dedup check
    if _in_kb(dedup_key) or _in_reading_list(dedup_key):
        log.info(f"  Already in KB or Reading List — skipping")
        notion.pages.update(page_id=queue_page_id, archived=True)
        return

    # Extract transcript
    log.info(f"  Extracting transcript ({video_id})...")
    proc = subprocess.run(
        [sys.executable, "youtube_extractor.py", url],
        capture_output=True, text=True,
        cwd=str(Path(__file__).parent),
    )
    if proc.returncode != 0 or not json_path.exists():
        raise RuntimeError(f"Transcript extraction failed:\n{(proc.stderr or proc.stdout)[-800:]}")
    log.info(f"  Transcript ready: {json_path}")

    # Extract + aggregate
    import summariser
    result      = summariser.extract_only(json_path)
    summary     = result["summary"]
    source_url  = result["source_url"]
    source_type = result["source_type"]

    # Novelty check
    dedup = _run_novelty(
        key_concepts=summary.get("key_concepts", []),
        claims=summary.get("key_claims", []),
        learnings=summary.get("concrete_learnings", []),
    )
    log.info(f"  Novelty: {dedup['verdict']} ({dedup['score']:.2f}) | "
             f"{len(dedup['concept_report'].get('new',[]))} new  "
             f"{len(dedup['concept_report'].get('covered',[]))} covered")

    # Write to Reading List
    rl_page_id = rl.write_to_reading_list(summary, dedup, source_url, source_type)
    log.info(f"  Reading List entry: {rl_page_id}")

    _save_export(video_id, summary, source_url, source_type, rl_page_id)
    notion.pages.update(page_id=queue_page_id, archived=True)
    log.info("  Done — queue page archived.")


# ── Process: File (Newsletter / Granola / PDF / Notes) ───────────────────────
def _process_file(file_path: str, queue_page_id: str, source_type: str):
    path = Path(file_path)
    filename = path.name

    if _in_kb(filename) or _in_reading_list(filename):
        log.info(f"  Already in KB or Reading List — skipping")
        notion.pages.update(page_id=queue_page_id, archived=True)
        return

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    content = path.read_text(encoding="utf-8", errors="ignore")
    word_count = len(content.split())
    if word_count < 100:
        log.info(f"  Skipped (only {word_count} words — minimum 100)")
        notion.pages.update(page_id=queue_page_id, archived=True)
        return

    log.info(f"  Extracting: {filename} ({word_count} words)")
    import db_seeder
    metadata = db_seeder.extract_metadata(content)
    if not metadata.get("title"):
        metadata["title"] = filename

    dedup = _run_novelty(
        key_concepts=metadata.get("key_concepts", []),
        claims=metadata.get("key_claims", []),
        learnings=metadata.get("concrete_learnings", []),
    )
    log.info(f"  Novelty: {dedup['verdict']} ({dedup['score']:.2f}) | "
             f"{len(dedup['concept_report'].get('new',[]))} new  "
             f"{len(dedup['concept_report'].get('covered',[]))} covered")

    rl_page_id = rl.write_to_reading_list(metadata, dedup, filename, source_type)
    log.info(f"  Reading List entry: {rl_page_id}")

    cache_key = f"file_{re.sub(r'[^\\w\\-.]', '_', filename)[:60]}"
    _save_export(cache_key, metadata, filename, source_type, rl_page_id)
    notion.pages.update(page_id=queue_page_id, archived=True)
    log.info("  Done — queue page archived.")


# ── Process: Web Link ─────────────────────────────────────────────────────────
def _process_web_link(url: str, queue_page_id: str):
    url_key = f"web_{_url_hash(url)}"

    if _in_kb(url) or _in_reading_list(url):
        log.info(f"  Already in KB or Reading List — skipping")
        notion.pages.update(page_id=queue_page_id, archived=True)
        return

    # Fetch page
    log.info(f"  Fetching: {url}")
    web = _fetch_web_content(url)
    log.info(f"  Page: '{web['title']}' ({len(web['text'])} chars)")

    # Chunk + extract + aggregate (reuse db_seeder's text pipeline)
    import db_seeder
    metadata = db_seeder.extract_metadata(web["text"])
    # Override title if db_seeder produced something generic
    if not metadata.get("title"):
        metadata["title"] = web["title"]

    # Novelty check
    dedup = _run_novelty(
        key_concepts=metadata.get("key_concepts", []),
        claims=metadata.get("key_claims", []),
        learnings=metadata.get("concrete_learnings", []),
    )
    log.info(f"  Novelty: {dedup['verdict']} ({dedup['score']:.2f}) | "
             f"{len(dedup['concept_report'].get('new',[]))} new  "
             f"{len(dedup['concept_report'].get('covered',[]))} covered")

    # Write to Reading List
    rl_page_id = rl.write_to_reading_list(metadata, dedup, url, "Link")
    log.info(f"  Reading List entry: {rl_page_id}")

    _save_export(url_key, metadata, url, "Link", rl_page_id)
    notion.pages.update(page_id=queue_page_id, archived=True)
    log.info("  Done — queue page archived.")


# ── Ingest Queue: dispatch one page ──────────────────────────────────────────
FILE_SOURCE_TYPES = {"Newsletter", "Granola", "PDF", "Notes", "Case"}

def process_page(page: dict):
    queue_page_id = page["id"]
    title_items   = page["properties"].get("URL", {}).get("title", [])
    url = "".join(t.get("text", {}).get("content", "") for t in title_items).strip()
    source_type   = (page["properties"].get("Source_Type", {}).get("select") or {}).get("name", "")

    if not url:
        _write_error(queue_page_id, "No URL found — paste a URL as the page title.")
        return

    log.info(f"Processing [{source_type or 'auto'}]: {url}")
    _set_queue_status(queue_page_id, "Processing")

    try:
        if source_type in FILE_SOURCE_TYPES or (not url.startswith("http") and Path(url).suffix):
            _process_file(url, queue_page_id, source_type or "Notes")
        elif _is_youtube(url):
            _process_youtube(url, queue_page_id)
        else:
            _process_web_link(url, queue_page_id)
    except Exception as exc:
        err = str(exc)
        log.error(f"  FAILED: {err[:200]}")
        _write_error(queue_page_id, f"❌ Failed processing: {url}\n\n{err}")


# ── Promote: Reading List → KB ────────────────────────────────────────────────
def _query_read_entries() -> list[dict]:
    resp = notion.databases.query(
        database_id=_rl_id(),
        filter={"property": "Status", "select": {"equals": "Read"}},
    )
    return resp.get("results", [])


def _get_cache_key(source: str, source_type: str) -> str:
    """Derive cache filename from source + type."""
    if source_type == "YouTube" or _is_youtube(source):
        try:
            return _extract_video_id(source)
        except ValueError:
            pass
    if source_type == "Link" or source.startswith("http"):
        return f"web_{_url_hash(source)}"
    # File-based sources (Granola, Newsletter, Case, Notes, PDF)
    safe = re.sub(r"[^\w\-.]", "_", source)[:60]
    return f"file_{safe}"


def promote_entry(entry: dict):
    rl_page_id  = entry["id"]
    title_items = entry["properties"].get("Title", {}).get("title", [])
    title       = "".join(t.get("text", {}).get("content", "") for t in title_items).strip()
    source_items = entry["properties"].get("Source", {}).get("rich_text", [])
    source      = "".join(t.get("text", {}).get("content", "") for t in source_items).strip()
    source_type = (entry["properties"].get("Source_Type", {}).get("select") or {}).get("name", "Notes")

    log.info(f"Promoting: {title}")

    try:
        cache_key   = _get_cache_key(source, source_type)
        cached      = _load_export(cache_key)
        if not cached:
            raise FileNotFoundError(
                f"Cached summary not found ({cache_key}.json). "
                "Re-submit the URL/file to the Ingest Queue / db_seeder to regenerate."
            )

        meta    = cached.pop("_meta", {})
        summary = cached  # everything except _meta

        # Write clean entry to KB
        kb_page_id = rl.write_to_kb(summary, source, source_type)
        log.info(f"  KB entry created: {kb_page_id}")

        # Embed in ChromaDB + index claims (no concept novelty blocks — user already consumed)
        import dedup_engine
        dedup_engine.check_novelty(kb_page_id)  # no claims arg → no novelty blocks appended
        all_claims = summary.get("key_claims", []) + summary.get("concrete_learnings", [])
        dedup_engine._upsert_claims(kb_page_id, summary.get("title", ""), all_claims)
        log.info(f"  Embedded + {len(all_claims)} claims indexed")

        # Archive Reading List entry
        notion.pages.update(page_id=rl_page_id, archived=True)
        log.info("  Reading List entry archived — promoted to KB.")

    except Exception as exc:
        err = str(exc)
        log.error(f"  PROMOTE FAILED: {err[:200]}")
        # Revert status to To Read and append error block
        notion.pages.update(
            page_id=rl_page_id,
            properties={"Status": {"select": {"name": "To Read"}}},
        )
        notion.blocks.children.append(
            block_id=rl_page_id,
            children=[{"object": "block", "type": "paragraph",
                       "paragraph": {"rich_text": [{"type": "text", "text": {"content": f"⚠️ Promote failed: {err[:1000]}"}}]}}],
        )


# ── Polling loops ─────────────────────────────────────────────────────────────
def poll(interval: int = 30, once: bool = False):
    log.info(f"Polling Ingest Queue  |  DB: {_queue_id()}  |  mode: {'once' if once else f'every {interval}s'}")
    if not once:
        log.info("Paste a YouTube URL or web link — it moves to Reading List when done.")
        log.info("Ctrl-C to stop.")

    while True:
        try:
            pages = _query_new_queue()
            if pages:
                log.info(f"Found {len(pages)} new item(s)")
            for page in pages:
                process_page(page)
        except KeyboardInterrupt:
            log.info("Stopped.")
            break
        except Exception as e:
            log.error(f"Poll error: {e}")

        if once:
            log.info("One-shot poll complete.")
            break
        time.sleep(interval)


def poll_promote(interval: int = 60):
    log.info(f"Promote poller: checking Reading List every {interval}s  |  DB: {_rl_id()}")
    log.info("Mark a Reading List entry as 'Read' to promote it to the KB.")
    log.info("Ctrl-C to stop.")
    while True:
        try:
            entries = _query_read_entries()
            if entries:
                log.info(f"Found {len(entries)} entry/entries marked Read")
            for entry in entries:
                promote_entry(entry)
        except KeyboardInterrupt:
            log.info("Stopped.")
            break
        except Exception as e:
            log.error(f"Promote poll error: {e}")
        time.sleep(interval)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Ingest Queue + Promote poller")
    sub = parser.add_subparsers(dest="cmd")

    poll_p = sub.add_parser("poll", help="Poll Ingest Queue → Reading List")
    poll_p.add_argument("--interval", type=int, default=30)
    poll_p.add_argument("--once", action="store_true", help="Process all pending items once, then exit (used by launchd)")

    promo_p = sub.add_parser("promote", help="Poll Reading List (Read) → KB")
    promo_p.add_argument("--interval", type=int, default=60)

    args = parser.parse_args()
    if args.cmd == "poll":
        poll(interval=args.interval, once=args.once)
    elif args.cmd == "promote":
        poll_promote(interval=args.interval)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
