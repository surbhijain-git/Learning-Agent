"""
ingest/newsletters_weekly.py
Usage: python ingest/newsletters_weekly.py [--months 8] [--weeks N] [--dry-run] [--save-only]

Fetches newsletters from the Gmail 'newsletters' label, groups by ISO week,
and writes one Notion KB entry per week synthesising all newsletters that week.

Senders tracked: AI Secret, Bloomberg Technology, Azeem Azhar (Exponential View)

--dry-run     Prints week groups and email counts without calling Claude or writing to Notion.
--months N    How many months back to fetch (default: 8)
--weeks N     Only process the N most recent weeks (e.g. --weeks 20)
--save-only   Save each week as a .txt file in raw_inputs/newsletters/ instead of writing to Notion.
              Run summariser separately on those files when ready.
"""

import argparse
import email
import email.header
import hashlib
import imaplib
import json
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv
import anthropic
from notion_client import Client

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
IMAP_HOST = "imap.gmail.com"
IMAP_PORT = 993
EMAIL_USER = os.getenv("NEWSLETTER_EMAIL")
EMAIL_PASS = os.getenv("NEWSLETTER_EMAIL_PASSWORD")

MAILBOX = "INBOX/Newsletter"

TARGET_SENDERS = ["ai secret", "bloomberg", "azeem", "exponential view"]

SONNET = "claude-sonnet-4-6"

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)
STATE_FILE = LOGS_DIR / "newsletter_weeks_ingested.json"

ai = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
notion = Client(auth=os.getenv("NOTION_TOKEN"))
NOTION_DB_ID = os.getenv("NOTION_DB_ID")


# ── HTML → plain text ─────────────────────────────────────────────────────────
def strip_html(html: str) -> str:
    html = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<(br|p|div|h[1-6]|li|tr)[^>]*>", "\n", html, flags=re.IGNORECASE)
    html = re.sub(r"<[^>]+>", "", html)
    entities = {"&amp;": "&", "&lt;": "<", "&gt;": ">", "&nbsp;": " ",
                "&quot;": '"', "&#39;": "'", "&mdash;": "—", "&ndash;": "–", "&hellip;": "…"}
    for ent, char in entities.items():
        html = html.replace(ent, char)
    lines = [l.strip() for l in html.splitlines() if l.strip()]
    return "\n".join(lines)


def decode_header(value: str) -> str:
    parts = email.header.decode_header(value)
    decoded = []
    for part, charset in parts:
        if isinstance(part, bytes):
            decoded.append(part.decode(charset or "utf-8", errors="replace"))
        else:
            decoded.append(part)
    return "".join(decoded)


def get_email_text(msg) -> str:
    plain_parts, html_parts = [], []
    if msg.is_multipart():
        for part in msg.walk():
            ct = part.get_content_type()
            if "attachment" in str(part.get("Content-Disposition", "")):
                continue
            charset = part.get_content_charset() or "utf-8"
            try:
                payload = part.get_payload(decode=True)
                if not payload:
                    continue
                text = payload.decode(charset, errors="replace")
            except Exception:
                continue
            if ct == "text/plain":
                plain_parts.append(text)
            elif ct == "text/html":
                html_parts.append(text)
    else:
        charset = msg.get_content_charset() or "utf-8"
        try:
            payload = msg.get_payload(decode=True)
            text = payload.decode(charset, errors="replace") if payload else ""
        except Exception:
            text = ""
        if msg.get_content_type() == "text/html":
            html_parts.append(text)
        else:
            plain_parts.append(text)

    if plain_parts:
        return "\n\n".join(plain_parts)
    elif html_parts:
        return strip_html("\n\n".join(html_parts))
    return ""


def is_target_sender(from_addr: str) -> bool:
    from_lower = from_addr.lower()
    return any(s in from_lower for s in TARGET_SENDERS)


def iso_week_key(dt: datetime) -> str:
    """Returns 'YYYY-WNN' e.g. '2026-W03'"""
    iso = dt.isocalendar()
    return f"{iso[0]}-W{iso[1]:02d}"


def week_start_date(week_key: str) -> str:
    """Returns Monday date string for a given 'YYYY-WNN' key."""
    year, week = week_key.split("-W")
    monday = datetime.strptime(f"{year}-W{week}-1", "%Y-W%W-%w")
    return monday.strftime("%Y-%m-%d")


# ── Load / save ingested weeks ────────────────────────────────────────────────
def load_ingested_weeks() -> set:
    if STATE_FILE.exists():
        return set(json.loads(STATE_FILE.read_text()))
    return set()


def save_ingested_weeks(weeks: set):
    STATE_FILE.write_text(json.dumps(sorted(weeks), indent=2))


# ── Claude: chunk-level extraction (Haiku) ────────────────────────────────────
CHUNK_SYSTEM = (
    "You are a knowledge extraction engine for a strategy consultant with an engineering background. "
    "From this newsletter content, extract two types of knowledge:\n\n"
    "INSIGHT: a genuinely non-obvious strategic idea, implication, or framework — something that changes how you think.\n"
    "LEARN: a specific product launch, company move, stat, named framework, technique, or tool — something concrete and referenceable.\n\n"
    "Output each item on its own line prefixed exactly with its type:\n"
    "INSIGHT: <single sentence>\n"
    "LEARN: <single sentence>\n\n"
    "Skip obvious statements and filler. Be precise."
)

HAIKU = "claude-haiku-4-5-20251001"


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


# ── Claude: aggregation (Sonnet) ──────────────────────────────────────────────
WEEKLY_AGG_SYSTEM = (
    "You are a knowledge synthesis engine for a strategy consultant with an engineering background. "
    "You will receive INSIGHTS and LEARNINGS extracted from a week of newsletters "
    "(AI Secret, Bloomberg Technology, Azeem Azhar). "
    "Synthesise them into one structured weekly knowledge entry. "
    "Focus on cross-newsletter connections and non-obvious strategic implications. "
    "Return ONLY valid JSON — no markdown fences:\n"
    "{\n"
    '  "title": "short topic label (3-6 words) — name the week\'s dominant subject, not the insight. E.g. \'AI Regulation Week\', \'LLM Benchmarking Trends\'",\n'
    '  "core_argument": "2-3 sentences — the central narrative connecting this week\'s developments",\n'
    '  "key_claims": ["4-6 specific, non-obvious claims worth remembering from this week"],\n'
    '  "concrete_learnings": ["all distinct product launches, company moves, stats, named frameworks, or techniques — be specific"],\n'
    '  "so_what": "1-2 sentences — what a strategy consultant would do differently knowing this week\'s news",\n'
    '  "key_concepts": ["3-5 short topic tags"]\n'
    "}"
)


def synthesise_week(week_key: str, emails: list[dict]) -> dict:
    """Full pipeline: chunk all emails → Haiku per chunk → Sonnet aggregate."""
    # Combine all emails into one text block
    sections = []
    for e in emails:
        sections.append(
            f"--- {e['sender']} | {e['date']} ---\n"
            f"Subject: {e['subject']}\n\n"
            f"{e['body']}"
        )
    full_text = "\n\n".join(sections)

    # Chunk and extract
    chunks = chunk_text(full_text, max_words=500)
    print(f"{len(chunks)} chunks", end=" ", flush=True)

    all_insights, all_learnings = [], []
    for chunk in chunks:
        result = extract_claims_from_chunk(chunk)
        all_insights.extend(result["insights"])
        all_learnings.extend(result["learnings"])

    # Deduplicate
    all_insights = list(dict.fromkeys(all_insights))
    all_learnings = list(dict.fromkeys(all_learnings))
    print(f"→ {len(all_insights)} insights, {len(all_learnings)} learnings", end=" ", flush=True)

    # Aggregate with Sonnet
    insights_block = "\n".join(f"- {c}" for c in all_insights)
    learnings_block = "\n".join(f"- {c}" for c in all_learnings)
    user_msg = f"STRATEGIC INSIGHTS:\n{insights_block}\n\nCONCRETE LEARNINGS:\n{learnings_block}"

    # If learnings list is very long, ask Claude to consolidate in the output
    if len(all_learnings) > 30:
        user_msg += (
            "\n\nNote: there are many learnings above. In concrete_learnings, "
            "consolidate related items into single precise statements so the final list has 10-15 entries."
        )

    response = ai.messages.create(
        model=SONNET,
        max_tokens=8192,
        system=WEEKLY_AGG_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )
    raw = response.content[0].text.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


# ── Notion write ──────────────────────────────────────────────────────────────
def write_to_notion(summary: dict, week_key: str, email_count: int) -> str:
    monday = week_start_date(week_key)
    summary_text = summary["core_argument"] + "\n\n" + summary["so_what"]

    body_blocks = [
        {
            "object": "block", "type": "heading_2",
            "heading_2": {"rich_text": [{"type": "text", "text": {"content": "What I Learned"}}]},
        },
        *[
            {"object": "block", "type": "bulleted_list_item",
             "bulleted_list_item": {"rich_text": [{"type": "text", "text": {"content": item}}]}}
            for item in summary.get("concrete_learnings", [])
        ],
        {
            "object": "block", "type": "heading_2",
            "heading_2": {"rich_text": [{"type": "text", "text": {"content": "Key Claims"}}]},
        },
        *[
            {"object": "block", "type": "bulleted_list_item",
             "bulleted_list_item": {"rich_text": [{"type": "text", "text": {"content": item}}]}}
            for item in summary.get("key_claims", [])
        ],
    ]

    page = notion.pages.create(
        parent={"database_id": NOTION_DB_ID},
        properties={
            "Title": {"title": [{"type": "text", "text": {"content": summary["title"]}}]},
            "Source": {"rich_text": [{"type": "text", "text": {"content": f"Gmail | {week_key} | {email_count} emails"}}]},
            "Source_Type": {"select": {"name": "Newsletter"}},
            "Date_Added": {"date": {"start": monday}},
            "Summary": {"rich_text": [{"type": "text", "text": {"content": summary_text}}]},
            "Key_Concepts": {"multi_select": [{"name": t} for t in summary.get("key_concepts", [])]},
            "Understanding_Level": {"select": {"name": "New"}},
            "Is_New_Info": {"checkbox": True},
            "Similarity_Score": {"number": 0},
        },
        children=body_blocks,
    )
    return page["id"]


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--months", type=int, default=8)
    parser.add_argument("--weeks", type=int, default=None, help="Only process N most recent weeks")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--save-only", action="store_true", help="Save .txt files instead of writing to Notion")
    args = parser.parse_args()

    if not EMAIL_USER or not EMAIL_PASS:
        print("NEWSLETTER_EMAIL or NEWSLETTER_EMAIL_PASSWORD not set in .env")
        sys.exit(1)

    since_date = datetime.now() - timedelta(days=args.months * 30)
    date_str = since_date.strftime("%d-%b-%Y")

    print(f"Connecting to Gmail as {EMAIL_USER}...")
    try:
        mail = imaplib.IMAP4_SSL(IMAP_HOST, IMAP_PORT)
        mail.login(EMAIL_USER, EMAIL_PASS)
    except Exception as e:
        print(f"IMAP login failed: {e}")
        sys.exit(1)

    # Select the newsletters label
    status, _ = mail.select(f'"{MAILBOX}"' if " " in MAILBOX else MAILBOX)
    if status != "OK":
        print(f"Could not select mailbox '{MAILBOX}'. Check the label name in Gmail.")
        mail.logout()
        sys.exit(1)

    status, msg_ids_raw = mail.search(None, f'SINCE "{date_str}"')
    if status != "OK":
        print("Search failed.")
        mail.logout()
        sys.exit(1)

    msg_ids = msg_ids_raw[0].split()
    print(f"Found {len(msg_ids)} emails since {date_str}. Filtering by sender...")

    # ── Fetch and group by week ───────────────────────────────────────────────
    weeks: dict[str, list[dict]] = {}
    matched = 0
    skipped = 0

    for msg_id in msg_ids:
        try:
            status, msg_data = mail.fetch(msg_id, "(RFC822)")
            if status != "OK" or not msg_data or not msg_data[0]:
                continue
            msg = email.message_from_bytes(msg_data[0][1])

            from_addr = decode_header(msg.get("From", ""))
            if not is_target_sender(from_addr):
                skipped += 1
                continue

            subject = decode_header(msg.get("Subject", "No Subject"))
            date_hdr = msg.get("Date", "")
            try:
                from email.utils import parsedate_to_datetime
                sent_dt = parsedate_to_datetime(date_hdr)
            except Exception:
                sent_dt = datetime.now()

            body = get_email_text(msg)
            if len(body.split()) < 50:
                continue

            wk = iso_week_key(sent_dt)
            weeks.setdefault(wk, []).append({
                "subject": subject,
                "sender": from_addr,
                "date": sent_dt.strftime("%Y-%m-%d"),
                "body": body,
            })
            matched += 1

        except Exception as e:
            print(f"  Error on message {msg_id}: {e}")

    mail.logout()

    print(f"Matched {matched} emails from target senders ({skipped} others skipped).")
    print(f"Grouped into {len(weeks)} weeks.\n")

    if not weeks:
        print("No matching emails found. Check sender names in TARGET_SENDERS.")
        sys.exit(0)

    # ── Apply --weeks limit (most recent N weeks) ─────────────────────────────
    sorted_week_keys = sorted(weeks.keys())
    if args.weeks:
        sorted_week_keys = sorted_week_keys[-args.weeks:]
        print(f"Limiting to {len(sorted_week_keys)} most recent weeks.\n")

    # ── Print week overview ───────────────────────────────────────────────────
    for wk in sorted_week_keys:
        print(f"  {wk} ({week_start_date(wk)}): {len(weeks[wk])} emails")

    if args.dry_run:
        print("\n[dry-run] Stopping here. Remove --dry-run to ingest.")
        sys.exit(0)

    # ── Save-only mode: write one .txt per week ───────────────────────────────
    if args.save_only:
        out_dir = Path("raw_inputs/newsletters")
        out_dir.mkdir(parents=True, exist_ok=True)
        saved = 0
        print()
        for wk in sorted_week_keys:
            filename = f"{wk}_newsletters.txt"
            out_path = out_dir / filename
            if out_path.exists():
                print(f"SKIP {wk} (file already exists)")
                continue
            emails = weeks[wk]
            sections = []
            for e in emails:
                sections.append(
                    f"--- {e['sender']} | {e['date']} ---\n"
                    f"Subject: {e['subject']}\n\n"
                    f"{e['body']}"
                )
            content = (
                f"Source: Newsletters {wk} ({len(emails)} emails)\n"
                f"Week: {week_start_date(wk)}\n"
                f"---\n\n"
                + "\n\n".join(sections)
            )
            out_path.write_text(content, encoding="utf-8")
            print(f"SAVED {wk} → {filename} ({len(emails)} emails)")
            saved += 1
        print(f"\nDone. {saved} weekly files saved to raw_inputs/newsletters/")
        sys.exit(0)

    # ── Ingest week by week → Notion ──────────────────────────────────────────
    ingested_weeks = load_ingested_weeks()
    added = skipped_weeks = 0

    print()
    for wk in sorted_week_keys:
        if wk in ingested_weeks:
            print(f"SKIP {wk} (already in Notion)")
            skipped_weeks += 1
            continue

        emails = weeks[wk]
        print(f"Processing {wk} ({len(emails)} emails)...", end=" ", flush=True)

        try:
            summary = synthesise_week(wk, emails)
        except Exception as e:
            print(f"FAIL (Claude: {e})")
            continue

        try:
            write_to_notion(summary, wk, len(emails))
            ingested_weeks.add(wk)
            save_ingested_weeks(ingested_weeks)
            print(f"DONE → \"{summary['title']}\"")
            added += 1
        except Exception as e:
            print(f"FAIL (Notion: {e})")

    print(f"\nDone. {added} weeks added. {skipped_weeks} skipped (already ingested).")


if __name__ == "__main__":
    main()
