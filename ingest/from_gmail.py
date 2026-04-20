"""
ingest/from_gmail.py
Usage: python ingest/from_gmail.py [--senders "a@b.com,c@d.com"] [--days 30] [--out raw_inputs/newsletters]

Connects to Gmail via IMAP using NEWSLETTER_EMAIL + NEWSLETTER_EMAIL_PASSWORD from .env.
Fetches newsletter emails from specified senders (or all emails if --senders not given).
Strips HTML to plain text. Saves each email as a .txt file in the output folder.
Skips emails already exported (dedup by message-id).

After running, point db_seeder.py at the output folder:
  python db_seeder.py raw_inputs/newsletters/
"""

import imaplib
import email
import email.header
import os
import re
import hashlib
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from email.utils import parsedate_to_datetime

from dotenv import load_dotenv

load_dotenv()

IMAP_HOST = "imap.gmail.com"
IMAP_PORT = 993
EMAIL_USER = os.getenv("NEWSLETTER_EMAIL")
EMAIL_PASS = os.getenv("NEWSLETTER_EMAIL_PASSWORD")


# ── HTML → plain text ─────────────────────────────────────────────────────────
def strip_html(html: str) -> str:
    """Best-effort HTML → plain text without external dependencies."""
    # Remove script/style blocks entirely
    html = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
    # Replace block-level tags with newlines
    html = re.sub(r"<(br|p|div|h[1-6]|li|tr)[^>]*>", "\n", html, flags=re.IGNORECASE)
    # Remove all remaining tags
    html = re.sub(r"<[^>]+>", "", html)
    # Decode common HTML entities
    entities = {
        "&amp;": "&", "&lt;": "<", "&gt;": ">",
        "&nbsp;": " ", "&quot;": '"', "&#39;": "'",
        "&mdash;": "—", "&ndash;": "–", "&hellip;": "…",
    }
    for ent, char in entities.items():
        html = html.replace(ent, char)
    # Collapse whitespace
    lines = [line.strip() for line in html.splitlines()]
    lines = [l for l in lines if l]
    return "\n".join(lines)


# ── Decode header value ───────────────────────────────────────────────────────
def decode_header(value: str) -> str:
    """Decode encoded email headers (e.g., =?utf-8?...?=)."""
    parts = email.header.decode_header(value)
    decoded = []
    for part, charset in parts:
        if isinstance(part, bytes):
            decoded.append(part.decode(charset or "utf-8", errors="replace"))
        else:
            decoded.append(part)
    return "".join(decoded)


# ── Slugify for filename ──────────────────────────────────────────────────────
def slugify(text: str, max_len: int = 50) -> str:
    text = re.sub(r"[^\w\s-]", "", text.lower())
    text = re.sub(r"[\s_-]+", "_", text).strip("_")
    return text[:max_len]


# ── Extract plain text from email message ─────────────────────────────────────
def get_email_text(msg) -> str:
    """Extract plain text from a multipart email, preferring text/plain."""
    plain_parts = []
    html_parts = []

    if msg.is_multipart():
        for part in msg.walk():
            ct = part.get_content_type()
            cd = str(part.get("Content-Disposition", ""))
            if "attachment" in cd:
                continue
            charset = part.get_content_charset() or "utf-8"
            try:
                payload = part.get_payload(decode=True)
                if payload is None:
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


# ── Build IMAP search query ───────────────────────────────────────────────────
def build_search_criteria(senders: list[str], since_date: datetime) -> str:
    date_str = since_date.strftime("%d-%b-%Y")
    date_criterion = f'SINCE "{date_str}"'

    if not senders:
        return date_criterion

    if len(senders) == 1:
        return f'(FROM "{senders[0]}" {date_criterion})'

    # IMAP OR is binary — nest for multiple senders
    def nest_or(items):
        if len(items) == 1:
            return f'FROM "{items[0]}"'
        return f'(OR FROM "{items[0]}" {nest_or(items[1:])})'

    return f'({nest_or(senders)} {date_criterion})'


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Fetch newsletters from Gmail via IMAP")
    parser.add_argument(
        "--senders", default="",
        help="Comma-separated list of sender email addresses to filter by. "
             "Leave blank to fetch all emails."
    )
    parser.add_argument(
        "--days", type=int, default=90,
        help="Fetch emails from the last N days (default: 90)"
    )
    parser.add_argument(
        "--out", default="raw_inputs/newsletters",
        help="Output folder for .txt files (default: raw_inputs/newsletters)"
    )
    parser.add_argument(
        "--mailbox", default="INBOX",
        help="IMAP mailbox to search (default: INBOX). Use '[Gmail]/All Mail' for all."
    )
    args = parser.parse_args()

    if not EMAIL_USER or not EMAIL_PASS:
        print("Error: NEWSLETTER_EMAIL or NEWSLETTER_EMAIL_PASSWORD not set in .env")
        return

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Track already-exported message IDs
    seen_file = out_dir / ".seen_ids"
    seen_ids = set()
    if seen_file.exists():
        seen_ids = set(seen_file.read_text().splitlines())

    senders = [s.strip() for s in args.senders.split(",") if s.strip()]
    since_date = datetime.now() - timedelta(days=args.days)

    print(f"Connecting to {IMAP_HOST} as {EMAIL_USER}...")
    try:
        mail = imaplib.IMAP4_SSL(IMAP_HOST, IMAP_PORT)
        mail.login(EMAIL_USER, EMAIL_PASS)
    except Exception as e:
        print(f"IMAP login failed: {e}")
        print("Check NEWSLETTER_EMAIL and NEWSLETTER_EMAIL_PASSWORD in .env")
        print("Make sure IMAP is enabled in Gmail Settings → See all settings → Forwarding and POP/IMAP")
        return

    mail.select(f'"{args.mailbox}"' if " " in args.mailbox else args.mailbox)

    search_criteria = build_search_criteria(senders, since_date)
    print(f"Searching: {search_criteria}")

    status, message_ids_raw = mail.search(None, search_criteria)
    if status != "OK":
        print("Search failed.")
        mail.logout()
        return

    message_ids = message_ids_raw[0].split()
    print(f"Found {len(message_ids)} emails. Processing...")

    exported = 0
    skipped = 0
    failed = 0

    for msg_id in message_ids:
        try:
            status, msg_data = mail.fetch(msg_id, "(RFC822)")
            if status != "OK" or not msg_data or not msg_data[0]:
                failed += 1
                continue

            raw_email = msg_data[0][1]
            msg = email.message_from_bytes(raw_email)

            # Dedup by Message-ID header
            message_id_header = msg.get("Message-ID", "").strip()
            if not message_id_header:
                # Fallback: hash the first 500 bytes
                message_id_header = hashlib.md5(raw_email[:500]).hexdigest()

            if message_id_header in seen_ids:
                skipped += 1
                continue

            subject = decode_header(msg.get("Subject", "No Subject"))
            from_addr = decode_header(msg.get("From", "Unknown"))
            date_str = msg.get("Date", "")

            try:
                sent_date = parsedate_to_datetime(date_str).strftime("%Y-%m-%d")
            except Exception:
                sent_date = datetime.now().strftime("%Y-%m-%d")

            body = get_email_text(msg)
            if not body.strip():
                skipped += 1
                continue

            # Filename based on Message-ID — stable across re-runs, no collision counter needed
            msg_id_safe = re.sub(r"[^\w\-]", "_", message_id_header.strip("<>"))[:80]
            filename = f"{msg_id_safe}.txt"

            # Add header metadata to file
            full_content = (
                f"Source: {from_addr}\n"
                f"Subject: {subject}\n"
                f"Date: {date_str}\n"
                f"---\n\n"
                f"{body}"
            )

            out_path = out_dir / filename

            out_path.write_text(full_content, encoding="utf-8")
            seen_ids.add(message_id_header)

            word_count = len(body.split())
            print(f"  Saved: {out_path.name} ({word_count} words)")
            exported += 1

        except Exception as e:
            print(f"  Error processing message {msg_id}: {e}")
            failed += 1

    mail.logout()

    # Persist seen IDs
    seen_file.write_text("\n".join(seen_ids), encoding="utf-8")

    print(f"\nDone. {exported} exported. {skipped} skipped (already seen or empty). {failed} failed.")
    if exported > 0:
        print(f"Files saved to: {out_dir}")
        print(f"Next step: python db_seeder.py {out_dir}")


if __name__ == "__main__":
    main()
