"""
morning_digest.py
Sends a morning email digest of your Reading List to MY_MAIN_EMAIL.

Two sections:
  🟢 New to You     — entries with verdict NEW (full summary + Notion link)
  🟡 New Angles     — entries with verdict RELATED (what's new + Notion link)

Skips COVERED entries (nothing genuinely new).

Usage:
    python morning_digest.py          # send digest
    python morning_digest.py --dry-run  # print to terminal, no email sent

Scheduled daily via launchd (com.learningagent.digest).
"""

import os
import sys
import smtplib
import argparse
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from dotenv import load_dotenv, find_dotenv
from notion_client import Client

load_dotenv(find_dotenv(usecwd=True), override=True)
notion = Client(auth=os.getenv("NOTION_TOKEN"))

READING_LIST_ID = os.getenv("NOTION_READING_LIST_ID", "").strip("'\"")
FROM_EMAIL      = os.getenv("NEWSLETTER_EMAIL", "").strip()
FROM_PASSWORD   = os.getenv("NEWSLETTER_EMAIL_PASSWORD", "").strip()
TO_EMAIL        = os.getenv("MY_MAIN_EMAIL", "").strip()


def _text(prop) -> str:
    items = prop.get("rich_text") or prop.get("title") or []
    return "".join(t.get("plain_text", "") for t in items)

def _select(prop) -> str:
    s = prop.get("select")
    return s["name"] if s else ""

def notion_url(page_id: str) -> str:
    return f"https://notion.so/{page_id.replace('-', '')}"


def fetch_reading_list() -> tuple[list, list]:
    """Returns (new_entries, related_entries) — both To Read only."""
    resp = notion.databases.query(
        database_id=READING_LIST_ID,
        filter={"property": "Status", "select": {"equals": "To Read"}},
        sorts=[{"property": "Date_Added", "direction": "descending"}],
    )
    new, related = [], []
    for p in resp.get("results", []):
        props   = p["properties"]
        verdict = _select(props.get("Verdict", {})).upper()
        entry = {
            "id":          p["id"],
            "title":       _text(props.get("Title", {})),
            "summary":     _text(props.get("Summary", {})),
            "source":      _text(props.get("Source", {})),
            "source_type": _select(props.get("Source_Type", {})),
            "new_chunks":  _text(props.get("New_Chunks", {})),
            "verdict":     verdict,
            "url":         notion_url(p["id"]),
        }
        if verdict == "NEW":
            new.append(entry)
        elif verdict in ("RELATED", "COE"):
            related.append(entry)
        # COVERED = skip

    return new, related


def build_html(new: list, related: list) -> str:
    today = datetime.now().strftime("%A, %B %-d")
    total = len(new) + len(related)

    if total == 0:
        return None

    def source_label(e):
        st = e.get("source_type", "")
        src = e.get("source", "")
        if st == "YouTube":
            return f"YouTube · <a href='{src}' style='color:#6b6b6b;'>{src[:60]}{'…' if len(src)>60 else ''}</a>"
        elif src.startswith("http"):
            return f"{st} · <a href='{src}' style='color:#6b6b6b;'>{src[:60]}{'…' if len(src)>60 else ''}</a>"
        else:
            return f"{st} · {src}"

    def entry_card_new(e):
        return f"""
        <div style="background:#f0fdf4;border:1px solid #86efac;border-radius:8px;padding:16px 20px;margin-bottom:12px;">
          <div style="font-size:13px;font-weight:700;color:#15803d;margin-bottom:6px;">🟢 {e['title']}</div>
          <div style="font-size:12px;color:#166534;margin-bottom:8px;">{source_label(e)}</div>
          <div style="font-size:13px;color:#1a1a1a;line-height:1.6;margin-bottom:10px;">{e['summary'] or 'No summary available.'}</div>
          <a href="{e['url']}" style="font-size:12px;color:#2563eb;font-weight:600;text-decoration:none;">Open in Notion →</a>
        </div>"""

    def entry_card_related(e):
        new_bits = e.get("new_chunks") or "See Notion for new angles."
        return f"""
        <div style="background:#fffbeb;border:1px solid #fcd34d;border-radius:8px;padding:16px 20px;margin-bottom:12px;">
          <div style="font-size:13px;font-weight:700;color:#92400e;margin-bottom:6px;">🟡 {e['title']}</div>
          <div style="font-size:12px;color:#78350f;margin-bottom:8px;">{source_label(e)}</div>
          <div style="font-size:12px;color:#6b6b6b;margin-bottom:4px;font-weight:600;text-transform:uppercase;letter-spacing:0.5px;">What's new</div>
          <div style="font-size:13px;color:#1a1a1a;line-height:1.6;margin-bottom:10px;">{new_bits}</div>
          <a href="{e['url']}" style="font-size:12px;color:#2563eb;font-weight:600;text-decoration:none;">Open in Notion →</a>
        </div>"""

    new_section = ""
    if new:
        new_section = f"""
        <div style="margin-bottom:28px;">
          <div style="font-size:14px;font-weight:700;color:#1a1a1a;margin-bottom:14px;padding-bottom:8px;border-bottom:2px solid #86efac;">
            🟢 New to You &nbsp;<span style="font-size:12px;font-weight:400;color:#6b6b6b;">{len(new)} item{'s' if len(new)!=1 else ''}</span>
          </div>
          {''.join(entry_card_new(e) for e in new)}
        </div>"""

    related_section = ""
    if related:
        related_section = f"""
        <div style="margin-bottom:28px;">
          <div style="font-size:14px;font-weight:700;color:#1a1a1a;margin-bottom:14px;padding-bottom:8px;border-bottom:2px solid #fcd34d;">
            🟡 New Angles on Known Topics &nbsp;<span style="font-size:12px;font-weight:400;color:#6b6b6b;">{len(related)} item{'s' if len(related)!=1 else ''}</span>
          </div>
          {''.join(entry_card_related(e) for e in related)}
        </div>"""

    return f"""
    <html><body style="margin:0;padding:0;background:#f7f7f5;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;">
    <div style="max-width:600px;margin:32px auto;background:#fff;border-radius:12px;border:1px solid #e8e8e4;overflow:hidden;">

      <!-- Header -->
      <div style="background:#1a1a1a;padding:24px 28px;">
        <div style="font-size:18px;font-weight:700;color:#fff;">Content Intelligence Agent</div>
        <div style="font-size:13px;color:#9ca3af;margin-top:4px;">Reading List Digest · {today}</div>
        <div style="margin-top:12px;display:inline-block;background:#374151;border-radius:6px;padding:4px 12px;font-size:12px;color:#d1d5db;">
          {total} item{'s' if total!=1 else ''} to review
        </div>
      </div>

      <!-- Body -->
      <div style="padding:28px;">
        {new_section}
        {related_section}

        <div style="text-align:center;padding-top:16px;border-top:1px solid #e8e8e4;">
          <a href="https://notion.so/{READING_LIST_ID.replace('-','')}"
             style="background:#2563eb;color:#fff;padding:10px 24px;border-radius:6px;font-size:13px;font-weight:600;text-decoration:none;display:inline-block;">
            Open Full Reading List
          </a>
        </div>
      </div>

      <!-- Footer -->
      <div style="padding:16px 28px;background:#f9fafb;border-top:1px solid #e8e8e4;font-size:11px;color:#9ca3af;text-align:center;">
        Sent by your Content Intelligence Agent · Mark entries as "Read" in Notion to promote them to the KB
      </div>
    </div>
    </body></html>"""


def send_email(html: str, total: int):
    today = datetime.now().strftime("%b %-d")
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"📚 Reading List Digest — {today} ({total} items)"
    msg["From"]    = FROM_EMAIL
    msg["To"]      = TO_EMAIL
    msg.attach(MIMEText(html, "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(FROM_EMAIL, FROM_PASSWORD)
        server.sendmail(FROM_EMAIL, TO_EMAIL, msg.as_string())

    print(f"Digest sent to {TO_EMAIL}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Print to terminal, don't send email")
    args = parser.parse_args()

    print("Fetching Reading List...")
    new, related = fetch_reading_list()
    total = len(new) + len(related)
    print(f"  {len(new)} new · {len(related)} related · {total} total")

    if total == 0:
        print("Nothing to send — Reading List is empty or all COVERED.")
        return

    html = build_html(new, related)

    if args.dry_run:
        print("\n--- DRY RUN ---")
        print(f"Would send to: {TO_EMAIL}")
        print(f"Subject: Reading List Digest — {datetime.now().strftime('%b %-d')} ({total} items)")
        print(f"\n🟢 New ({len(new)}):")
        for e in new:
            print(f"  · {e['title']}")
        print(f"\n🟡 Related ({len(related)}):")
        for e in related:
            print(f"  · {e['title']}")
    else:
        send_email(html, total)


if __name__ == "__main__":
    main()
