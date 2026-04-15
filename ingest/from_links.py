"""
ingest/from_links.py
Usage: python ingest/from_links.py [--links links.txt] [--out raw_inputs/links]

Reads a list of URLs (one per line) from a text file, fetches each page,
strips HTML to plain text, and saves as .txt files for db_seeder.py.

Skips blank lines and lines starting with #.
Skips URLs already fetched (dedup by URL stored in .seen_urls).

After running, point db_seeder.py at the output folder:
  python db_seeder.py raw_inputs/links/
"""

import argparse
import hashlib
import re
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import requests

DEFAULT_LINKS_FILE = Path(__file__).parent.parent / "links.txt"
DEFAULT_OUT = Path(__file__).parent.parent / "raw_inputs" / "links"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}
TIMEOUT = 15


# ── HTML → plain text ─────────────────────────────────────────────────────────
def strip_html(html: str) -> str:
    html = re.sub(r"<(script|style|nav|footer|header)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<(br|p|div|h[1-6]|li|tr|article|section)[^>]*>", "\n", html, flags=re.IGNORECASE)
    html = re.sub(r"<[^>]+>", "", html)
    entities = {
        "&amp;": "&", "&lt;": "<", "&gt;": ">",
        "&nbsp;": " ", "&quot;": '"', "&#39;": "'",
        "&mdash;": "—", "&ndash;": "–", "&hellip;": "…",
    }
    for ent, char in entities.items():
        html = html.replace(ent, char)
    lines = [line.strip() for line in html.splitlines()]
    lines = [l for l in lines if l]
    return "\n".join(lines)


# ── Derive a safe filename from a URL ─────────────────────────────────────────
def url_to_filename(url: str) -> str:
    parsed = urlparse(url)
    domain = parsed.netloc.replace("www.", "")
    path_slug = re.sub(r"[^\w-]", "_", parsed.path.strip("/"))[:50]
    url_hash = hashlib.md5(url.encode()).hexdigest()[:6]
    name = f"{domain}_{path_slug}_{url_hash}" if path_slug else f"{domain}_{url_hash}"
    return re.sub(r"_+", "_", name).strip("_") + ".txt"


# ── Fetch and extract ─────────────────────────────────────────────────────────
def fetch_url(url: str) -> tuple[str, str]:
    """Returns (text, status) where status is 'ok' or an error string."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "")
        if "text/html" not in content_type and "text/plain" not in content_type:
            return "", f"unsupported content type: {content_type}"
        if "text/plain" in content_type:
            return resp.text.strip(), "ok"
        return strip_html(resp.text), "ok"
    except requests.exceptions.Timeout:
        return "", "timeout"
    except requests.exceptions.HTTPError as e:
        return "", f"HTTP {e.response.status_code}"
    except Exception as e:
        return "", str(e)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Fetch URLs and stage for db_seeder")
    parser.add_argument(
        "--links", default=str(DEFAULT_LINKS_FILE),
        help=f"Path to links file, one URL per line (default: links.txt)"
    )
    parser.add_argument(
        "--out", default=str(DEFAULT_OUT),
        help="Output folder for .txt files (default: raw_inputs/links)"
    )
    args = parser.parse_args()

    links_file = Path(args.links)
    if not links_file.exists():
        print(f"Links file not found: {links_file}")
        print("Create it with one URL per line. Lines starting with # are ignored.")
        sys.exit(1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load seen URLs
    seen_file = out_dir / ".seen_urls"
    seen_urls = set(seen_file.read_text().splitlines()) if seen_file.exists() else set()

    # Parse URLs from file
    urls = []
    for line in links_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            urls.append(line)

    if not urls:
        print("No URLs found in links file.")
        sys.exit(0)

    print(f"Found {len(urls)} URLs. Processing...")

    fetched = skipped = failed = 0

    for i, url in enumerate(urls, 1):
        print(f"  [{i}/{len(urls)}] {url}", end=" ... ", flush=True)

        if url in seen_urls:
            print("SKIP (already fetched)")
            skipped += 1
            continue

        text, status = fetch_url(url)

        if status != "ok" or not text.strip():
            reason = status if status != "ok" else "empty response"
            print(f"FAIL ({reason})")
            failed += 1
            continue

        word_count = len(text.split())
        if word_count < 100:
            print(f"SKIP (only {word_count} words — too short)")
            skipped += 1
            continue

        filename = url_to_filename(url)
        today = datetime.today().strftime("%Y-%m-%d")
        content = f"Source: {url}\nDate_Fetched: {today}\n---\n\n{text}"

        (out_dir / filename).write_text(content, encoding="utf-8")
        seen_urls.add(url)
        print(f"OK ({word_count} words) → {filename}")
        fetched += 1

    seen_file.write_text("\n".join(seen_urls), encoding="utf-8")

    print(f"\nDone. {fetched} fetched. {skipped} skipped. {failed} failed.")
    if fetched > 0:
        print(f"Next step: python db_seeder.py {out_dir}")


if __name__ == "__main__":
    main()
