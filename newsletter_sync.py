"""
newsletter_sync.py
Full newsletter pipeline in one shot:
  1. Fetch new emails from Gmail (last 2 days, matches daily schedule)
  2. Add each new newsletter file to the Ingest Queue → poller picks it up every 2 days

Designed to be run by launchd every day.
Can also be run manually: python newsletter_sync.py

Logs to logs/newsletter_sync.log
"""

import sys
import subprocess
import logging
from datetime import datetime
from pathlib import Path

LOGS_DIR     = Path("logs");               LOGS_DIR.mkdir(exist_ok=True)
NEWSLETTERS  = Path("raw_inputs/newsletters")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "newsletter_sync.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

PYTHON = sys.executable


def run(cmd: list[str], label: str) -> bool:
    """Run a subprocess, log output, return True on success."""
    log.info(f"▶ {label}")
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).parent),
    )
    if result.stdout.strip():
        for line in result.stdout.strip().splitlines():
            log.info(f"  {line}")
    if result.stderr.strip():
        for line in result.stderr.strip().splitlines():
            log.warning(f"  {line}")
    if result.returncode != 0:
        log.error(f"  ✗ {label} exited with code {result.returncode}")
        return False
    log.info(f"  ✓ {label} done")
    return True


def main():
    log.info("=" * 60)
    log.info(f"Newsletter sync started at {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    log.info("=" * 60)

    # Step 1: Fetch new emails from Gmail (2 days covers the daily schedule gap)
    ok = run(
        [PYTHON, "ingest/from_gmail.py", "--days", "2", "--out", str(NEWSLETTERS)],
        "Gmail fetch (last 2 days)",
    )
    if not ok:
        log.error("Gmail fetch failed — check NEWSLETTER_EMAIL / NEWSLETTER_EMAIL_PASSWORD in .env")
        log.error("Also ensure IMAP is enabled in Gmail settings and you're using an App Password.")
        sys.exit(1)

    # Step 2: Add each new newsletter file to the Ingest Queue
    txt_files = sorted(NEWSLETTERS.glob("*.txt")) if NEWSLETTERS.exists() else []
    if not txt_files:
        log.info("No newsletter files found — nothing to queue.")
        sys.exit(0)

    import ingest_queue
    queued = 0
    for f in txt_files:
        result = ingest_queue.add_to_queue(str(f.resolve()), "Newsletter")
        if result:
            queued += 1

    log.info(f"Newsletter sync complete. {queued}/{len(txt_files)} files added to Ingest Queue.")


if __name__ == "__main__":
    main()
