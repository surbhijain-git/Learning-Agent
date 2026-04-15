"""
granola_sync.py
Scans the Talks/ folder for new Granola meeting notes and adds them to the Ingest Queue.
The Ingest Queue poller (ingest_queue.py poll --once) picks them up every 2 days.

Designed to be run by launchd every week.
Can also be run manually: python granola_sync.py

Logs to logs/granola_sync.log
"""

import sys
import logging
from datetime import datetime
from pathlib import Path

LOGS_DIR   = Path("logs"); LOGS_DIR.mkdir(exist_ok=True)
TALKS_DIR  = Path("Talks")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "granola_sync.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


def main():
    log.info("=" * 60)
    log.info(f"Granola sync started at {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    log.info("=" * 60)

    if not TALKS_DIR.exists():
        log.info(f"Talks/ folder not found — nothing to queue.")
        sys.exit(0)

    files = sorted(
        f for f in TALKS_DIR.iterdir()
        if f.suffix.lower() in (".txt", ".md") and f.is_file()
    )

    if not files:
        log.info("No .txt/.md files found in Talks/ — nothing to queue.")
        sys.exit(0)

    import ingest_queue
    queued = 0
    skipped = 0
    for f in files:
        result = ingest_queue.add_to_queue(str(f.resolve()), "Granola")
        if result:
            queued += 1
        else:
            skipped += 1

    log.info(f"Granola sync complete. {queued} queued, {skipped} skipped (already exists).")


if __name__ == "__main__":
    main()
