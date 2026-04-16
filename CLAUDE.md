# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

The **Content Intelligence Agent** — a learning system that ingests YouTube videos, newsletters, PDFs, and Granola meeting notes, extracts structured knowledge via Claude, and writes entries to a Notion knowledge base (KB) and updates her reading list. Built as the primary agent for Surbhi's HBS NYW final project. Currently she is planning to use this to build KB for AI, architecture can be used for learning about any industry/ topic while working on any job/ consulting.

## Setup

```bash
pip install -r requirements.txt
brew install ffmpeg          # required for Whisper fallback transcription
python setup_check.py        # verify .env keys are populated
python notion_setup.py       # create Notion DB schemas, auto-writes NOTION_DB_ID to .env
```

Required `.env` keys: `ANTHROPIC_API_KEY`, `NOTION_TOKEN`, `NEWSLETTER_EMAIL`, `NEWSLETTER_EMAIL_PASSWORD`  
Optional: `NOTION_DB_ID`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, `READWISE_TOKEN`

## Core Pipeline

The pipeline is: **ingest → transcript JSON → summariser → Notion KB**

### Step 1: Ingest (produce raw text or transcript JSON)


| Source                | Command                                                                                        |
| --------------------- | ---------------------------------------------------------------------------------------------- |
| YouTube               | `python youtube_extractor.py <url>` → saves `raw_transcripts/<video_id>.json`                  |
| Gmail newsletters     | `python ingest/from_gmail.py [--senders "a@b.com"] [--days 30] [--out raw_inputs/newsletters]` |
| PDFs                  | `python ingest/from_pdf.py <pdf_folder> [--out raw_inputs/pdfs]`                               |
| Granola meeting notes | Granola notes auto-save to `Talks/` via a scheduled CoWork task. No ingest script needed — just run `db_seeder.py Talks/` to push to Notion (see below). |


### Step 2: Summarise and write to Notion

```bash
# From a YouTube transcript JSON:
python summariser.py raw_transcripts/<video_id>.json

# Skip re-extraction, use cached export:
python summariser.py raw_transcripts/<video_id>.json --notion-only

# From newsletters/PDFs/Granola output folders:
python db_seeder.py raw_inputs/newsletters/
python db_seeder.py raw_inputs/pdfs/
python db_seeder.py Talks/          # Granola notes — run weekly
```

## Architecture

### Two extraction paths

`**youtube_extractor.py` + `summariser.py**` — YouTube-specific, richer output:

- Captions first; falls back to yt-dlp + Whisper (base model) if unavailable
- Chunks transcript into ~500-word segments
- **Chunk-level**: Haiku extracts `INSIGHT:` / `LEARN:` lines per chunk
- **Aggregation**: model chosen by chunk count (≤3 → Haiku, ≤8 → Sonnet, >8 → Opus)
- Writes structured page to Notion with "What I Learned" and "Key Claims" body blocks
- Runs a self-test (field validation + Notion round-trip check)

`**db_seeder.py`** — generic for text files (newsletters, PDFs, Granola):

- Deduplicates by Notion `Source` field (filename) before calling Claude
- Rate-limited at 13s between API calls (free tier: 5 req/min)
- Haiku-only extraction; simpler JSON schema (title, summary, key_concepts)
- Logs failures to `logs/failed_imports.txt`

### Notion KB schema

Two databases managed by `notion_setup.py`:

- **Learning Agent KB** (`NOTION_DB_ID`): main knowledge store. Key fields: `Title`, `Source`, `Source_Type`, `Summary`, `Key_Concepts` (multi-select), `Understanding_Level` (New / Understood / Needs Revisit / Explore Further), `Is_New_Info`, `Similarity_Score`, `Embedding_ID`
- **Reading List** (`NOTION_READING_LIST_ID`): staging queue. Fields: `Status` (To Read / Read), `Verdict` (NEW / RELATED / COe), `New_Chunks`

### Model constants (in `summariser.py`)

- `HAIKU  = "claude-haiku-4-5-20251001"`
- `SONNET = "claude-sonnet-4-6"`
- `OPUS   = "claude-opus-4-6"`

### ChromaDB

`chroma_db/` contains a local vector store (used for semantic deduplication / similarity scoring). The `Similarity_Score` and `Embedding_ID` Notion fields hook into this.

## Notion Access

**Always use the Notion API via Python (`notion_client` + `NOTION_TOKEN` from `.env`), never the Notion MCP tools.**
The MCP Notion integration is connected to a different Notion account and cannot access this project's databases.

For any Notion read/write/schema changes, use:
```python
from notion_client import Client
notion = Client(auth=os.getenv("NOTION_TOKEN"))
```

## Key Behaviors to Know

- `youtube_extractor.py` saves to `raw_transcripts/<video_id>.json` and always overwrites if re-run
- `summariser.py` saves a fallback export to `exports/<video_id>.txt` if Notion write fails; use `--notion-only` to retry from that cache
- `db_seeder.py` skips files under 100 words and files already in Notion (by Source=filename match)
- `ingest/from_gmail.py` deduplicates by Message-ID header, persisting seen IDs to `.seen_ids` in the output folder
- Granola notes are auto-saved to `Talks/` via a scheduled CoWork task — `ingest/from_granola.py` does not exist and is not needed
- HBS case PDFs are often scanned images — `from_pdf.py` will warn and skip these (OCR not implemented)
- `newsletter_sync.py` queues newsletters as **file pointers** (not content) — txt files stay in `raw_inputs/newsletters/` until `ingest_queue.py poll` reads and deletes them. Do NOT store newsletter content inline in Notion (hits 2000-char rich_text limit). The GitHub Actions workflow runs both in the same job so files persist between steps.
- `notion_client` v2.2.1 dropped `databases.query` — all Notion DB queries use direct `httpx.post` to `https://api.notion.com/v1/databases/{id}/query` with `Notion-Version: 2022-06-28`. Do not use `notion.databases.query` anywhere.

