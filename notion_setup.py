from dotenv import load_dotenv, set_key
import os
from datetime import date
from notion_client import Client
from pathlib import Path

load_dotenv()

token = os.getenv("NOTION_TOKEN")
if not token:
    raise SystemExit("NOTION_TOKEN not found in .env")

notion = Client(auth=token)

PARENT_PAGE_ID = os.getenv("NOTION_PARENT_PAGE_ID", "")
if not PARENT_PAGE_ID:
    raise SystemExit("NOTION_PARENT_PAGE_ID not set in .env")
ENV_PATH = Path(__file__).parent / ".env"

# ════════════════════════════════════════════════════════════════════════════
# DB 1 — Learning Agent KB (skip if already exists)
# ════════════════════════════════════════════════════════════════════════════
existing_db_id = os.getenv("NOTION_DB_ID")
if existing_db_id:
    try:
        db = notion.databases.retrieve(existing_db_id)
        print(f"[DB 1] Already exists: {db['title'][0]['text']['content']}")
        print(f"       NOTION_DB_ID = {db['id']}")
    except Exception:
        print("[DB 1] NOTION_DB_ID in .env is invalid — recreating Learning Agent KB...")
        db = notion.databases.create(
            parent={"type": "page_id", "page_id": PARENT_PAGE_ID},
            title=[{"type": "text", "text": {"content": "Learning Agent KB"}}],
            properties={
                "Title": {"title": {}},
                "Source": {"rich_text": {}},
                "Source_Type": {
                    "select": {
                        "options": [
                            {"name": "YouTube",    "color": "red"},
                            {"name": "PDF",        "color": "blue"},
                            {"name": "Newsletter", "color": "green"},
                            {"name": "Notes",      "color": "yellow"},
                            {"name": "Test",       "color": "gray"},
                        ]
                    }
                },
                "Date_Added": {"date": {}},
                "Summary": {"rich_text": {}},
                "Key_Concepts": {"multi_select": {"options": []}},
                "Understanding_Level": {
                    "select": {
                        "options": [
                            {"name": "New",             "color": "blue"},
                            {"name": "Understood",      "color": "green"},
                            {"name": "Needs Revisit",   "color": "orange"},
                            {"name": "Explore Further", "color": "purple"},
                        ]
                    }
                },
                "Is_New_Info":      {"checkbox": {}},
                "Similarity_Score": {"number": {"format": "number"}},
                "Embedding_ID":     {"rich_text": {}},
                "Last_Reviewed":    {"date": {}},
                "Is_Stale":         {"checkbox": {}},
            },
        )
        new_id = db["id"]
        set_key(str(ENV_PATH), "NOTION_DB_ID", new_id)
        print(f"[DB 1] Created. NOTION_DB_ID = {new_id}")
else:
    print("[DB 1] NOTION_DB_ID not set — creating Learning Agent KB...")
    db = notion.databases.create(
        parent={"type": "page_id", "page_id": PARENT_PAGE_ID},
        title=[{"type": "text", "text": {"content": "Learning Agent KB"}}],
        properties={
            "Title": {"title": {}},
            "Source": {"rich_text": {}},
            "Source_Type": {
                "select": {
                    "options": [
                        {"name": "YouTube",    "color": "red"},
                        {"name": "PDF",        "color": "blue"},
                        {"name": "Newsletter", "color": "green"},
                        {"name": "Notes",      "color": "yellow"},
                        {"name": "Test",       "color": "gray"},
                    ]
                }
            },
            "Date_Added": {"date": {}},
            "Summary": {"rich_text": {}},
            "Key_Concepts": {"multi_select": {"options": []}},
            "Understanding_Level": {
                "select": {
                    "options": [
                        {"name": "New",             "color": "blue"},
                        {"name": "Understood",      "color": "green"},
                        {"name": "Needs Revisit",   "color": "orange"},
                        {"name": "Explore Further", "color": "purple"},
                    ]
                }
            },
            "Is_New_Info":      {"checkbox": {}},
            "Similarity_Score": {"number": {"format": "number"}},
            "Embedding_ID":     {"rich_text": {}},
            "Last_Reviewed":    {"date": {}},
            "Is_Stale":         {"checkbox": {}},
        },
    )
    new_id = db["id"]
    set_key(str(ENV_PATH), "NOTION_DB_ID", new_id)
    print(f"[DB 1] Created. NOTION_DB_ID = {new_id}")

# ════════════════════════════════════════════════════════════════════════════
# DB 2 — Reading List (create if NOTION_READING_LIST_ID is unset / placeholder)
# ════════════════════════════════════════════════════════════════════════════
PLACEHOLDERS = {"FILL_IN_LATER", "", None}

existing_rl_id = os.getenv("NOTION_READING_LIST_ID")
rl_db_id = None
rl_just_created = False

if existing_rl_id not in PLACEHOLDERS:
    try:
        rl_db = notion.databases.retrieve(existing_rl_id)
        rl_db_id = rl_db["id"]
        print(f"\n[DB 2] Already exists: {rl_db['title'][0]['text']['content']}")
        print(f"       NOTION_READING_LIST_ID = {rl_db_id}")
    except Exception:
        print("\n[DB 2] NOTION_READING_LIST_ID in .env is invalid — recreating Reading List...")
        existing_rl_id = None  # fall through to create

if existing_rl_id in PLACEHOLDERS or rl_db_id is None:
    print("\n[DB 2] Creating Reading List database...")
    rl_db = notion.databases.create(
        parent={"type": "page_id", "page_id": PARENT_PAGE_ID},
        title=[{"type": "text", "text": {"content": "Reading List"}}],
        properties={
            "Title": {"title": {}},
            "Source": {"rich_text": {}},
            "Source_Type": {
                "select": {
                    "options": [
                        {"name": "YouTube",    "color": "red"},
                        {"name": "Newsletter", "color": "green"},
                        {"name": "PDF",        "color": "blue"},
                        {"name": "Notes",      "color": "yellow"},
                    ]
                }
            },
            "Date_Added": {"date": {}},
            "Summary": {"rich_text": {}},
            "New_Chunks": {"rich_text": {}},
            "Verdict": {
                "select": {
                    "options": [
                        {"name": "NEW",     "color": "green"},
                        {"name": "RELATED", "color": "yellow"},
                        {"name": "COe",     "color": "blue"},
                    ]
                }
            },
            "Similarity_Score": {"number": {"format": "number"}},
            "Status": {
                "select": {
                    "options": [
                        {"name": "To Read", "color": "orange"},
                        {"name": "Read",    "color": "green"},
                    ]
                }
            },
        },
    )
    rl_db_id = rl_db["id"]
    set_key(str(ENV_PATH), "NOTION_READING_LIST_ID", rl_db_id)
    rl_just_created = True
    print(f"[DB 2] Created.")
    print(f"\nNOTION_READING_LIST_ID = {rl_db_id}\n")
    print("       .env updated automatically.")

# ════════════════════════════════════════════════════════════════════════════
# Test entry — only written when DB 2 is freshly created
# ════════════════════════════════════════════════════════════════════════════
if rl_just_created:
    today = str(date.today())
    test_title = f"Reading List Test — {today}"

    print(f"\n[TEST] Writing test entry: '{test_title}' ...")
    page = notion.pages.create(
        parent={"database_id": rl_db_id},
        properties={
            "Title": {
                "title": [{"type": "text", "text": {"content": test_title}}]
            },
            "Source_Type": {"select": {"name": "Newsletter"}},
            "Status":      {"select": {"name": "To Read"}},
            "Date_Added":  {"date": {"start": today}},
        },
    )
    created_id = page["id"]

    result = notion.pages.retrieve(created_id)
    read_title = result["properties"]["Title"]["title"][0]["text"]["content"]

    if read_title == test_title:
        print(f"[TEST] PASS — entry confirmed: '{read_title}'")
    else:
        print(f"[TEST] FAIL — expected '{test_title}', got '{read_title}'")
else:
    print("\n[TEST] Skipped — both databases already exist, nothing to verify.")

# ════════════════════════════════════════════════════════════════════════════
# DB 3 — Ingest Queue (skip if already exists)
# ════════════════════════════════════════════════════════════════════════════
existing_iq_id = os.getenv("NOTION_INGEST_QUEUE_ID", "")
iq_db_id = None

if existing_iq_id:
    try:
        iq_db = notion.databases.retrieve(existing_iq_id)
        if not iq_db.get("archived"):
            iq_db_id = iq_db["id"]
            print(f"\n[DB 3] Already exists: {iq_db['title'][0]['text']['content']}")
            print(f"       NOTION_INGEST_QUEUE_ID = {iq_db_id}")
    except Exception:
        print("\n[DB 3] NOTION_INGEST_QUEUE_ID in .env is invalid — recreating Ingest Queue...")

if not iq_db_id:
    print("\n[DB 3] Creating Ingest Queue database...")
    iq_db = notion.databases.create(
        parent={"type": "page_id", "page_id": PARENT_PAGE_ID},
        title=[{"type": "text", "text": {"content": "Ingest Queue"}}],
        description=[{"type": "text", "text": {"content": "Paste a YouTube URL. It disappears when done, stays with an error if it fails."}}],
        properties={
            "URL":    {"title": {}},
            "Status": {
                "select": {
                    "options": [
                        {"name": "Processing", "color": "blue"},
                        {"name": "Failed",     "color": "red"},
                    ]
                }
            },
            "Error":   {"rich_text": {}},
            "Content": {"rich_text": {}},
            "Source_Type": {
                "select": {
                    "options": [
                        {"name": "YouTube",    "color": "red"},
                        {"name": "Link",       "color": "purple"},
                        {"name": "Newsletter", "color": "green"},
                        {"name": "Granola",    "color": "yellow"},
                        {"name": "PDF",        "color": "blue"},
                        {"name": "Notes",      "color": "gray"},
                    ]
                }
            },
        },
    )
    iq_db_id = iq_db["id"]
    set_key(str(ENV_PATH), "NOTION_INGEST_QUEUE_ID", iq_db_id)
    print(f"[DB 3] Created. NOTION_INGEST_QUEUE_ID = {iq_db_id}")
    print("       .env updated automatically.")
