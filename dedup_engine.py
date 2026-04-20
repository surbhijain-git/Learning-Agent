"""
dedup_engine.py
Semantic dedup engine — two ChromaDB collections:
  learning_agent_kb       — one vector per KB page (for doc-level NEW/RELATED/COVERED)
  learning_agent_kb_claims — one vector per claim/learning bullet (for concept-level report)

Usage:
    python dedup_engine.py sync              # embed un-embedded entries
    python dedup_engine.py sync --force      # re-embed ALL (use after schema changes)
    python dedup_engine.py check <page_id>   # doc-level novelty for one page
    python dedup_engine.py test              # self-test
"""

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv, find_dotenv
from notion_client import Client
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

load_dotenv(find_dotenv(usecwd=True), override=True)

notion = Client(auth=os.getenv("NOTION_TOKEN"))
NOTION_DB_ID           = os.getenv("NOTION_DB_ID")
NOTION_READING_LIST_ID = os.getenv("NOTION_READING_LIST_ID", "")

CHROMA_PATH     = Path("chroma_db")
DOC_COLLECTION  = "learning_agent_kb"
CLAIM_COLLECTION = "learning_agent_kb_claims"

# ── Domain baseline calibration ───────────────────────────────────────────────
# all-MiniLM-L6-v2 assigns ~0.45-0.55 cosine similarity to ANY two AI articles
# just from shared vocabulary (tokens like "model", "agent", "LLM", "strategy").
# Raw scores cluster at 0.5-0.6 for everything, making thresholds meaningless.
#
# Fix: normalize raw scores relative to this domain baseline so:
#   0.0 = same broad topic (AI), genuinely different content → NEW
#   0.5 = same AI sub-topic, overlapping framing → RELATED
#   1.0 = identical content (near-verbatim duplicate) → COVERED
#
# What the two comparison layers evaluate:
#   Doc-level:   key_concepts tags + full "Key Claims & Learnings" body text
#                → answers "is this document covering the same ground?"
#   Claim-level: each individual bullet compared against all stored bullets
#                → answers "has this specific insight appeared before?"
#
DOMAIN_BASELINE       = 0.50   # ambient similarity floor for AI-focused articles
CLAIM_DOMAIN_BASELINE = 0.38   # shorter texts → lower baseline

# Doc-level thresholds (applied to NORMALIZED scores 0.0–1.0)
# COVERED: normalized ≥ 0.65 → raw ≥ 0.50 + 0.65×0.50 = 0.825
# RELATED: normalized ≥ 0.20 → raw ≥ 0.50 + 0.20×0.50 = 0.60
COVERED_THRESHOLD = 0.65
RELATED_THRESHOLD = 0.20

# Claim-level thresholds (applied to NORMALIZED scores)
# COVERED: normalized ≥ 0.68 → raw ≥ 0.38 + 0.68×0.62 = 0.80
# RELATED: normalized ≥ 0.18 → raw ≥ 0.38 + 0.18×0.62 = 0.49
CLAIM_COVERED_THRESHOLD = 0.68
CLAIM_RELATED_THRESHOLD = 0.18


def _normalize(raw: float, baseline: float = DOMAIN_BASELINE) -> float:
    """Rescale raw cosine similarity relative to the domain baseline.
    Maps [baseline, 1.0] → [0.0, 1.0]. Scores at or below baseline → 0.0."""
    if raw <= baseline:
        return 0.0
    return round((raw - baseline) / (1.0 - baseline), 4)

_doc_col   = None
_claim_col = None


def _ef():
    return SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")


def get_collection():
    global _doc_col
    if _doc_col is None:
        client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        _doc_col = client.get_or_create_collection(
            name=DOC_COLLECTION,
            embedding_function=_ef(),
            metadata={"hnsw:space": "cosine"},
        )
    return _doc_col


def get_claims_collection():
    global _claim_col
    if _claim_col is None:
        client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        _claim_col = client.get_or_create_collection(
            name=CLAIM_COLLECTION,
            embedding_function=_ef(),
            metadata={"hnsw:space": "cosine"},
        )
    return _claim_col


# ── Embed text builders ───────────────────────────────────────────────────────

def make_embed_text(key_concepts: list[str], body_text: str) -> str:
    """
    Doc-level embedding: content only — no title, no summary.
    key_concepts = stable multi-select tags
    body_text    = "What I Learned" + "Key Claims" bullet verbatim text
    """
    concepts = ", ".join(key_concepts or [])
    body = (body_text or "")[:4000]
    return f"Topics: {concepts}\n\nKey claims and learnings:\n{body}"


# ── Notion helpers ────────────────────────────────────────────────────────────

def _rich_text_to_text(items: list[dict]) -> str:
    return "".join(part.get("plain_text", "") for part in items if part.get("plain_text"))


def _get_page_blocks(page_id: str) -> list[dict]:
    """Return all blocks from a Notion page."""
    blocks, cursor = [], None
    while True:
        kwargs = {"block_id": page_id, "page_size": 100}
        if cursor:
            kwargs["start_cursor"] = cursor
        resp = notion.blocks.children.list(**kwargs)
        blocks.extend(resp.get("results", []))
        if not resp.get("has_more"):
            break
        cursor = resp.get("next_cursor")
    return blocks


def _get_page_body_text(page_id: str) -> str:
    """Flatten all block text into a single string (for doc-level embedding)."""
    texts = []
    for block in _get_page_blocks(page_id):
        bt = block.get("type")
        if not bt:
            continue
        data = block.get(bt, {})
        if "rich_text" in data:
            t = _rich_text_to_text(data["rich_text"]).strip()
            if t:
                texts.append(t)
    return "\n".join(texts)


def _get_page_bullets(page_id: str) -> list[str]:
    """
    Extract only bullet-point text from a page body, stopping before
    the 'Concept Novelty' heading (which is derived, not source content).
    Used for the claims index.
    """
    bullets = []
    for block in _get_page_blocks(page_id):
        bt = block.get("type")
        if bt == "heading_3":
            text = _rich_text_to_text(block.get("heading_3", {}).get("rich_text", []))
            if "Concept Novelty" in text:
                break  # stop here — everything after is derived
        if bt == "bulleted_list_item":
            t = _rich_text_to_text(block.get("bulleted_list_item", {}).get("rich_text", []))
            if t.strip():
                bullets.append(t.strip())
    return bullets


def _get_notion_entry(page_id: str) -> tuple[str, list[str], str]:
    """Returns (title, key_concepts, body_text) for a Notion KB page."""
    page = notion.pages.retrieve(page_id)
    props = page["properties"]
    title = _rich_text_to_text(props.get("Title", {}).get("title", []))
    concepts = [
        c.get("name", "").strip()
        for c in props.get("Key_Concepts", {}).get("multi_select", [])
        if c.get("name")
    ]
    body_text = _get_page_body_text(page_id)
    return title, concepts, body_text


# ── Claims index helpers ──────────────────────────────────────────────────────

def _claim_ids(page_id: str, n: int) -> list[str]:
    return [f"{page_id}__c{i}" for i in range(n)]


def _upsert_claims(page_id: str, page_title: str, claims: list[str]):
    """Store individual claim/learning bullets in the claims collection."""
    if not claims:
        return
    col = get_claims_collection()
    ids = _claim_ids(page_id, len(claims))
    col.upsert(
        ids=ids,
        documents=claims,
        metadatas=[{"title": page_title, "page_id": page_id} for _ in claims],
    )


def _delete_claims(page_id: str):
    """Remove all claim entries for a page (used when re-syncing)."""
    col = get_claims_collection()
    try:
        # ChromaDB doesn't support prefix delete — query then delete
        existing = col.get(where={"page_id": {"$eq": page_id}})
        if existing["ids"]:
            col.delete(ids=existing["ids"])
    except Exception:
        pass


# ── Concept-level novelty report ──────────────────────────────────────────────

def concept_novelty_report(
    claims: list[str],
    learnings: list[str],
    self_page_id: str,
) -> dict:
    """
    Compare each new claim/learning against the CLAIMS index (claim vs claim).
    Returns {"new": [...], "related": [...], "covered": [...]}.
    """
    result = {"new": [], "related": [], "covered": []}
    col = get_claims_collection()

    all_items = [t for t in (claims + learnings) if t.strip()]
    if not all_items:
        return result

    # If claims index is empty or only has entries from self_page_id, everything is NEW
    if col.count() == 0:
        result["new"] = [{"text": t} for t in all_items]
        return result

    for text in all_items:
        n = min(5, col.count())
        res = col.query(
            query_texts=[text],
            n_results=n,
            include=["distances", "metadatas"],
        )
        # Find best match that isn't from the current page
        best_score = 0.0
        best_title = ""
        for dist, meta in zip(res["distances"][0], res["metadatas"][0]):
            if meta.get("page_id") == self_page_id:
                continue
            raw = 1.0 - dist
            best_score = _normalize(raw, CLAIM_DOMAIN_BASELINE)
            best_title = meta.get("title", "")
            break

        if best_score >= CLAIM_COVERED_THRESHOLD:
            result["covered"].append({"text": text, "match_title": best_title, "score": best_score})
        elif best_score >= CLAIM_RELATED_THRESHOLD:
            result["related"].append({"text": text, "match_title": best_title, "score": best_score})
        else:
            result["new"].append({"text": text})

    return result


def _append_novelty_blocks(page_id: str, report: dict):
    """Append a Concept Novelty section to the KB page."""
    new_items     = report.get("new", [])
    related_items = report.get("related", [])
    covered_items = report.get("covered", [])

    def para(text: str) -> dict:
        return {"object": "block", "type": "paragraph",
                "paragraph": {"rich_text": [{"type": "text", "text": {"content": text}}]}}

    def bullet(text: str) -> dict:
        return {"object": "block", "type": "bulleted_list_item",
                "bulleted_list_item": {"rich_text": [{"type": "text", "text": {"content": text}}]}}

    blocks = [
        {"object": "block", "type": "divider", "divider": {}},
        {"object": "block", "type": "heading_3",
         "heading_3": {"rich_text": [{"type": "text", "text": {"content": "Concept Novelty"}}]}},
    ]

    if new_items:
        blocks.append(para("🟢 New to your KB:"))
        blocks += [bullet(x["text"]) for x in new_items]
    if related_items:
        blocks.append(para("🟡 Related (new angle):"))
        blocks += [bullet(f"{x['text']}  →  \"{x['match_title'][:50]}\" ({x['score']:.2f})") for x in related_items]
    if covered_items:
        blocks.append(para("🔁 Already in KB:"))
        blocks += [bullet(f"{x['text']}  →  \"{x['match_title'][:50]}\" ({x['score']:.2f})") for x in covered_items]

    if not new_items and not related_items and not covered_items:
        blocks.append(para("No claims to analyse."))

    notion.blocks.children.append(block_id=page_id, children=blocks)


# ── Main novelty check ────────────────────────────────────────────────────────

def check_novelty(
    notion_page_id: str,
    claims: list[str] | None = None,
    learnings: list[str] | None = None,
) -> dict:
    """
    Doc-level: embed page → compare to doc collection → NEW/RELATED/COVERED.
    Concept-level (when claims/learnings provided): compare each claim to claims index.
    Updates Notion properties and appends Concept Novelty section to page body.
    """
    doc_col = get_collection()

    # 1. Fetch from Notion
    title, key_concepts, body_text = _get_notion_entry(notion_page_id)
    if not title:
        print("  [dedup] Skipped — no title found")
        return {"verdict": "SKIP", "score": 0.0, "match_title": "", "concept_report": {}}

    embed_text = make_embed_text(key_concepts, body_text)

    # 2. Doc-level similarity
    verdict, score, match_title, match_id = "NEW", 0.0, "", ""
    if doc_col.count() > 0:
        n = min(6, doc_col.count())
        res = doc_col.query(query_texts=[embed_text], n_results=n,
                            include=["distances", "metadatas"])
        for dist, rid, meta in zip(res["distances"][0], res["ids"][0], res["metadatas"][0]):
            if rid == notion_page_id:
                continue
            raw = 1.0 - dist
            score = _normalize(raw)
            match_id = rid
            match_title = meta.get("title", "")
            break
        if not match_id:
            score, verdict = 0.0, "NEW"
        elif score >= COVERED_THRESHOLD:
            verdict = "COVERED"
        elif score >= RELATED_THRESHOLD:
            verdict = "RELATED"
        else:
            verdict = "NEW"

    # 3. Print doc-level result (normalized score: 0.0=just-AI-topic, 1.0=identical)
    s = f"{score:.2f}"
    if verdict == "COVERED":
        print(f"  COVERED ({s}): similar to [{match_title}]")
    elif verdict == "RELATED":
        print(f"  RELATED ({s}): new angle, related to [{match_title}]")
    else:
        print(f"  NEW ({s}): added to knowledge base")

    # 4. Update Notion properties
    notion.pages.update(
        page_id=notion_page_id,
        properties={
            "Is_New_Info":      {"checkbox": verdict == "NEW"},
            "Similarity_Score": {"number": round(score, 4)},
            "Embedding_ID":     {"rich_text": [{"type": "text", "text": {"content": notion_page_id}}]},
        },
    )

    # 5. Store in doc collection
    doc_col.upsert(
        ids=[notion_page_id],
        documents=[embed_text],
        metadatas=[{"title": title}],
    )

    # 6. Concept-level report
    concept_report = {}
    if claims is not None or learnings is not None:
        print("  Running concept novelty analysis...")
        concept_report = concept_novelty_report(
            claims=claims or [],
            learnings=learnings or [],
            self_page_id=notion_page_id,
        )
        n_new = len(concept_report.get("new", []))
        n_rel = len(concept_report.get("related", []))
        n_cov = len(concept_report.get("covered", []))
        print(f"  Concepts: {n_new} new  {n_rel} related  {n_cov} already covered")
        _append_novelty_blocks(notion_page_id, concept_report)

        # 7. Index the new claims so future entries can match against them
        all_claims = (claims or []) + (learnings or [])
        _upsert_claims(notion_page_id, title, all_claims)

    return {
        "verdict":        verdict,
        "score":          score,
        "match_title":    match_title,
        "concept_report": concept_report,
    }


# ── Standalone novelty check (no KB page required) ────────────────────────────

def check_novelty_standalone(
    key_concepts: list[str],
    body_text: str,
    claims: list[str] | None = None,
    learnings: list[str] | None = None,
) -> dict:
    """
    Doc-level + concept-level novelty check WITHOUT needing a Notion KB page.
    Used by the ingest queue when writing to Reading List (pre-KB stage).
    Read-only: does NOT update ChromaDB or Notion.
    Returns {"verdict", "score", "match_title", "concept_report"}.
    """
    doc_col = get_collection()
    embed_text = make_embed_text(key_concepts, body_text)

    verdict, score, match_title = "NEW", 0.0, ""
    if doc_col.count() > 0:
        n = min(6, doc_col.count())
        res = doc_col.query(
            query_texts=[embed_text],
            n_results=n,
            include=["distances", "metadatas"],
        )
        for dist, meta in zip(res["distances"][0], res["metadatas"][0]):
            raw = 1.0 - dist
            score = _normalize(raw)
            match_title = meta.get("title", "")
            break
        if score >= COVERED_THRESHOLD:
            verdict = "COVERED"
        elif score >= RELATED_THRESHOLD:
            verdict = "RELATED"
        else:
            verdict = "NEW"

    concept_report = {}
    if claims is not None or learnings is not None:
        concept_report = concept_novelty_report(
            claims=claims or [],
            learnings=learnings or [],
            self_page_id="__standalone__",  # no real KB page yet — nothing to exclude
        )

    return {
        "verdict":        verdict,
        "score":          score,
        "match_title":    match_title,
        "concept_report": concept_report,
    }


# ── Sync ──────────────────────────────────────────────────────────────────────

def _fetch_all_notion_pages() -> list[dict]:
    pages, has_more, cursor = [], True, None
    while has_more:
        kwargs = {"database_id": NOTION_DB_ID, "page_size": 100}
        if cursor:
            kwargs["start_cursor"] = cursor
        resp = notion.databases.query(**kwargs)
        pages.extend(resp.get("results", []))
        has_more = resp.get("has_more", False)
        cursor = resp.get("next_cursor")
    return pages


def sync_all(force: bool = False):
    """
    Embed KB entries into both collections.
    --force re-embeds ALL (use after schema changes).
    """
    print("Fetching all Notion pages...")
    pages = _fetch_all_notion_pages()
    print(f"Found {len(pages)} pages in KB.")

    to_embed = pages if force else [
        p for p in pages
        if not p["properties"].get("Embedding_ID", {}).get("rich_text", [])
    ]
    label = f"--force: re-embedding all {len(to_embed)}" if force else f"{len(to_embed)} missing Embedding_ID"
    print(f"{label} — embedding now.\n")

    doc_col = get_collection()
    embedded = skipped = 0

    for i, page in enumerate(to_embed, start=1):
        page_id = page["id"]
        title_items = page["properties"].get("Title", {}).get("title", [])
        title = title_items[0]["text"]["content"] if title_items else ""

        if not title:
            print(f"  [{i}/{len(to_embed)}] SKIP (no title)")
            skipped += 1
            continue

        key_concepts = [
            c.get("name", "").strip()
            for c in page["properties"].get("Key_Concepts", {}).get("multi_select", [])
            if c.get("name")
        ]
        body_text  = _get_page_body_text(page_id)
        embed_text = make_embed_text(key_concepts, body_text)

        print(f"  [{i}/{len(to_embed)}] Embedding: {title[:60]}", end=" ... ", flush=True)

        # Doc collection
        doc_col.upsert(
            ids=[page_id],
            documents=[embed_text],
            metadatas=[{"title": title}],
        )

        # Claims collection — extract bullets from page body
        bullets = _get_page_bullets(page_id)
        if bullets:
            _delete_claims(page_id)   # clear stale entries before re-indexing
            _upsert_claims(page_id, title, bullets)

        notion.pages.update(
            page_id=page_id,
            properties={"Embedding_ID": {"rich_text": [{"type": "text", "text": {"content": page_id}}]}},
        )
        print(f"done ({len(bullets)} claims indexed)")
        embedded += 1
        time.sleep(0.3)

    print(f"\nSync complete. {embedded} embedded, {skipped} skipped.")
    print(f"Claims collection: {get_claims_collection().count()} total claim vectors.")


# ── Rescore ──────────────────────────────────────────────────────────────────

def rescore_all():
    """
    Re-compute Similarity_Score (normalized) for every KB entry and write to Notion.
    Run after changing thresholds or normalization logic.
    Requires ChromaDB to be synced first (run sync --force).
    """
    print("Fetching all Notion pages...")
    pages = _fetch_all_notion_pages()
    doc_col = get_collection()
    print(f"Rescoring {len(pages)} entries...\n")

    updated = skipped = 0
    for i, page in enumerate(pages, 1):
        page_id = page["id"]
        title_items = page["properties"].get("Title", {}).get("title", [])
        title = title_items[0]["text"]["content"] if title_items else ""
        if not title:
            skipped += 1
            continue

        key_concepts = [
            c.get("name", "").strip()
            for c in page["properties"].get("Key_Concepts", {}).get("multi_select", [])
            if c.get("name")
        ]
        body_text  = _get_page_body_text(page_id)
        embed_text = make_embed_text(key_concepts, body_text)

        score, match_title, verdict = 0.0, "", "NEW"
        if doc_col.count() > 1:
            n = min(6, doc_col.count())
            res = doc_col.query(query_texts=[embed_text], n_results=n,
                                include=["distances", "metadatas"])
            for dist, rid, meta in zip(res["distances"][0], res["ids"][0], res["metadatas"][0]):
                if rid == page_id:
                    continue
                raw = 1.0 - dist
                score = _normalize(raw)
                match_title = meta.get("title", "")
                break

            if score >= COVERED_THRESHOLD:
                verdict = "COVERED"
            elif score >= RELATED_THRESHOLD:
                verdict = "RELATED"

        notion.pages.update(
            page_id=page_id,
            properties={"Similarity_Score": {"number": score}},
        )
        print(f"  [{i:02d}/{len(pages)}] {title[:55]:<55}  {verdict:<8}  score={score:.3f}  [{match_title[:35]}]")
        updated += 1
        time.sleep(0.15)

    print(f"\nRescore complete. {updated} updated, {skipped} skipped.")


# ── Rescore Reading List ──────────────────────────────────────────────────────

def rescore_reading_list():
    """
    Re-compute normalized Similarity_Score + Verdict for all Reading List entries.
    Re-embeds each item's summary text and queries ChromaDB for nearest KB neighbor.
    """
    import httpx

    if not NOTION_READING_LIST_ID:
        print("NOTION_READING_LIST_ID not set — skipping.")
        return

    HEADERS = {
        "Authorization": f"Bearer {os.getenv('NOTION_TOKEN', '')}",
        "Notion-Version": "2022-06-28",
        "Content-Type": "application/json",
    }

    # Fetch all Reading List pages
    pages, cursor = [], None
    while True:
        body = {"page_size": 100}
        if cursor:
            body["start_cursor"] = cursor
        resp = httpx.post(
            f"https://api.notion.com/v1/databases/{NOTION_READING_LIST_ID}/query",
            headers=HEADERS, json=body,
        ).json()
        pages.extend(resp.get("results", []))
        if not resp.get("has_more"):
            break
        cursor = resp.get("next_cursor")

    print(f"Rescoring {len(pages)} Reading List entries...\n")

    doc_col = get_collection()
    if doc_col.count() == 0:
        print("ChromaDB KB collection is empty — run sync first.")
        return

    updated = 0
    for i, page in enumerate(pages, 1):
        props = page["properties"]

        title_items = props.get("Title", {}).get("title", [])
        title = "".join(t.get("plain_text", "") for t in title_items)

        summary_items = props.get("Summary", {}).get("rich_text", [])
        summary_text = "".join(t.get("plain_text", "") for t in summary_items)

        if not summary_text.strip():
            print(f"  [{i:02d}/{len(pages)}] SKIP (no summary): {title[:55]}")
            continue

        # Use full page body (claims/learnings bullets) for richer embedding
        # Summary alone is 2-3 sentences vs KB entries which have 10-20 claim bullets
        body_bullets = _get_page_body_text(page["id"])
        body_text = summary_text + ("\n\n" + body_bullets if body_bullets else "")
        embed_text = make_embed_text([], body_text)

        score, match_title, verdict = 0.0, "", "NEW"
        n = min(6, doc_col.count())
        res = doc_col.query(query_texts=[embed_text], n_results=n,
                            include=["distances", "metadatas"])
        for dist, meta in zip(res["distances"][0], res["metadatas"][0]):
            raw = 1.0 - dist
            score = _normalize(raw)
            match_title = meta.get("title", "")
            break

        if score >= COVERED_THRESHOLD:
            verdict = "COVERED"
        elif score >= RELATED_THRESHOLD:
            verdict = "RELATED"

        # Update Notion
        notion.pages.update(
            page_id=page["id"],
            properties={
                "Similarity_Score": {"number": score},
                "Verdict":          {"select": {"name": verdict}},
            },
        )
        print(f"  [{i:02d}/{len(pages)}] {title[:55]:<55}  {verdict:<8}  score={score:.3f}  [{match_title[:35]}]")
        updated += 1
        time.sleep(0.15)

    print(f"\nRescore complete. {updated} Reading List entries updated.")


# ── Self-test ─────────────────────────────────────────────────────────────────

def self_test():
    """
    Test 1 (COVERED): query existing entry's own content → doc similarity ≈ 1.0
    Test 2 (NEW):     supply chain text → doc similarity < 0.65
    Test 3 (claim COVERED): near-verbatim KB claim → claim similarity ≥ 0.72
    Test 4 (claim NEW):     off-topic claim → claim similarity < 0.55
    """
    print("=== Self-test: Semantic Dedup Engine ===\n")
    doc_col   = get_collection()
    claim_col = get_claims_collection()
    failures  = []

    if doc_col.count() == 0:
        print("SKIP — doc collection empty. Run sync first.")
        return

    pages = _fetch_all_notion_pages()
    test_page = next(
        (p for p in pages if p["properties"].get("Embedding_ID", {}).get("rich_text", [])),
        None
    )

    # ── Test 1: doc COVERED ───────────────────────────────────────────────────
    print("[Test 1] Doc COVERED — query with existing entry's own content...")
    if test_page:
        page_id = test_page["id"]
        title = (test_page["properties"].get("Title", {}).get("title", [{}])[0]
                 .get("text", {}).get("content", ""))
        key_concepts = [c.get("name", "") for c in test_page["properties"]
                        .get("Key_Concepts", {}).get("multi_select", [])]
        body = _get_page_body_text(page_id)
        embed_text = make_embed_text(key_concepts, body)
        res = doc_col.query(query_texts=[embed_text], n_results=1, include=["distances", "metadatas"])
        raw = 1.0 - res["distances"][0][0]
        sim = _normalize(raw)
        top = res["metadatas"][0][0].get("title", "")
        print(f"  Testing: {title[:60]}  →  match: [{top[:50]}]  raw={raw:.2f}  normalized={sim:.2f}")
        if sim >= COVERED_THRESHOLD:
            print(f"  Test 1 PASS — normalized {sim:.2f} >= {COVERED_THRESHOLD}\n")
        else:
            msg = f"Test 1 FAIL: normalized similarity {sim:.2f} < {COVERED_THRESHOLD} (raw={raw:.2f})"
            print(f"  {msg}\n"); failures.append(msg)
    else:
        print("  SKIP — no embedded pages.\n")

    # ── Test 2: doc NEW ───────────────────────────────────────────────────────
    print("[Test 2] Doc NEW — supply chain text (expect normalized = 0.0)...")
    new_text = ("Cold chain disruption in pharmaceutical freight forwarding. "
                "Warehouse inventory LIFO/FIFO optimisation. Cross-border customs clearance. "
                "Port congestion, demurrage charges, 3PL vendor selection.")
    res = doc_col.query(query_texts=[new_text], n_results=1, include=["distances", "metadatas"])
    raw = 1.0 - res["distances"][0][0]
    sim = _normalize(raw)
    top = res["metadatas"][0][0].get("title", "") if res["metadatas"] else ""
    print(f"  Nearest match: [{top[:60]}]  raw={raw:.2f}  normalized={sim:.2f}")
    if sim < RELATED_THRESHOLD:
        print(f"  Test 2 PASS — normalized {sim:.2f} < {RELATED_THRESHOLD}\n")
    else:
        msg = f"Test 2 FAIL: normalized {sim:.2f} >= {RELATED_THRESHOLD} (raw={raw:.2f})"
        print(f"  {msg}\n"); failures.append(msg)

    # ── Test 3: claim COVERED ─────────────────────────────────────────────────
    if claim_col.count() > 0:
        print("[Test 3] Claim COVERED — near-verbatim KB claim vs claims index...")
        # Use text very close to an actual stored claim — should score ~0.90+
        known_claim = "Distribution incumbency is now more defensible than model quality: Google Gemini reaches two billion monthly searchers through AI Overviews regardless of benchmark rankings"
        res = claim_col.query(query_texts=[known_claim], n_results=1, include=["distances", "metadatas"])
        raw = 1.0 - res["distances"][0][0]
        sim = _normalize(raw, CLAIM_DOMAIN_BASELINE)
        top = res["metadatas"][0][0].get("title", "") if res["metadatas"] else ""
        print(f"  Nearest match: [{top[:60]}]  raw={raw:.2f}  normalized={sim:.2f}")
        if sim >= CLAIM_COVERED_THRESHOLD:
            print(f"  Test 3 PASS — normalized {sim:.2f} >= {CLAIM_COVERED_THRESHOLD} (COVERED)\n")
        else:
            msg = f"Test 3 FAIL: normalized {sim:.2f} < {CLAIM_COVERED_THRESHOLD} — known theme not flagged COVERED (raw={raw:.2f})"
            print(f"  {msg}\n"); failures.append(msg)

        # ── Test 4: claim NEW ─────────────────────────────────────────────────
        print("[Test 4] Claim NEW — off-topic claim vs claims index...")
        off_claim = "Autonomous drone swarms use mesh networking for last-mile pharmaceutical logistics"
        res = claim_col.query(query_texts=[off_claim], n_results=1, include=["distances", "metadatas"])
        raw = 1.0 - res["distances"][0][0]
        sim = _normalize(raw, CLAIM_DOMAIN_BASELINE)
        top = res["metadatas"][0][0].get("title", "") if res["metadatas"] else ""
        print(f"  Nearest match: [{top[:60]}]  raw={raw:.2f}  normalized={sim:.2f}")
        if sim < CLAIM_RELATED_THRESHOLD:
            print(f"  Test 4 PASS — normalized {sim:.2f} < {CLAIM_RELATED_THRESHOLD} (NEW)\n")
        else:
            msg = f"Test 4 FAIL: normalized {sim:.2f} >= {CLAIM_RELATED_THRESHOLD} (raw={raw:.2f})"
            print(f"  {msg}\n"); failures.append(msg)
    else:
        print("[Test 3/4] SKIP — claims collection empty. Run sync first.\n")

    print("PASS — all tests correct." if not failures
          else "FAIL:\n" + "\n".join(f"  - {f}" for f in failures))


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Semantic dedup engine for Learning Agent KB")
    sub = parser.add_subparsers(dest="cmd")

    sync_p = sub.add_parser("sync", help="Embed entries (--force to re-embed all)")
    sync_p.add_argument("--force", action="store_true")

    check_p = sub.add_parser("check", help="Check novelty for one Notion page")
    check_p.add_argument("page_id")

    sub.add_parser("rescore", help="Re-compute normalized Similarity_Score for all KB entries in Notion")
    sub.add_parser("rescore-rl", help="Re-compute normalized Similarity_Score for all Reading List entries")
    sub.add_parser("test", help="Run self-test")

    args = parser.parse_args()

    if args.cmd == "sync":
        sync_all(force=getattr(args, "force", False))
    elif args.cmd == "rescore":
        rescore_all()
    elif args.cmd == "rescore-rl":
        rescore_reading_list()
    elif args.cmd == "check":
        result = check_novelty(args.page_id)
        print(f"Verdict: {result['verdict']}  Score: {result['score']:.4f}")
    elif args.cmd == "test":
        self_test()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
