"""
youtube_extractor.py
Usage: python youtube_extractor.py <youtube_url>

Pulls transcript via captions first; falls back to yt-dlp + Whisper if unavailable.
Saves output to raw_transcripts/<video_id>.json
"""

import sys
import os
import json
import re
import subprocess
import shutil
from pathlib import Path
from datetime import datetime

# ── Validate argument ─────────────────────────────────────────────────────────
if len(sys.argv) < 2:
    print("Usage: python youtube_extractor.py <youtube_url>")
    sys.exit(1)

URL = sys.argv[1]
OUTPUT_DIR = Path("raw_transcripts")
OUTPUT_DIR.mkdir(exist_ok=True)


def extract_video_id(url: str) -> str:
    patterns = [
        r"(?:v=|youtu\.be/|embed/|shorts/)([A-Za-z0-9_-]{11})",
    ]
    for pat in patterns:
        m = re.search(pat, url)
        if m:
            return m.group(1)
    raise ValueError(f"Could not extract video ID from URL: {url}")


def check_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def chunk_text(text: str, max_words: int = 500) -> list[str]:
    """Split text into ~max_words chunks at sentence boundaries."""
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


def get_yt_metadata(url: str) -> dict:
    """Get video title and uploader via yt-dlp."""
    try:
        result = subprocess.run(
            ["yt-dlp", "--print", "%(title)s\n%(uploader)s", "--no-download", url],
            capture_output=True, text=True, timeout=30
        )
        lines = result.stdout.strip().splitlines()
        return {
            "title": lines[0] if len(lines) > 0 else "Unknown Title",
            "uploader": lines[1] if len(lines) > 1 else "Unknown",
        }
    except Exception:
        return {"title": "Unknown Title", "uploader": "Unknown"}


# ── Step 1: Extract video ID ──────────────────────────────────────────────────
try:
    video_id = extract_video_id(URL)
except ValueError:
    print("Invalid YouTube link. Please provide a valid URL.")
    sys.exit(1)

output_path = OUTPUT_DIR / f"{video_id}.json"

print(f"Video ID: {video_id}")

# ── Step 2: Try captions first ────────────────────────────────────────────────
transcript_text = None
transcript_method = None

try:
    from youtube_transcript_api import YouTubeTranscriptApi
    api = YouTubeTranscriptApi()
    fetched = api.fetch(video_id)
    transcript_text = " ".join(snippet.text for snippet in fetched)
    transcript_method = "captions"
    print("Transcript: captions")
except Exception as caption_err:
    print(f"Captions unavailable ({caption_err}) — trying Whisper fallback...")

    # ── Check ffmpeg before attempting Whisper ────────────────────────────────
    if not check_ffmpeg():
        print("Install ffmpeg: brew install ffmpeg (Mac) or winget install ffmpeg (Windows)")
        sys.exit(1)

    # ── yt-dlp download ───────────────────────────────────────────────────────
    audio_path = OUTPUT_DIR / f"audio_{video_id}.mp3"

    print(f"Downloading audio to {audio_path} ...")
    result = subprocess.run(
        [
            "yt-dlp",
            "--extract-audio",
            "--audio-format", "mp3",
            "-o", str(audio_path),
            URL,
        ],
        capture_output=True, text=True
    )

    # Explicit file existence + size check
    if not audio_path.exists() or audio_path.stat().st_size == 0:
        raise RuntimeError(
            f"yt-dlp download failed — audio file not found at {audio_path}"
        )

    print("Audio downloaded. Transcribing with Whisper (base model)...")

    # ── Whisper transcription ─────────────────────────────────────────────────
    import whisper
    model = whisper.load_model("base")
    whisper_result = model.transcribe(str(audio_path))
    transcript_text = whisper_result["text"]
    transcript_method = "whisper"

    # Delete audio after successful transcription
    audio_path.unlink()
    print("Transcript: whisper fallback")

# ── Step 3: Chunk transcript ──────────────────────────────────────────────────
chunks = chunk_text(transcript_text)
print(f"Split into {len(chunks)} chunks (~500 words each)")

# ── Step 4: Get metadata ──────────────────────────────────────────────────────
meta = get_yt_metadata(URL)

# ── Step 5: Save JSON ─────────────────────────────────────────────────────────
output = {
    "video_id": video_id,
    "url": URL,
    "title": meta["title"],
    "uploader": meta["uploader"],
    "source_type": "YouTube",
    "transcript_method": transcript_method,
    "date_extracted": datetime.today().strftime("%Y-%m-%d"),
    "chunks": chunks,
}

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"Saved to {output_path}")
