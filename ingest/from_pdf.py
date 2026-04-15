"""
ingest/from_pdf.py
Usage: python ingest/from_pdf.py <pdf_folder> [--out raw_inputs/pdfs]

Extracts text from PDF files using PyMuPDF (fitz).
Saves each PDF as a .txt file in the output folder.
Skips scanned-only PDFs (no text layer) with a warning.
Skips already-converted PDFs (dedup by filename).

After running, point db_seeder.py at the output folder:
  python db_seeder.py raw_inputs/pdfs/

Note: HBS cases are often scanned images — those will show a warning.
For scanned PDFs, OCR would be required (not implemented here).
"""

import sys
import argparse
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def extract_text_from_pdf(pdf_path: Path) -> tuple[str, str]:
    """
    Returns (text, status) where status is 'ok', 'scanned', or 'error'.
    Requires PyMuPDF: pip install PyMuPDF
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("PyMuPDF not installed. Run: pip install PyMuPDF --break-system-packages")

    try:
        doc = fitz.open(str(pdf_path))
        pages_text = []
        total_chars = 0

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            pages_text.append(text)
            total_chars += len(text.strip())

        doc.close()
        full_text = "\n\n".join(pages_text).strip()

        # Heuristic: if <50 chars per page on average, likely scanned
        avg_chars = total_chars / max(len(pages_text), 1)
        if avg_chars < 50:
            return full_text, "scanned"

        return full_text, "ok"

    except Exception as e:
        return "", f"error: {e}"


def main():
    parser = argparse.ArgumentParser(description="Extract text from PDFs for db_seeder")
    parser.add_argument("pdf_folder", help="Folder containing PDF files")
    parser.add_argument(
        "--out", default="raw_inputs/pdfs",
        help="Output folder for .txt files (default: raw_inputs/pdfs)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-extract even if .txt already exists"
    )
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_folder)
    if not pdf_dir.is_dir():
        print(f"Not a directory: {pdf_dir}")
        sys.exit(1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(pdf_dir.glob("*.pdf")) + sorted(pdf_dir.glob("*.PDF"))
    total = len(pdfs)

    if total == 0:
        print(f"No PDF files found in {pdf_dir}")
        sys.exit(0)

    print(f"Found {total} PDFs in {pdf_dir}")

    converted = 0
    skipped_exist = 0
    skipped_scanned = 0
    failed = 0

    for i, pdf_path in enumerate(pdfs, start=1):
        out_filename = pdf_path.stem + ".txt"
        out_path = out_dir / out_filename
        print(f"  [{i}/{total}] {pdf_path.name}", end=" ... ", flush=True)

        # Dedup check
        if out_path.exists() and not args.force:
            print("SKIP (already converted)")
            skipped_exist += 1
            continue

        text, status = extract_text_from_pdf(pdf_path)

        if status == "scanned":
            if len(text.strip()) > 200:
                # Some text exists even in "mostly scanned" PDFs — save it
                header = f"Source: {pdf_path.name}\nExtracted: partial (likely scanned PDF)\n---\n\n"
                out_path.write_text(header + text, encoding="utf-8")
                word_count = len(text.split())
                print(f"PARTIAL ({word_count} words, likely scanned — OCR not run)")
                converted += 1
            else:
                print(f"SKIP (scanned PDF, no text layer — OCR required)")
                skipped_scanned += 1
            continue

        if status.startswith("error") or not text.strip():
            print(f"FAIL ({status})")
            failed += 1
            continue

        # Add metadata header to the text file
        header = f"Source: {pdf_path.name}\n---\n\n"
        out_path.write_text(header + text, encoding="utf-8")
        word_count = len(text.split())
        print(f"OK ({word_count} words)")
        converted += 1

    print(
        f"\nDone. {converted} converted. "
        f"{skipped_exist} skipped (already done). "
        f"{skipped_scanned} skipped (scanned/no text). "
        f"{failed} failed."
    )

    if skipped_scanned > 0:
        print(
            f"\nNote: {skipped_scanned} scanned PDFs skipped. "
            "HBS cases are often scanned — you can manually copy key excerpts into .txt files."
        )

    if converted > 0:
        print(f"\nNext step: python db_seeder.py {out_dir}")


if __name__ == "__main__":
    main()
