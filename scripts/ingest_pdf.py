#!/usr/bin/env python
"""Ingest chemistry PDFs into raw extraction storage."""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ingestion import PDFIngestor  # noqa: E402


def process_one(ingestor: PDFIngestor, pdf_path: Path, doc_id: str | None) -> bool:
    try:
        ingestor.ingest(str(pdf_path), doc_id=doc_id)
        return True
    except Exception as exc:
        tqdm.write(f"  FAIL {pdf_path.name}: {exc}", file=sys.stderr)
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "pdf_path", nargs="*",
        help="PDF file(s) to ingest.  Use --pdf-dir for a whole directory.",
    )
    parser.add_argument(
        "--pdf-dir", default=None,
        help="Process all PDFs in this directory.",
    )
    parser.add_argument("--doc-id", default=None, help="Document id (single PDF only).")
    parser.add_argument(
        "--base-dir", default=str(PROJECT_ROOT),
        help="Project base directory.",
    )
    parser.add_argument(
        "--copy-to-raw-dir", action="store_true",
        help="Copy PDF into data/raw_pdfs/{doc_id}.pdf before parsing.",
    )
    args = parser.parse_args()

    # Collect PDF paths
    if args.pdf_dir:
        pdf_paths = sorted(Path(args.pdf_dir).glob("*.pdf"))
    elif args.pdf_path:
        pdf_paths = [Path(p) for p in args.pdf_path]
    else:
        parser.error("Either pdf_path or --pdf-dir is required.")

    if not pdf_paths:
        print("No PDFs found.", file=sys.stderr)
        return 1

    ingestor = PDFIngestor(
        base_dir=args.base_dir,
        copy_to_raw_dir=args.copy_to_raw_dir,
    )

    def _cleanup() -> None:
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    ok = 0
    for pdf_path in tqdm(pdf_paths, desc="Ingesting", unit="pdf"):
        doc_id = args.doc_id or pdf_path.stem
        if process_one(ingestor, pdf_path, doc_id):
            ok += 1
        _cleanup()

    print(f"\nDone: {ok}/{len(pdf_paths)} succeeded")
    return 0 if ok == len(pdf_paths) else 1


if __name__ == "__main__":
    raise SystemExit(main())
