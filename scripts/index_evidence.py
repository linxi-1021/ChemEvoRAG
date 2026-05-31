#!/usr/bin/env python
"""Build the Qdrant vector index from stored evidence JSON files.

Usage:
    conda activate chemevorag
    cd ChemEvoRAG_Phase1-main
    python scripts/index_evidence.py                     # all documents
    python scripts/index_evidence.py --doc-id paper_001   # single document
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from retrieval import DenseRetriever
from storage import LocalStore
from storage.qdrant_store import QdrantStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--doc-id", default=None, help="Index a single document instead of all."
    )
    parser.add_argument(
        "--base-dir",
        default=str(PROJECT_ROOT),
        help="Project base directory.",
    )
    parser.add_argument(
        "--qdrant-path",
        default=None,
        help="Qdrant data directory. Defaults to data/indexes/qdrant/",
    )
    args = parser.parse_args()

    base = Path(args.base_dir)
    qdrant_path = (
        str(base / "data" / "indexes" / "qdrant")
        if args.qdrant_path is None
        else args.qdrant_path
    )

    store = LocalStore(base_dir=base)
    qdrant = QdrantStore(path=qdrant_path)
    retriever = DenseRetriever(qdrant)

    doc_ids = [args.doc_id] if args.doc_id else None
    count = retriever.index(store, doc_ids=doc_ids)
    print(f"Indexed {count} evidence points into {qdrant_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
