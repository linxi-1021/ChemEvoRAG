#!/usr/bin/env python
"""Build evidence JSON files from a stored raw extraction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evidence import EvidenceBuilder  # noqa: E402
from normalization import RDKitNormalizer  # noqa: E402
from storage import LocalStore  # noqa: E402


def build_one(doc_id: str, base_dir: str, skip_normalize: bool, normalizer: RDKitNormalizer | None) -> bool:
    store = LocalStore(base_dir=base_dir)
    try:
        raw_payload = store.load_parsed(doc_id)
    except Exception as exc:
        print(f"  FAIL {doc_id}: {exc}", file=sys.stderr)
        return False

    bundle = EvidenceBuilder().build(raw_payload)

    if not skip_normalize and normalizer and normalizer.available:
        ok = fail = 0
        for i, card in enumerate(bundle.molecules):
            result = normalizer.normalize(card)
            if result.normalization_status == "success":
                ok += 1
            else:
                fail += 1
            bundle.molecules[i] = result
        if ok or fail:
            print(f"  normalize: {ok} ok, {fail} failed", file=sys.stderr)

    store.save_blocks(doc_id, bundle.blocks)
    store.save_molecules(doc_id, bundle.molecules)
    store.save_reactions(doc_id, bundle.reactions)
    store.save_facts(doc_id, bundle.facts)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc-id", default=None, help="Document id to build.")
    parser.add_argument("--all", action="store_true", help="Build all parsed documents.")
    parser.add_argument(
        "--base-dir",
        default=str(PROJECT_ROOT),
        help="Project base directory. Defaults to this repository root.",
    )
    parser.add_argument(
        "--skip-normalize",
        action="store_true",
        help="Skip RDKit normalization of molecule cards.",
    )
    args = parser.parse_args()

    if not args.doc_id and not args.all:
        parser.error("Either --doc-id or --all is required.")

    normalizer = RDKitNormalizer() if not args.skip_normalize else None

    if args.doc_id:
        ok = build_one(args.doc_id, args.base_dir, args.skip_normalize, normalizer)
        return 0 if ok else 1

    # Batch mode
    from tqdm import tqdm

    parsed_dir = Path(args.base_dir) / "data" / "parsed"
    doc_ids = sorted(
        p.stem.replace(".raw_extraction", "")
        for p in parsed_dir.glob("*.raw_extraction.json")
    )
    if not doc_ids:
        print("No parsed documents found.", file=sys.stderr)
        return 1

    ok = 0
    for doc_id in tqdm(doc_ids, desc="Building", unit="doc"):
        if build_one(doc_id, args.base_dir, args.skip_normalize, normalizer):
            ok += 1

    print(f"\nDone: {ok}/{len(doc_ids)} succeeded")
    return 0 if ok == len(doc_ids) else 1


if __name__ == "__main__":
    raise SystemExit(main())

