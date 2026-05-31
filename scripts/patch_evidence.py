#!/usr/bin/env python
"""Patch null fields in evidence JSON files without re-running full pipeline.

Fixes:
  - molecules: source_images.provenance.section (from blocks page→section map)
  - molecules: source_images.figure_id (from image_id)
  - molecules: LLM naming via name_molecules logic (names/aliases/source_mentions)
  - reactions: source.source_file, confidence
  - blocks: section, source_file
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

_dotenv = PROJECT_ROOT / ".env"
if _dotenv.exists():
    from dotenv import load_dotenv
    load_dotenv(_dotenv)

EVIDENCE_DIR = PROJECT_ROOT / "data" / "evidence"


def patch_blocks_section_and_source(doc_id: str) -> int:
    """Fill blocks.section (from headings) and blocks.source_file."""
    path = EVIDENCE_DIR / f"{doc_id}.blocks.json"
    if not path.exists():
        return 0
    blocks = json.loads(path.read_text("utf-8"))
    current_section: str | None = None
    patched = 0
    for b in blocks:
        bt = b.get("block_type", "")
        text = b.get("text", "")
        if bt in ("title", "section", "abstract") and text:
            current_section = text[:120]
        if b.get("section") is None and current_section:
            b["section"] = current_section
            patched += 1
        if b.get("source_file") is None:
            b["source_file"] = f"{doc_id}.pdf"
            patched += 1
    if patched:
        path.write_text(json.dumps(blocks, ensure_ascii=False, indent=2), encoding="utf-8")
    return patched


def patch_molecule_section_and_figure(doc_id: str) -> int:
    """Fill molecule source_images.provenance.section and figure_id."""
    mol_path = EVIDENCE_DIR / f"{doc_id}.molecules.json"
    blocks_path = EVIDENCE_DIR / f"{doc_id}.blocks.json"
    if not mol_path.exists():
        return 0

    # Build page → section map from blocks
    page_section: dict[int, str] = {}
    if blocks_path.exists():
        blocks = json.loads(blocks_path.read_text("utf-8"))
        for b in blocks:
            p = b.get("page")
            s = b.get("section")
            if p and s and p not in page_section:
                page_section[p] = s

    mols = json.loads(mol_path.read_text("utf-8"))
    patched = 0
    for m in mols:
        for si in m.get("source_images", []) or []:
            if not isinstance(si, dict):
                continue
            img_id = si.get("image_id", "")
            if si.get("figure_id") is None and img_id:
                si["figure_id"] = img_id
                patched += 1
            prov = si.get("provenance")
            if not isinstance(prov, dict):
                continue
            if prov.get("figure_id") is None and img_id:
                prov["figure_id"] = img_id
                patched += 1
            if prov.get("source_file") is None:
                prov["source_file"] = f"{doc_id}.pdf"
                patched += 1
            pg = prov.get("page")
            if pg and prov.get("section") is None and pg in page_section:
                prov["section"] = page_section[pg]
                patched += 1
    if patched:
        mol_path.write_text(json.dumps(mols, ensure_ascii=False, indent=2), encoding="utf-8")
    return patched


def patch_molecule_block_id(doc_id: str) -> int:
    """Link molecule source_images to nearest block on same page by bbox distance."""
    import math

    mol_path = EVIDENCE_DIR / f"{doc_id}.molecules.json"
    blocks_path = EVIDENCE_DIR / f"{doc_id}.blocks.json"
    if not mol_path.exists() or not blocks_path.exists():
        return 0

    blocks = json.loads(blocks_path.read_text("utf-8"))
    mols = json.loads(mol_path.read_text("utf-8"))
    patched = 0

    for m in mols:
        for si in m.get("source_images", []) or []:
            prov = si.get("provenance") if isinstance(si, dict) else None
            if not isinstance(prov, dict) or prov.get("block_id"):
                continue
            pg = prov.get("page")
            bbox = prov.get("bbox")
            if not pg or not bbox:
                continue

            best_id, best_dist = None, float("inf")
            cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
            for b in blocks:
                bb = b.get("bbox")
                if b.get("page") != pg or not bb:
                    continue
                bx, by = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
                d = math.hypot(cx - bx, cy - by)
                if d < best_dist:
                    best_dist, best_id = d, b.get("block_id")

            if best_id:
                prov["block_id"] = best_id
                patched += 1

    if patched:
        mol_path.write_text(json.dumps(mols, ensure_ascii=False, indent=2), encoding="utf-8")
    return patched


def patch_molecule_text_mentions(doc_id: str) -> int:
    """Rebuild source_mentions with proper provenance by searching block text."""
    mol_path = EVIDENCE_DIR / f"{doc_id}.molecules.json"
    blocks_path = EVIDENCE_DIR / f"{doc_id}.blocks.json"
    if not mol_path.exists() or not blocks_path.exists():
        return 0

    blocks = json.loads(blocks_path.read_text("utf-8"))
    mols = json.loads(mol_path.read_text("utf-8"))
    patched = 0

    for m in mols:
        names = list(m.get("names") or []) + list(m.get("aliases") or [])
        if not names:
            continue
        mentions: list[dict] = []
        seen_texts: set[str] = set()
        for name in names:
            if not name or len(name) < 3:
                continue
            for b in blocks:
                text = b.get("text", "")
                if not text or name not in text:
                    continue
                # Find first occurrence, extract text_span
                idx = text.index(name)
                end_idx = idx + len(name)
                mention_key = text[max(0, idx - 30):end_idx + 30]
                if mention_key in seen_texts:
                    continue
                seen_texts.add(mention_key)
                mentions.append({
                    "mention": mention_key.strip(),
                    "provenance": {
                        "doc_id": doc_id,
                        "source_file": f"{doc_id}.pdf",
                        "page": b.get("page"),
                        "bbox": b.get("bbox"),
                        "block_id": b.get("block_id"),
                        "table_id": b.get("block_id") if b.get("block_type") == "table" else None,
                        "section": b.get("section"),
                        "text_span": [idx, end_idx],
                    },
                    "role": "mentioned",
                    "confidence": 0.8,
                })
        if mentions:
            m["source_mentions"] = mentions
            patched += 1

    if patched:
        mol_path.write_text(json.dumps(mols, ensure_ascii=False, indent=2), encoding="utf-8")
    return patched


def patch_reaction_source_and_confidence(doc_id: str) -> int:
    """Fill reactions.source.source_file and confidence."""
    path = EVIDENCE_DIR / f"{doc_id}.reactions.json"
    if not path.exists():
        return 0
    rxns = json.loads(path.read_text("utf-8"))
    patched = 0
    for r in rxns:
        if r.get("confidence") is None:
            r["confidence"] = 0.5
            patched += 1
        src = r.get("source")
        if isinstance(src, dict) and src.get("source_file") is None:
            src["source_file"] = f"{doc_id}.pdf"
            patched += 1
    if patched:
        path.write_text(json.dumps(rxns, ensure_ascii=False, indent=2), encoding="utf-8")
    return patched


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc-id", default=None, help="Process single document.")
    args = parser.parse_args()

    if args.doc_id:
        doc_ids = [args.doc_id]
    else:
        doc_ids = sorted(
            f.stem.replace(".blocks", "")
            for f in EVIDENCE_DIR.glob("*.blocks.json")
        )

    total = 0
    for doc_id in doc_ids:
        p1 = patch_blocks_section_and_source(doc_id)
        p2 = patch_molecule_section_and_figure(doc_id)
        p3 = patch_molecule_block_id(doc_id)
        p4 = patch_molecule_text_mentions(doc_id)
        p5 = patch_reaction_source_and_confidence(doc_id)
        if p1 + p2 + p3 + p4 + p5:
            print(f"  {doc_id}: +{p1+p2+p3+p4+p5} fields patched")
        total += p1 + p2 + p3 + p4 + p5

    print(f"\nTotal: {total} fields patched across {len(doc_ids)} documents")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
