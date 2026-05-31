# Dual-Engine PDF Parsing Implementation Plan

> **Goal:** Replace single-tool PDF parsing with MinerU (text/layout) + OpenChemIE (molecule/reaction from figures) dual-engine pipeline.

**Architecture:** Read MinerU's pre-saved output from `data/mineru_output/{doc_id}/extracted/`, convert text items to DocumentBlock, feed extracted images to OpenChemIE for molecule/reaction recognition, merge results via LocalStore.

**Tech Stack:** Python 3.10, existing OpenChemIE adapter, RDKit normalizer, LocalStore

---

### Task 1: Create `scripts/dual_parse.py`

**Files:**
- Create: `scripts/dual_parse.py`

**Steps:**

- [ ] **Step 1: Write the script skeleton with argument parsing**

```python
#!/usr/bin/env python
"""Dual-engine PDF parsing: MinerU (text/layout) + OpenChemIE (molecules/reactions).

Reads MinerU output from data/mineru_output/{doc_id}/extracted/, converts
content_list_v2.json items to DocumentBlock, feeds extracted images to
OpenChemIE for molecule/reaction recognition, saves merged evidence.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evidence import DocumentBlock
from storage import LocalStore


def _extract_text(item: dict) -> str:
    """Extract plain text from nested MinerU content structure."""
    content = item.get("content", {})
    # title → title_content
    for key in ("title_content", "paragraph_content", "list_content"):
        parts = content.get(key, [])
        if parts:
            return " ".join(
                p.get("content", "") if isinstance(p, dict) else str(p)
                for p in parts
            ).strip()
    # Direct content string
    text = content.get("content", "")
    if isinstance(text, str) and text.strip():
        return text.strip()
    # Image caption
    caption = content.get("image_caption", [])
    if caption:
        return " ".join(
            c.get("content", "") if isinstance(c, dict) else str(c)
            for c in caption
        ).strip()
    return ""


def _text_items_to_blocks(
    doc_id: str, pages: list[list[dict]]
) -> list[DocumentBlock]:
    """Convert non-image content_list items to DocumentBlock objects."""
    blocks: list[DocumentBlock] = []
    BLOCK_TYPE_MAP = {
        "title": "title", "paragraph": "paragraph", "list": "paragraph",
        "table": "table", "page_header": "paragraph", "page_footer": "paragraph",
        "page_number": "paragraph", "page_aside_text": "paragraph",
        "abstract": "abstract", "section": "section",
    }
    idx = 0
    for page_idx, page_items in enumerate(pages):
        for item in page_items:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "image":
                continue
            text = _extract_text(item)
            if not text:
                continue
            raw_type = item.get("type", "paragraph")
            block_type = BLOCK_TYPE_MAP.get(raw_type, "paragraph")
            bbox = item.get("bbox")
            block = DocumentBlock(
                block_id=f"block_{doc_id}_{idx:04d}",
                doc_id=doc_id,
                block_type=block_type,
                text=text,
                page=int(page_idx) + 1,
                bbox=[float(v) for v in bbox] if bbox and len(bbox) == 4 else None,
                confidence=0.85,
            )
            blocks.append(block)
            idx += 1
    # Link blocks
    for j in range(len(blocks)):
        if j > 0:
            blocks[j].prev_block_id = blocks[j - 1].block_id
        if j < len(blocks) - 1:
            blocks[j].next_block_id = blocks[j + 1].block_id
    return blocks


def _load_images_with_meta(
    images_dir: Path, pages: list[list[dict]]
) -> list[dict]:
    """Load images and attach page/bbox metadata from content_list."""
    result: list[dict] = []
    for page_idx, page_items in enumerate(pages):
        for item in page_items:
            if not isinstance(item, dict) or item.get("type") != "image":
                continue
            img_src = item.get("content", {}).get("image_source", {})
            img_path = img_src.get("path", "")
            if not img_path:
                continue
            full_path = images_dir / Path(img_path).name
            if not full_path.exists():
                continue
            try:
                pil_img = Image.open(full_path).convert("RGB")
            except Exception:
                continue
            bbox = item.get("bbox")
            result.append({
                "image": pil_img,
                "page": int(page_idx) + 1,
                "bbox": [float(v) for v in bbox] if bbox and len(bbox) == 4 else None,
                "image_hash": full_path.stem,
                "sub_type": item.get("sub_type", ""),
            })
    return result


def _run_openchemie(
    images_meta: list[dict], doc_id: str
) -> tuple[list, list]:
    """Feed images to OpenChemIE, return (molecule_dicts, reaction_dicts)."""
    from parsing.openchemie_adapter import OpenChemIEAdapter

    adapter = OpenChemIEAdapter()
    pil_images = [m["image"] for m in images_meta]
    raw = adapter.parse_figures(pil_images, doc_id=doc_id)

    molecules: list[dict] = []
    reactions: list[dict] = []

    for item in raw.get("molecules", []):
        idx = item.get("figure_index", 0)
        meta = images_meta[idx] if idx < len(images_meta) else {}
        item["page"] = meta.get("page")
        item["bbox"] = meta.get("bbox")
        item["image_hash"] = meta.get("image_hash", "")
        molecules.append(item)

    for item in raw.get("reactions", []):
        idx = item.get("figure_index", 0)
        meta = images_meta[idx] if idx < len(images_meta) else {}
        item["page"] = meta.get("page")
        item["bbox"] = meta.get("bbox")
        item["image_hash"] = meta.get("image_hash", "")
        reactions.append(item)

    return molecules, reactions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mineru-dir",
        default=str(PROJECT_ROOT / "data" / "mineru_output"),
        help="Directory containing MinerU output per document.",
    )
    parser.add_argument(
        "--doc-id", default=None,
        help="Process a single document instead of all.",
    )
    args = parser.parse_args()

    mineru_root = Path(args.mineru_dir)
    if args.doc_id:
        doc_dirs = [mineru_root / args.doc_id]
    else:
        doc_dirs = sorted(mineru_root.iterdir())

    doc_dirs = [d for d in doc_dirs if d.is_dir() and (d / "extracted").is_dir()]
    if not doc_dirs:
        print("No MinerU output directories found.", file=sys.stderr)
        return 1

    store = LocalStore(base_dir=PROJECT_ROOT)
    ok = 0

    for doc_dir in tqdm(doc_dirs, desc="Dual parsing", unit="doc"):
        doc_id = doc_dir.name
        extracted = doc_dir / "extracted"

        # Find content_list_v2.json
        cl_files = list(extracted.glob("*_content_list_v2.json"))
        if not cl_files:
            tqdm.write(f"  SKIP {doc_id}: no content_list_v2.json")
            continue
        pages = json.loads(cl_files[0].read_text("utf-8"))

        # Step 1: Build blocks from text items
        blocks = _text_items_to_blocks(doc_id, pages)
        store.save_blocks(doc_id, blocks)
        tqdm.write(f"  {doc_id}: {len(blocks)} blocks")

        # Step 2: Load images
        images_dir = extracted / "images"
        images_meta = _load_images_with_meta(images_dir, pages)
        if not images_meta:
            tqdm.write(f"  {doc_id}: no images to process")
            ok += 1
            continue

        # Step 3: OpenChemIE
        tqdm.write(f"  {doc_id}: {len(images_meta)} images → OpenChemIE ...")
        try:
            mols, rxns = _run_openchemie(images_meta, doc_id)
        except Exception as exc:
            tqdm.write(f"  {doc_id}: OpenChemIE failed: {exc}")
            mols, rxns = [], []

        # Step 4: Save
        # (Simplified — full MoleculeCard/ReactionEventCard construction TBD)
        tqdm.write(f"  {doc_id}: {len(mols)} molecules, {len(rxns)} reactions")
        ok += 1

        # Memory cleanup
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    print(f"\nDone: {ok}/{len(doc_dirs)} documents")
    return 0 if ok == len(doc_dirs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

This is the skeleton — actual implementation requires verifying OpenChemIEAdapter's `parse_figures` method exists and returns the right format. The skeleton handles the MinerU → DocumentBlock flow correctly and sets up the image → OpenChemIE pipeline.
