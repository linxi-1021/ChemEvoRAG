#!/usr/bin/env python
"""Complete PDF parsing pipeline: MinerU API + OpenChemIE dual-engine.

Flow:
  1. Upload PDFs via MinerU batch API → poll → download full_zip_url
  2. Convert content_list_v2.json items → DocumentBlock (text/tables/titles)
  3. Feed extracted images → OpenChemIE for molecule/reaction recognition
  4. Save blocks, molecules, reactions via LocalStore

Cached: if data/mineru_output/{doc_id}/extracted/ already exists, MinerU is skipped.
"""

from __future__ import annotations

import argparse
import gc
import io as _io
import json
import os
import sys
import time as _time
import zipfile as _zipfile
from pathlib import Path

import requests as _requests
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
MODEL_DIR = PROJECT_ROOT / "data" / "models"
MINERU_ROOT = PROJECT_ROOT / "data" / "mineru_output"

# Load .env for MINERU_TOKEN
_dotenv = PROJECT_ROOT / ".env"
if _dotenv.exists():
    from dotenv import load_dotenv
    load_dotenv(_dotenv)

for _env, _subdir in [("HF_HOME", "huggingface"), ("TORCH_HOME", "torch")]:
    os.environ.setdefault(_env, str(MODEL_DIR / _subdir))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evidence import (
    DocumentBlock, MoleculeCard, MoleculeImageSource, MoleculeMention,
    ReactionEventCard, ReactionParticipant, SourceProvenance, YieldValue,
)
from normalization import RDKitNormalizer
from storage import LocalStore

MINERU_BASE = "https://mineru.net"
POLL_INTERVAL = 15
MAX_POLL_MINUTES = 120

BLOCK_TYPE_MAP = {
    "title": "title", "paragraph": "paragraph", "list": "paragraph",
    "table": "table", "abstract": "abstract", "section": "section",
    "page_header": "paragraph", "page_footer": "paragraph",
    "page_number": "paragraph", "page_aside_text": "paragraph",
}


# ── MinerU API ─────────────────────────────────────────────────


def _mineru_headers() -> dict:
    token = os.environ.get("MINERU_TOKEN", "")
    if not token or token == "your_token_here":
        raise RuntimeError("MINERU_TOKEN not set in .env")
    return {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}


def _mineru_upload_and_poll(pdf_paths: list[Path], pdf_dir: Path) -> None:
    """Upload PDFs to MinerU, poll until done, download and extract all."""
    MINERU_ROOT.mkdir(parents=True, exist_ok=True)
    headers = _mineru_headers()

    # Step 1: batch upload URLs
    print(f"\n{'='*60}")
    print(f"Step 1: Requesting upload URLs for {len(pdf_paths)} files ...")
    data = {
        "files": [{"name": p.name, "data_id": p.stem} for p in pdf_paths],
        "model_version": "vlm",
        "enable_table": True,
        "enable_formula": False,
        "language": "en",
    }
    r = _requests.post(f"{MINERU_BASE}/api/v4/file-urls/batch", headers=headers, json=data, timeout=30)
    r.raise_for_status()
    body = r.json()
    if body.get("code") != 0:
        raise RuntimeError(f"batch create failed: {body}")
    batch_id = body["data"]["batch_id"]
    file_urls = body["data"]["file_urls"]
    print(f"  batch_id: {batch_id}")

    # Step 2: upload
    print(f"\nStep 2: Uploading PDFs ...")
    for pdf, url in tqdm(list(zip(pdf_paths, file_urls)), desc="Uploading", unit="pdf"):
        r2 = _requests.put(url, data=pdf.read_bytes(), timeout=300)
        if r2.status_code != 200:
            tqdm.write(f"  FAIL {pdf.name}: HTTP {r2.status_code}", file=sys.stderr)

    # Step 3: poll
    print(f"\nStep 3: Waiting for parsing (polling every {POLL_INTERVAL}s) ...")
    poll_url = f"{MINERU_BASE}/api/v4/extract-results/batch/{batch_id}"
    deadline = _time.time() + MAX_POLL_MINUTES * 60
    results: list[dict] = []
    with tqdm(desc="Parsing", unit=" files", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]") as pbar:
        while _time.time() < deadline:
            r3 = _requests.get(poll_url, headers=headers, timeout=30)
            r3.raise_for_status()
            b = r3.json()
            if b.get("code") != 0:
                _time.sleep(POLL_INTERVAL)
                continue
            results = b["data"].get("extract_result", [])
            done = sum(1 for x in results if x["state"] in ("done", "failed"))
            pbar.total = len(results)
            pbar.n = done
            pbar.refresh()
            if done == len(results):
                break
            _time.sleep(POLL_INTERVAL)
        else:
            tqdm.write("  Timeout", file=sys.stderr)

    # Step 4: download and extract
    print(f"\nStep 4: Downloading results ...")
    ok = fail = 0
    for item in results:
        fname = item["file_name"]
        if item["state"] != "done":
            tqdm.write(f"  SKIP {fname}: {item['state']} {item.get('err_msg', '')}", file=sys.stderr)
            fail += 1
            continue
        doc_dir = MINERU_ROOT / Path(fname).stem
        extract_dir = doc_dir / "extracted"
        if extract_dir.is_dir():
            ok += 1
            continue
        doc_dir.mkdir(parents=True, exist_ok=True)
        try:
            _download_and_extract(item["full_zip_url"], extract_dir)
            # vlm hallucination fallback: if <50% ASCII, re-parse with pipeline
            md_files = list(extract_dir.glob("*.md"))
            if md_files:
                md_text = md_files[0].read_text("utf-8", errors="replace")
                if (sum(1 for c in md_text if ord(c) < 128) / max(len(md_text), 1)) < 0.5:
                    tqdm.write(f"  {fname}: vlm produced non-English — re-parsing with pipeline ...")
                    pdf_path = pdf_dir / fname
                    if pdf_path.exists():
                        _reparse_with_pipeline(fname, pdf_path, extract_dir, headers)
            ok += 1
        except Exception as exc:
            tqdm.write(f"  FAIL {fname}: {exc}", file=sys.stderr)
            fail += 1
    print(f"  done={ok}, failed={fail}")


# ── MinerU → DocumentBlock ────────────────────────────────────


def _extract_text(item: dict) -> str:
    content = item.get("content", {})
    for key in ("title_content", "paragraph_content", "list_content"):
        parts = content.get(key, [])
        if parts:
            return " ".join(
                p.get("content", "") if isinstance(p, dict) else str(p)
                for p in parts
            ).strip()
    text = content.get("content", "")
    if isinstance(text, str) and text.strip():
        return text.strip()
    caption = content.get("image_caption", [])
    if isinstance(caption, list) and caption:
        return " ".join(
            c.get("content", "") if isinstance(c, dict) else str(c)
            for c in caption
        ).strip()
    # Table: convert HTML to readable text + footnotes + caption
    if item.get("type") == "table":
        return _extract_table_text(content)
    return ""


def _extract_table_text(content: dict) -> str:
    """Convert MinerU table HTML to searchable plain text."""
    parts: list[str] = []
    html = content.get("html", "")
    if html:
        parts.append(_html_table_to_text(html))
    # Table caption
    for cap in content.get("table_caption", []) or []:
        if isinstance(cap, dict):
            parts.append(cap.get("content", ""))
        elif isinstance(cap, str):
            parts.append(cap)
    # Table footnotes (often contain reaction conditions)
    for fn in content.get("table_footnote", []) or []:
        if isinstance(fn, dict):
            parts.append(fn.get("content", ""))
        elif isinstance(fn, str):
            parts.append(fn)
    return " | ".join(p.strip() for p in parts if p and p.strip())


def _html_table_to_text(html: str) -> str:
    """Extract rows from an HTML table as pipe-separated text."""
    import re
    rows: list[str] = []
    for tr_match in re.finditer(r"<tr[^>]*>(.*?)</tr>", html, re.DOTALL):
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr_match.group(1), re.DOTALL)
        # Strip HTML tags from cell content
        clean_cells = [re.sub(r"<[^>]+>", "", c).strip() for c in cells]
        if any(clean_cells):
            rows.append(" | ".join(clean_cells))
    return "; ".join(rows)


def _build_blocks(doc_id: str, pages: list) -> list[DocumentBlock]:
    blocks: list[DocumentBlock] = []
    idx = 0
    current_section: str | None = None
    for page_idx, page_items in enumerate(pages):
        if not isinstance(page_items, list):
            continue
        for item in page_items:
            if not isinstance(item, dict) or item.get("type") == "image":
                continue
            raw_type = item.get("type", "paragraph")
            text = _extract_text(item)
            # Track current section from headings
            if raw_type in ("title", "section", "abstract"):
                if text:
                    current_section = text[:120]
                if not text:
                    continue
            if not text:
                continue
            bbox = item.get("bbox")
            blocks.append(DocumentBlock(
                block_id=f"block_{doc_id}_{idx:04d}",
                doc_id=doc_id,
                block_type=BLOCK_TYPE_MAP.get(raw_type, "paragraph"),
                text=text,
                page=int(page_idx) + 1,
                bbox=[float(v) for v in bbox] if bbox and len(bbox) == 4 else None,
                section=current_section,
                source_file=f"{doc_id}.pdf",
                confidence=0.85,
                raw_payload={"mineru_type": raw_type},
            ))
            idx += 1
    for j in range(len(blocks)):
        if j > 0:
            blocks[j].prev_block_id = blocks[j - 1].block_id
        if j < len(blocks) - 1:
            blocks[j].next_block_id = blocks[j + 1].block_id
    return blocks


def _load_images(images_dir: Path, pages: list) -> list[dict]:
    result: list[dict] = []
    for page_idx, page_items in enumerate(pages):
        if not isinstance(page_items, list):
            continue
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
            })
    return result


# ── OpenChemIE ─────────────────────────────────────────────────


def _init_local_models(model):
    weights = [
        ("molscribe", "swin_base_char_aux_1m680k.pth"),
        ("pdfparser", "publaynet-tf_efficientdet_d1.pth.tar"),
        ("rxnscribe", "pix2seq_reaction_full.ckpt"),
        ("moldet",    "best_hf.ckpt"),
        ("coref",     "coref_best_hf.ckpt"),
        ("chemner",   "best.ckpt"),
    ]
    for suffix, filename in weights:
        candidate = MODEL_DIR / filename
        if not candidate.exists():
            continue
        init = getattr(model, f"init_{suffix}", None)
        if init is None:
            continue
        try:
            import torch as _t
            _t.cuda.empty_cache()
            init(str(candidate))
        except Exception:
            pass

    ms = MODEL_DIR / "swin_base_char_aux_1m680k.pth"
    ms1 = MODEL_DIR / "swin_base_char_aux_1m.pth"
    if ms.exists() or ms1.exists():
        try:
            from molscribe import MolScribe as MS
            ms_path = str(ms if ms.exists() else ms1)
            import rxnscribe.interface as ri
            def _make_local_ms(self):
                return MS(ms_path, device=self.device)
            ri.RxnScribe.get_molscribe = _make_local_ms
            ri.MolDetect.get_molscribe = _make_local_ms
        except Exception:
            pass

    cre_dir = MODEL_DIR / "chemrxnextractor-training-modules"
    if cre_dir.exists():
        try:
            model.init_chemrxnextractor(str(cre_dir))
        except Exception:
            pass


def _run_openchemie(
    model, images_meta: list[dict], doc_id: str
) -> tuple[list[MoleculeCard], list[ReactionEventCard]]:
    pil_images = [m["image"] for m in images_meta]
    if not pil_images:
        return [], []

    mol_cards: list[MoleculeCard] = []
    rxn_cards: list[ReactionEventCard] = []
    mi = 0
    ri = 0

    # Molecules
    try:
        mol_results = model.extract_molecules_from_figures(pil_images, batch_size=2)
    except Exception as exc:
        tqdm.write(f"  molecules_from_figures failed: {exc}", file=sys.stderr)
        mol_results = []

    for fig_idx, fig in enumerate(mol_results or []):
        meta = images_meta[fig_idx] if fig_idx < len(images_meta) else {}
        for m in fig.get("molecules", []) or []:
            smi = m.get("smiles", "")
            raw_score = m.get("score", 0.0)
            conf = min(1.0, max(0.0, float(raw_score) / 3000.0))
            mol_cards.append(MoleculeCard(
                molecule_card_id=f"mol_{doc_id}_{mi:04d}",
                doc_id=doc_id,
                raw_smiles=smi if smi and smi != "<invalid>" else None,
                source_images=[MoleculeImageSource(
                    provenance=SourceProvenance(
                        doc_id=doc_id,
                        source_file=f"{doc_id}.pdf",
                        page=meta.get("page"),
                        bbox=meta.get("bbox"),
                    ),
                    figure_id=meta.get("image_hash", ""),
                    image_id=meta.get("image_hash", ""),
                    bbox=m.get("bbox"),
                    confidence=conf,
                )] if meta else [],
                confidence=conf,
                raw_payload={"mineru_image_hash": meta.get("image_hash", "")},
            ))
            mi += 1

    # Reactions
    try:
        rxn_results = model.extract_reactions_from_figures(pil_images, batch_size=2)
    except Exception as exc:
        tqdm.write(f"  reactions_from_figures failed: {exc}", file=sys.stderr)
        rxn_results = []

    for fig_idx, fig in enumerate(rxn_results or []):
        meta = images_meta[fig_idx] if fig_idx < len(images_meta) else {}
        for r in fig.get("reactions", []) or []:
            reactants = [ReactionParticipant(
                smiles=p.get("smiles", ""), name=p.get("name", ""), role="reactant",
            ) for p in r.get("reactants", [])]
            products = [ReactionParticipant(
                smiles=p.get("smiles", ""), name=p.get("name", ""), role="product",
            ) for p in r.get("products", [])]
            rxn_cards.append(ReactionEventCard(
                reaction_event_id=f"rxn_{doc_id}_{ri:04d}",
                doc_id=doc_id,
                reactants=reactants,
                products=products,
                source=SourceProvenance(
                    doc_id=doc_id,
                    source_file=f"{doc_id}.pdf",
                    page=meta.get("page"),
                    bbox=meta.get("bbox"),
                ) if meta else None,
                confidence=0.5,
                raw_payload={"mineru_image_hash": meta.get("image_hash", "")},
            ))
            ri += 1

    return mol_cards, rxn_cards


# ── Text Entity Extraction ────────────────────────────────────

_TEXT_EXTRACTION_SYSTEM = """\
You are a chemistry entity extraction assistant. Given text from a chemistry \
paper, extract all chemical compounds mentioned.

Return a JSON array. Each element:
{
  "name": "<compound name, label, or abbreviation>",
  "smiles": "<SMILES string if inferable, else null>",
  "aliases": ["<other names for the same compound>"]
}

Rules:
- Extract ALL chemical compounds: solvents, catalysts, reagents, products, substrates.
- Include common abbreviations (DCE, THF, DMF, TEMPO, etc.).
- Include compound labels (1a, 2b, etc.) if they appear.
- For well-known compounds, provide SMILES (e.g. "DCE" → "ClCCCl").
- Do NOT extract general chemical terms (e.g. "acid", "base", "salt") unless tied to a specific compound.
- If no compounds found, return [].
- Return ONLY the JSON array, no markdown fences, no explanation."""


def _extract_molecules_from_text(
    blocks: list[DocumentBlock], doc_id: str
) -> list[MoleculeCard]:
    """Use LLM to extract chemical entities from paragraph blocks."""
    import os
    key = os.environ.get("API_KEY")
    if not key:
        return []

    try:
        from openai import OpenAI  # type: ignore
    except ImportError:
        return []

    # Only process paragraph/section blocks with substantial text
    text_blocks = [
        b for b in blocks
        if b.block_type in ("paragraph", "section", "abstract")
        and b.text and len(b.text) >= 50
    ]
    if not text_blocks:
        return []

    client = OpenAI(api_key=key, base_url=os.environ.get("BASE_URL") or None)
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")

    # Batch paragraphs to reduce API calls (max 5 per call)
    batch_size = 5
    all_molecules: list[MoleculeCard] = []
    mol_idx = 0
    seen_names: set[str] = set()

    for i in range(0, len(text_blocks), batch_size):
        batch = text_blocks[i:i + batch_size]
        batch_text = "\n\n---\n\n".join(
            f"[Block {b.block_id}, page {b.page}]\n{b.text}"
            for b in batch
        )

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _TEXT_EXTRACTION_SYSTEM},
                    {"role": "user", "content": batch_text},
                ],
                temperature=0.0,
                max_tokens=4096,
            )
            raw = (response.choices[0].message.content or "").strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
            entries = json.loads(raw)
        except Exception:
            continue

        if not isinstance(entries, list):
            continue

        for entry in entries:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name") or "").strip()
            if not name or name.lower() in seen_names:
                continue
            seen_names.add(name.lower())

            smiles = entry.get("smiles")
            aliases = [str(a) for a in (entry.get("aliases") or []) if a]

            # Use first block in batch as source
            source_block = batch[0]
            source = SourceProvenance(
                doc_id=doc_id,
                source_file=f"{doc_id}.pdf",
                page=source_block.page,
                bbox=source_block.bbox,
                block_id=source_block.block_id,
                section=source_block.section,
            )

            all_molecules.append(MoleculeCard(
                molecule_card_id=f"mol_txt_{doc_id}_{mol_idx:04d}",
                doc_id=doc_id,
                names=[name],
                raw_smiles=str(smiles) if smiles else None,
                aliases=aliases,
                source_mentions=[MoleculeMention(
                    mention=name,
                    provenance=source,
                    role="text_extraction",
                    confidence=0.7,
                )],
                source_images=[],
                confidence=0.7,
                raw_payload={"source_blocks": [b.block_id for b in batch]},
            ))
            mol_idx += 1

    return all_molecules


def _enrich_molecule_mentions(
    mol_cards: list[MoleculeCard], blocks: list[DocumentBlock]
) -> list[MoleculeCard]:
    """Search block text for molecule names/SMILES and populate source_mentions."""
    result: list[MoleculeCard] = []
    for card in mol_cards:
        # Collect existing mention block_ids to avoid duplicates
        existing_block_ids: set[str] = set()
        for m in card.source_mentions:
            if m.provenance and m.provenance.block_id:
                existing_block_ids.add(m.provenance.block_id)

        # Build search terms from names, aliases, and SMILES
        names = [n for n in card.names if n and len(n) >= 2]
        aliases = [a for a in card.aliases if a and len(a) >= 2]
        smiles = card.canonical_smiles or card.raw_smiles
        search_terms = set(n.lower() for n in names + aliases)
        if smiles and len(smiles) >= 3:
            search_terms.add(smiles.lower())

        if not search_terms:
            result.append(card)
            continue

        new_mentions: list[MoleculeMention] = []
        seen: set[str] = set()
        for block in blocks:
            if block.block_id in existing_block_ids:
                continue
            text = block.text or ""
            if not text:
                continue
            text_lower = text.lower()
            for term in search_terms:
                if term not in text_lower:
                    continue
                idx = text_lower.index(term)
                context = text[max(0, idx - 30):idx + len(term) + 30].strip()
                if context in seen:
                    continue
                seen.add(context)
                new_mentions.append(MoleculeMention(
                    mention=context,
                    provenance=SourceProvenance(
                        doc_id=card.doc_id,
                        source_file=f"{card.doc_id}.pdf",
                        page=block.page,
                        bbox=block.bbox,
                        block_id=block.block_id,
                        section=block.section,
                    ),
                    role="mentioned",
                    confidence=0.8,
                ))
                break  # one mention per block

        if new_mentions:
            merged = list(card.source_mentions) + new_mentions
            card = card.model_copy(update={"source_mentions": merged}, deep=True)
        result.append(card)
    return result


def _normalize_molecules(cards: list[MoleculeCard]) -> list[MoleculeCard]:
    normalizer = RDKitNormalizer()
    if not normalizer.available:
        return cards
    result: list[MoleculeCard] = []
    for card in cards:
        try:
            result.append(normalizer.normalize(card))
        except Exception:
            result.append(card)
    return result


# ── Molecule Deduplication ────────────────────────────────────


def _merge_molecules(
    existing: list[MoleculeCard], new_cards: list[MoleculeCard]
) -> list[MoleculeCard]:
    """Merge new MoleculeCards into existing ones by name or SMILES.

    If a new card's name or SMILES matches an existing card, the new name
    is added as an alias on the existing card. Otherwise the new card is appended.
    """
    # Build lookup: name → index, smiles → index
    name_idx: dict[str, int] = {}
    smiles_idx: dict[str, int] = {}
    for i, card in enumerate(existing):
        for n in card.names:
            if n:
                name_idx[n.lower()] = i
        smi = card.canonical_smiles or card.raw_smiles
        if smi:
            smiles_idx[smi] = i

    for card in new_cards:
        match_idx: int | None = None
        # Match by SMILES
        smi = card.canonical_smiles or card.raw_smiles
        if smi and smi in smiles_idx:
            match_idx = smiles_idx[smi]
        # Match by name
        if match_idx is None:
            for n in card.names:
                if n and n.lower() in name_idx:
                    match_idx = name_idx[n.lower()]
                    break

        if match_idx is not None:
            # Merge: add new name as alias on existing card
            target = existing[match_idx]
            new_aliases = list(target.aliases)
            for n in card.names:
                if n and n not in new_aliases and n not in target.names:
                    new_aliases.append(n)
            for a in card.aliases:
                if a and a not in new_aliases and a not in target.names:
                    new_aliases.append(a)
            existing[match_idx] = target.model_copy(
                update={"aliases": new_aliases}, deep=True
            )
        else:
            # New molecule: append and register in lookup
            idx = len(existing)
            existing.append(card)
            for n in card.names:
                if n:
                    name_idx[n.lower()] = idx
            if smi:
                smiles_idx[smi] = idx

    return existing


# ── Table Reaction Extraction ─────────────────────────────────

_TABLE_EXTRACTION_SYSTEM = """\
You are a chemistry data extraction assistant. Given a table from a chemistry \
paper (as plain text with rows separated by semicolons and columns by pipes), \
extract two things: (1) molecules mentioned, (2) reaction entries per row.

Return a single JSON object with two keys:
{
  "molecules": [
    {
      "name": "<compound label, e.g. '1a', '2b'>",
      "smiles": "<SMILES string if present, else null>",
      "aliases": ["<real chemical name, other names or abbreviations>"]
    }
  ],
  "reactions": [
    {
      "entry": "<entry number or label>",
      "reactants": ["<name or formula>"],
      "products": ["<name or formula>"],
      "solvents": ["<solvent name>"],
      "catalysts": ["<catalyst>"],
      "temperature": "<value with unit or null>",
      "time": "<value with unit or null>",
      "yield_value": "<number or null>",
      "yield_unit": "<% or null>",
      "other_conditions": ["<any other notable conditions>"]
    }
  ]
}

Rules:
- Extract ONLY what is explicitly in the table. Do NOT invent data.
- For molecules: extract ALL unique compounds from the table (substrates, products, solvents, catalysts).
- CRITICAL: For compound labels like "1a", "2b", you MUST put the real chemical name into "aliases". \
Look for it in: (1) the "Substrate" or "Product" column which may contain the real name, \
(2) table footnotes (e.g. "[a] 1a = 1-phenylethanol"), (3) column headers or captions. \
For example: {"name": "1a", "smiles": null, "aliases": ["1-phenylethanol"]}. \
If the real name is truly not in the table, leave aliases empty.
- If a column does not exist in the table, set that field to [] or null.
- For yield, extract the numeric value only (e.g. 85, not "85%").
- If the table is not a reaction table (e.g. characterization data, NMR peaks), return {"molecules": [], "reactions": []}.
- Include footnote text as context for catalysts/conditions if present.
- Return ONLY the JSON object, no markdown fences, no explanation."""


def _extract_from_tables(
    blocks: list[DocumentBlock], doc_id: str
) -> tuple[list[MoleculeCard], list[ReactionEventCard]]:
    """Use LLM to extract MoleculeCards and ReactionEventCards from table blocks."""
    import os
    key = os.environ.get("API_KEY")
    if not key:
        tqdm.write(f"  [table-llm] no API_KEY, skipping", file=sys.stderr)
        return [], []

    try:
        from openai import OpenAI  # type: ignore
    except ImportError:
        tqdm.write(f"  [table-llm] openai not installed, skipping", file=sys.stderr)
        return [], []

    client = OpenAI(api_key=key, base_url=os.environ.get("BASE_URL") or None)
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")

    table_blocks = [b for b in blocks if b.block_type == "table" and b.text]
    if not table_blocks:
        return [], []

    tqdm.write(f"  [table-llm] {doc_id}: extracting from {len(table_blocks)} tables ...", file=sys.stderr)
    all_molecules: list[MoleculeCard] = []
    all_reactions: list[ReactionEventCard] = []
    mol_idx = 0
    rxn_idx = 0
    seen_mol_names: set[str] = set()

    for block in table_blocks:
        user_msg = (
            f"Table from page {block.page}, section: {block.section or 'unknown'}\n\n"
            f"{block.text}"
        )
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _TABLE_EXTRACTION_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=4096,
            )
            raw = (response.choices[0].message.content or "").strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
            result = json.loads(raw)
        except Exception as exc:
            tqdm.write(f"  [table-llm] {doc_id} {block.block_id}: {exc}", file=sys.stderr)
            continue

        if not isinstance(result, dict):
            continue

        source = SourceProvenance(
            doc_id=doc_id,
            source_file=f"{doc_id}.pdf",
            page=block.page,
            bbox=block.bbox,
            block_id=block.block_id,
            section=block.section,
        )

        # Extract molecules
        for mol in (result.get("molecules") or []):
            if not isinstance(mol, dict):
                continue
            name = str(mol.get("name") or "").strip()
            if not name or name.lower() in seen_mol_names:
                continue
            seen_mol_names.add(name.lower())
            smiles = mol.get("smiles")
            aliases = [str(a) for a in (mol.get("aliases") or []) if a]
            all_molecules.append(MoleculeCard(
                molecule_card_id=f"mol_tbl_{doc_id}_{mol_idx:04d}",
                doc_id=doc_id,
                names=[name],
                raw_smiles=str(smiles) if smiles else None,
                aliases=aliases,
                source_mentions=[MoleculeMention(
                    mention=name,
                    provenance=source,
                    role="table_extraction",
                    confidence=0.7,
                )],
                source_images=[],
                confidence=0.7,
                raw_payload={"table_block_id": block.block_id},
            ))
            mol_idx += 1

        # Extract reactions
        for entry in (result.get("reactions") or []):
            if not isinstance(entry, dict):
                continue
            reactants = [
                ReactionParticipant(name=str(r), role="reactant")
                for r in (entry.get("reactants") or [])
                if r
            ]
            products = [
                ReactionParticipant(name=str(p), role="product")
                for p in (entry.get("products") or [])
                if p
            ]
            solvents = [str(s) for s in (entry.get("solvents") or []) if s]
            catalysts = [str(c) for c in (entry.get("catalysts") or []) if c]
            temp = entry.get("temperature")
            time_val = entry.get("time")
            yield_raw = entry.get("yield_value")
            yield_unit = entry.get("yield_unit") or "%"

            yield_obj = None
            if yield_raw is not None:
                try:
                    yv = float(yield_raw)
                    nv = yv / 100 if yield_unit == "%" else yv
                    yield_obj = YieldValue(
                        value=yv, unit=yield_unit,
                        normalized_value=nv, raw_text=str(yield_raw),
                    )
                except (ValueError, TypeError):
                    pass

            all_reactions.append(ReactionEventCard(
                reaction_event_id=f"rxn_tbl_{doc_id}_{rxn_idx:04d}",
                doc_id=doc_id,
                reactants=reactants,
                products=products,
                reagents=[],
                catalysts=catalysts,
                solvents=solvents,
                temperature=str(temp) if temp else None,
                time=str(time_val) if time_val else None,
                yield_value=yield_obj,
                source=source,
                confidence=0.7,
                raw_payload={"table_block_id": block.block_id, "entry": entry},
            ))
            rxn_idx += 1

    return all_molecules, all_reactions


# ── Main ──────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pdf-dir", default=str(PROJECT_ROOT / "data" / "pdfs"),
        help="Directory containing PDF files.",
    )
    parser.add_argument(
        "--doc-id", default=None,
        help="Process a single PDF (by filename stem, e.g. '2').",
    )
    parser.add_argument(
        "--start-doc-id", default="1",
        help="Start processing from this doc id (default: '1').",
    )
    parser.add_argument(
        "--end-doc-id", default="-1",
        help="Stop processing at this doc id (default: '-1' = last). Use '10' to stop at doc 10.",
    )
    parser.add_argument(
        "--skip-mineru", action="store_true",
        help="Skip MinerU API calls — use cached data/mineru_output/.",
    )
    parser.add_argument(
        "--skip-openchemie", action="store_true",
        help="Skip OpenChemIE processing (blocks only).",
    )
    parser.add_argument(
        "--skip-table-extraction", action="store_true",
        help="Skip LLM table extraction.",
    )
    parser.add_argument(
        "--skip-text-extraction", action="store_true",
        help="Skip LLM text extraction.",
    )
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir)
    store = LocalStore(base_dir=PROJECT_ROOT)

    # ── Collect PDFs ──
    if args.doc_id:
        pdf_paths = [pdf_dir / f"{args.doc_id}.pdf"]
        pdf_paths = [p for p in pdf_paths if p.exists()]
        if not pdf_paths:
            print(f"PDF not found: {args.doc_id}.pdf", file=sys.stderr)
            return 1
    else:
        pdf_paths = sorted(p for p in pdf_dir.glob("*.pdf") if "_part" not in p.stem)

    if not pdf_paths:
        print("No PDFs found.", file=sys.stderr)
        return 1

    # ── Step A: MinerU ──
    if not args.skip_mineru:
        missing = [p for p in pdf_paths
                   if not (MINERU_ROOT / p.stem / "extracted").is_dir()]
        if missing:
            print(f"{len(missing)} PDFs need MinerU parsing")
            _mineru_upload_and_poll(missing)
        else:
            print("All PDFs already have cached MinerU output — skipping API calls.")
    else:
        print("--skip-mineru: using cached MinerU output.")

    # ── Step B: Dual parse ──
    def _numeric_key(name: str) -> int:
        """Extract leading numeric part for sorting and range comparison."""
        num = ""
        for ch in name:
            if ch.isdigit():
                num += ch
            else:
                break
        return int(num) if num else 0

    doc_dirs = sorted(
        (d for d in MINERU_ROOT.iterdir()
         if d.is_dir() and (d / "extracted").is_dir()
         and (args.doc_id is None or d.name == args.doc_id)),
        key=lambda d: _numeric_key(d.name),
    )

    # Filter by start/end doc id range
    if not args.doc_id and (args.start_doc_id != "1" or args.end_doc_id != "-1"):
        start = _numeric_key(args.start_doc_id)
        end = _numeric_key(args.end_doc_id) if args.end_doc_id != "-1" else float("inf")
        doc_dirs = [
            d for d in doc_dirs
            if start <= _numeric_key(d.name) <= end
        ]
    if not doc_dirs:
        print("No MinerU output found. Run without --skip-mineru first.", file=sys.stderr)
        return 1

    # Load OpenChemIE model once
    oc_model = None
    if not args.skip_openchemie:
        from openchemie import OpenChemIE
        print("Loading OpenChemIE models (once)...")
        oc_model = OpenChemIE(device="cpu")
        _init_local_models(oc_model)

    ok = 0
    for doc_dir in tqdm(doc_dirs, desc="Dual parsing", unit="doc"):
        doc_id = doc_dir.name
        extracted = doc_dir / "extracted"
        cl_files = list(extracted.glob("*_content_list_v2.json"))
        if not cl_files:
            tqdm.write(f"  SKIP {doc_id}: no content_list_v2.json", file=sys.stderr)
            continue

        pages = json.loads(cl_files[0].read_text("utf-8"))

        # Blocks
        blocks = _build_blocks(doc_id, pages)
        store.save_blocks(doc_id, blocks)

        # OpenChemIE — only overwrite molecules/reactions when model is active
        mol_cards: list[MoleculeCard] = []
        rxn_cards: list[ReactionEventCard] = []
        if oc_model is not None:
            images_dir = extracted / "images"
            images_meta = _load_images(images_dir, pages)
            if images_meta:
                mol_cards, rxn_cards = _run_openchemie(oc_model, images_meta, doc_id)

        # LLM table extraction
        if not args.skip_table_extraction:
            tbl_mols, tbl_rxns = _extract_from_tables(blocks, doc_id)
            if tbl_mols:
                mol_cards = _merge_molecules(mol_cards, tbl_mols)
            if tbl_rxns:
                rxn_cards.extend(tbl_rxns)

        # LLM text extraction — extract new molecules from paragraph text
        if not args.skip_text_extraction:
            txt_mols = _extract_molecules_from_text(blocks, doc_id)
            if txt_mols:
                mol_cards = _merge_molecules(mol_cards, txt_mols)

        # Merge with existing data (load → merge → save)
        try:
            existing_mols = store.load_molecules(doc_id)
        except (FileNotFoundError, TypeError):
            existing_mols = []
        if mol_cards:
            if existing_mols:
                mol_cards = _merge_molecules(existing_mols, mol_cards)
            mol_cards = _enrich_molecule_mentions(mol_cards, blocks)
            mol_cards = _normalize_molecules(mol_cards)
            store.save_molecules(doc_id, mol_cards)

        try:
            existing_rxns = store.load_reactions(doc_id)
        except (FileNotFoundError, TypeError):
            existing_rxns = []
        if rxn_cards:
            if existing_rxns:
                # Deduplicate reactions by event_id
                existing_ids = {r.reaction_event_id for r in existing_rxns}
                new_rxns = [r for r in rxn_cards if r.reaction_event_id not in existing_ids]
                rxn_cards = existing_rxns + new_rxns
            store.save_reactions(doc_id, rxn_cards)

        tqdm.write(
            f"  {doc_id}: {len(blocks)} blocks, "
            f"{len(mol_cards)} molecules, {len(rxn_cards)} reactions"
        )
        ok += 1

    print(f"\nDone: {ok}/{len(doc_dirs)} documents")
    return 0 if ok == len(doc_dirs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
