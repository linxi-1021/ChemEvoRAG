#!/usr/bin/env python
"""LLM-assisted molecule naming: extract compound names from paper text.

For each document, reads blocks.json, sends full text to LLM asking for
a molecule name table, validates names against source text, then patches
names/aliases/source_mentions into molecules.json.

Cached in data/evidence/{doc_id}.mol_names_llm.json — re-running reuses cache.
"""

from __future__ import annotations

import argparse
import json
import os
import re
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

SYSTEM_PROMPT = """\
Extract ALL chemical compounds from this chemistry paper.
CRITICAL: For EACH compound, include ALL names used in this paper as aliases:
  - Compound numbers/labels: "3a", "1", "6ae"
  - Abbreviations: "DCHA", "PMB", "Hex-n", "n-Hex"
  - Shorthand names: "alcohol 3a", "ketone 2a", "allenol 1"
  - Reagent/common names: "TEMPO", "NaCl", "DCE"

Return ONLY a JSON object (no markdown):
{
  "molecules": [
    {"compound_label": "3a", "name": "1-phenylethanol", "aliases": ["3a", "alcohol 3a", "1-phenylethanol"], "page": 2, "evidence": "verbatim quote"}
  ]
}
Rules: extract EVERY compound. Do NOT invent. Every entry must have evidence from the text."""


def _call_llm(blocks_text: str) -> dict | None:
    key = os.environ.get("API_KEY")
    url = os.environ.get("BASE_URL")
    model = os.environ.get("LLM_MODEL", "gpt-5-mini")
    if not key:
        return None

    from openai import OpenAI
    client = OpenAI(api_key=key, base_url=url or None)

    # Trim to fit context (~30K chars leaves room for response)
    max_chars = 25000
    text = blocks_text[:max_chars]

    try:
        r = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Here is the paper text:\n\n{text}"},
            ],
            temperature=0.1,
            seed=42,
            max_tokens=32768,
            extra_body={},
        )
        raw = r.choices[0].message.content
    except Exception as exc:
        print(f"  LLM call failed: {exc}", file=sys.stderr)
        return None

    if not raw:
        return None
    raw = raw.strip()
    # Remove invalid JSON control characters (0x00-0x1F except \t\n\r)
    raw = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", raw)
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        debug_path = PROJECT_ROOT / "data" / "evidence" / "_debug_llm_raw.txt"
        debug_path.write_text(raw, encoding="utf-8")
        print(f"    JSON parse failed — raw saved to {debug_path}", file=sys.stderr)
        return None


def _validate_names(llm_result: dict, blocks_text: str) -> list[dict]:
    """Keep only entries whose evidence appears in the source text."""
    molecules = llm_result.get("molecules", [])
    if not isinstance(molecules, list):
        return []

    validated: list[dict] = []
    for m in molecules:
        if not isinstance(m, dict):
            continue
        evidence = (m.get("evidence") or "").strip()
        # Fuzzy match: any 30-char substring of evidence must appear in text
        found = False
        for start in range(0, len(evidence) - 29):
            snippet = evidence[start:start + 30]
            if snippet in blocks_text:
                found = True
                break
        if found:
            validated.append(m)
    return validated


def _match_to_molecule_cards(
    mol_cards: list[dict], llm_mols: list[dict]
) -> list[dict]:
    """Match LLM-extracted names to MoleculeCards by page proximity."""
    if not llm_mols:
        return mol_cards

    for card in mol_cards:
        # Get page from source_images
        page: int | None = None
        for si in card.get("source_images", []) or []:
            p = si.get("provenance", {}).get("page") if isinstance(si.get("provenance"), dict) else si.get("page")
            if p:
                page = p
                break

        # Find LLM entries on same page (or nearby)
        candidates = []
        for llm in llm_mols:
            llm_page = llm.get("page", 0)
            try:
                llm_page = int(llm_page)
            except (TypeError, ValueError):
                continue
            if page and abs(llm_page - page) <= 1:
                candidates.append(llm)
        if not candidates:
            candidates = llm_mols[:5]  # fallback: first 5

        # Assign first unmatched name
        assigned = None
        for c in candidates:
            if not c.get("_assigned"):
                assigned = c
                c["_assigned"] = True
                break

        if assigned:
            names = list(card.get("names") or [])
            aliases = list(card.get("aliases") or [])
            label = assigned.get("compound_label", "")
            name = assigned.get("name", "")
            if label and label not in names:
                names.append(label)
            if name and name not in names:
                names.append(name)
            for a in assigned.get("aliases", []) or []:
                if a and a not in aliases and a not in names:
                    aliases.append(a)
            card["names"] = names
            card["aliases"] = aliases

    return mol_cards


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc-id", default=None, help="Process single document.")
    parser.add_argument("--no-cache", action="store_true", help="Ignore LLM cache.")
    args = parser.parse_args()

    ev_dir = PROJECT_ROOT / "data" / "evidence"
    mol_files = sorted(ev_dir.glob("*.molecules.json"))

    if args.doc_id:
        mol_files = [ev_dir / f"{args.doc_id}.molecules.json"]
        mol_files = [f for f in mol_files if f.exists()]
        if not mol_files:
            print(f"No molecules for doc {args.doc_id}", file=sys.stderr)
            return 1

    ok = 0
    for mol_path in mol_files:
        doc_id = mol_path.stem.replace(".molecules", "")
        blocks_path = ev_dir / f"{doc_id}.blocks.json"
        if not blocks_path.exists():
            print(f"  SKIP {doc_id}: no blocks.json", file=sys.stderr)
            continue

        # Cache
        cache_path = ev_dir / f"{doc_id}.mol_names_llm.json"
        if cache_path.exists() and not args.no_cache:
            llm_mols = json.loads(cache_path.read_text("utf-8"))
            llm_mols = llm_mols if isinstance(llm_mols, list) else []
            print(f"  {doc_id}: using cached ({len(llm_mols)} LLM entries)")
        else:
            blocks = json.loads(blocks_path.read_text("utf-8"))
            # Build text with block_id markers
            lines: list[str] = []
            for b in blocks:
                t = b.get("text", "")
                p = b.get("page", "?")
                bt = b.get("block_type", "paragraph")
                bid = b.get("block_id", "")
                if t:
                    lines.append(f"[{bt} p{p} {bid}] {t}")
            text = "\n\n".join(lines)

            print(f"  {doc_id}: {len(blocks)} blocks, {len(text)} chars → LLM ...")
            llm_result = _call_llm(text)
            if llm_result is None:
                print(f"    LLM returned nothing — skipping", file=sys.stderr)
                continue

            llm_mols = _validate_names(llm_result, text)
            cache_path.write_text(
                json.dumps(llm_mols, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"    {len(llm_mols)} validated names → {cache_path}")

        # Patch molecules.json
        mols = json.loads(mol_path.read_text("utf-8"))
        mols = _match_to_molecule_cards(mols, llm_mols)
        mol_path.write_text(json.dumps(mols, ensure_ascii=False, indent=2), encoding="utf-8")

        # Count improvements
        with_names = sum(1 for m in mols if m.get("names"))
        print(f"    {with_names}/{len(mols)} molecules now have names")
        ok += 1

    print(f"\nDone: {ok} documents")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
