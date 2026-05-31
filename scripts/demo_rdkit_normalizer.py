#!/usr/bin/env python
"""Interactive demo of the RDKit normalizer with real molecular examples.

Usage:
    conda activate chemevorag
    cd ChemEvoRAG_Phase1-main
    python scripts/demo_rdkit_normalizer.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evidence import MoleculeCard
from normalization import RDKitNormalizer


def main() -> int:
    n = RDKitNormalizer()

    # ── 1. availability ─────────────────────────────────────────────
    print("=" * 60)
    print("1. RDKit AVAILABILITY")
    print("=" * 60)
    print(f"   RDKit installed: {n.available}")
    if not n.available:
        print("   Install: conda install -c conda-forge rdkit")
        return 1
    print()

    # ── 2. canonical SMILES ─────────────────────────────────────────
    print("=" * 60)
    print("2. CANONICAL SMILES (same molecule, different notations)")
    print("=" * 60)
    examples = [
        ("OCC", "ethanol (hydroxyl on C1)"),
        ("CCO", "ethanol (hydroxyl on C2)"),
        ("C(O)C", "ethanol (branched notation)"),
        ("c1ccccc1", "benzene (aromatic lowercase)"),
        ("C1=CC=CC=C1", "benzene (Kekule form)"),
    ]
    for smiles, desc in examples:
        result = n.to_canonical_smiles(smiles)
        valid = n.validate_smiles(smiles)
        print(f"   {smiles:20s} -> {result or 'INVALID':20s}  ({desc})")
    print()

    # ── 3. InChIKey generation ──────────────────────────────────────
    print("=" * 60)
    print("3. InChIKey GENERATION (standard chemical fingerprint)")
    print("=" * 60)
    molecules = [
        ("CCO", "ethanol"),
        ("CC(=O)OC1=CC=CC=C1C(=O)O", "aspirin"),
        ("C1CCC(CC1)NC2CCCCC2", "dicyclohexylamine"),
        ("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "caffeine"),
        ("CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", "ibuprofen"),
    ]
    for smiles, name in molecules:
        inchikey = n.to_inchikey(smiles)
        canonical = n.to_canonical_smiles(smiles)
        changed = " (canonical)" if smiles != canonical else ""
        print(f"   {name:25s}  {smiles:35s} → {canonical or '?':35s}{changed}")
        print(f"   {'':25s}  InChIKey: {inchikey}")
        print()
    print()

    # ── 4. invalid SMILES detection ─────────────────────────────────
    print("=" * 60)
    print("4. INVALID SMILES DETECTION")
    print("=" * 60)
    bad = [
        ("not-a-smiles", "random text"),
        ("C1CC", "unclosed ring"),
        ("Z9Q", "element Z does not exist"),
    ]
    for smiles, desc in bad:
        valid = n.validate_smiles(smiles)
        print(f"   {smiles:20s}  valid={valid}   ({desc})")
    print()

    # ── 5. full MoleculeCard normalization ──────────────────────────
    print("=" * 60)
    print("5. FULL MoleculeCard NORMALIZATION")
    print("=" * 60)
    cards = [
        MoleculeCard(
            molecule_card_id="mol_ethanol",
            doc_id="demo",
            names=["ethanol", "EtOH"],
            raw_smiles="OCC",
        ),
        MoleculeCard(
            molecule_card_id="mol_aspirin",
            doc_id="demo",
            names=["aspirin"],
            raw_smiles="CC(=O)Oc1ccccc1C(=O)O",
        ),
        MoleculeCard(
            molecule_card_id="mol_junk",
            doc_id="demo",
            names=["unknown compound"],
            raw_smiles="X9Z-invalid",
        ),
        MoleculeCard(
            molecule_card_id="mol_no_smiles",
            doc_id="demo",
            names=["mystery substance"],
        ),
    ]
    for card in cards:
        result = n.normalize(card)
        name = result.names[0] if result.names else "N/A"
        print(f"   --- BEFORE ---")
        print(f"       molecule_card_id:  {card.molecule_card_id}")
        print(f"       doc_id:            {card.doc_id}")
        print(f"       names:             {card.names}")
        print(f"       raw_smiles:        {card.raw_smiles or '(none)'}")
        print(f"       canonical_smiles:  {card.canonical_smiles or '(none)'}")
        print(f"       inchi_key:         {card.inchi_key or '(none)'}")
        print(f"       status:            {card.normalization_status}")
        print(f"   --- AFTER ---")
        print(f"       canonical_smiles:  {result.canonical_smiles or '(none)'}")
        print(f"       inchi_key:         {result.inchi_key or '(none)'}")
        print(f"       status:            {result.normalization_status}")
        if result.errors:
            for err in result.errors:
                print(f"       ERROR: {err}")
        assert card.normalization_status == "not_attempted", "Original card was mutated!"
        print(f"       (original card unchanged: OK)")
        print()

    # ── 6. batch processing ─────────────────────────────────────────
    print("=" * 60)
    print("6. BATCH PROCESSING")
    print("=" * 60)
    batch = [
        MoleculeCard(
            molecule_card_id=f"batch_{i:02d}",
            doc_id="demo",
            raw_smiles=smi,
        )
        for i, smi in enumerate(
            ["CCO", "CC(=O)O", "c1ccccc1", "CCN(CC)CC", "C1CCCCC1"]
        )
    ]
    results = n.normalize_batch(batch)
    for r in results:
        print(f"   {r.molecule_card_id}: {r.raw_smiles:12s} -> {r.canonical_smiles:12s}  [{r.normalization_status}]")
    print()

    print("Demo complete: all assertions passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
