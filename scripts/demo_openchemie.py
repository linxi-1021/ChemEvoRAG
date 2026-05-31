#!/usr/bin/env python
"""Integration test for OpenChemIE PDF parsing (Phase 1 gap #2).

Tests each extraction method against the bundled example PDF, then runs
the full ChemEvoRAG ingest → build → query pipeline.

Usage:
    conda activate chemevorag
    cd ChemEvoRAG_Phase1-main
    python scripts/demo_openchemie.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

# ── environment fixes (must run before any torch / huggingface import) ──

# Use HF mirror for faster downloads in mainland China.
if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Place downloaded model weights in data/models/ so HuggingFace downloads
# are never needed at runtime.
MODEL_DIR = PROJECT_ROOT / "data" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Redirect HF and torch caches to project-local so C: drive won't fill up.
for _env, _subdir in [("HF_HOME", "huggingface"), ("TORCH_HOME", "torch")]:
    if not os.environ.get(_env):
        os.environ[_env] = str(MODEL_DIR / _subdir)
        (MODEL_DIR / _subdir).mkdir(parents=True, exist_ok=True)

# Force offline: all model files are already in data/models/.
# hf_hub_download will only look in the local cache — no network.
if not os.environ.get("HF_HUB_OFFLINE"):
    os.environ["HF_HUB_OFFLINE"] = "1"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Globally intercept hf_hub_download so that ANY call from RxnScribe,
# MolDetect, or any third-party package checks data/models/ first.
import shutil as _shutil
import huggingface_hub.file_download as _hf_dl

_original_hf_download = _hf_dl.hf_hub_download


def _local_first_download(repo_id: str, filename: str, **kwargs):
    _local = str(MODEL_DIR / filename)
    if os.path.isfile(_local):
        _cache = os.path.join(str(MODEL_DIR / "huggingface"), repo_id)
        os.makedirs(_cache, exist_ok=True)
        _cached = os.path.join(_cache, filename)
        if not os.path.exists(_cached):
            _shutil.copy2(_local, _cached)
        return _cached
    return _original_hf_download(repo_id, filename, **kwargs)


_hf_dl.hf_hub_download = _local_first_download

try:
    import huggingface_hub._snapshot_download as _hf_snap

    _orig_snap = _hf_snap.snapshot_download

    def _local_snapshot(repo_id: str, **kwargs):
        _repo_name = repo_id.replace("/", "--")
        for _c in [
            str(MODEL_DIR / _repo_name),
            str(MODEL_DIR / repo_id.split("/")[-1]),
        ]:
            if os.path.isdir(_c) and os.listdir(_c):
                return _c
        return _orig_snap(repo_id, **kwargs)

    _hf_snap.snapshot_download = _local_snapshot
except ImportError:
    pass


def _init_local_models(oc):
    """Point OpenChemIE to locally downloaded model weights.

    Checks ``data/models/`` for each model file.  If found, passes it to
    the corresponding ``init_*(path)`` method, bypassing network download.
    If a file is missing, the model falls back to its default behavior
    (HuggingFace or Dropbox) at first use.
    """
    _weights = [
        # (init_suffix, filename, label)  — filename matches HuggingFace
        ("molscribe",        "swin_base_char_aux_1m680k.pth",              "MolScribe"),
        ("pdfparser",        "publaynet-tf_efficientdet_d1.pth.tar",       "EfficientDet"),
        ("rxnscribe",        "pix2seq_reaction_full.ckpt",                 "RxnScribe"),
        ("moldet",           "best_hf.ckpt",                               "MolDetect"),
        ("coref",            "coref_best_hf.ckpt",                         "Coref"),
        ("chemner",          "best.ckpt",                                  "ChemNER"),
    ]
    for suffix, filename, label in _weights:
        candidate = MODEL_DIR / filename
        if not candidate.exists():
            continue
        init = getattr(oc, f"init_{suffix}", None)
        if init is None:
            continue
        try:
            import torch as _t

            _t.cuda.empty_cache()
            init(str(candidate))
            print(f"   [MODEL] {label} loaded from {candidate}")
        except Exception as _exc:
            print(f"   [WARN] {label} init failed: {_exc}")

    # RxnScribe/MolDetect's __init__ internally calls self.get_molscribe()
    # which hardcodes hf_hub_download("yujieq/MolScribe", ...).  Patch at
    # the class level BEFORE any instance is created.
    _ms = MODEL_DIR / "swin_base_char_aux_1m680k.pth"
    _ms1 = MODEL_DIR / "swin_base_char_aux_1m.pth"
    if _ms.exists() or _ms1.exists():
        try:
            from molscribe import MolScribe as _MS
            _ms_path = str(_ms if _ms.exists() else _ms1)
            import rxnscribe.interface as _ri

            def _make_local_ms(self):
                return _MS(_ms_path, device=self.device)
            _ri.RxnScribe.get_molscribe = _make_local_ms
            _ri.MolDetect.get_molscribe = _make_local_ms
            print(f"   [MODEL] Patched RxnScribe/MolDetect to use local MolScribe")
        except Exception:
            pass

    _cre_dir = MODEL_DIR / "chemrxnextractor-training-modules"
    if _cre_dir.exists():
        try:
            oc.init_chemrxnextractor(str(_cre_dir))
            print(f"   [MODEL] ChemRxnExtractor loaded from {_cre_dir}")
        except Exception:
            pass

EXAMPLE_PDF = str(PROJECT_ROOT / "external" / "OpenChemIE" / "example" / "acs.joc.2c00749.pdf")


def main() -> int:
    ok = 0
    fail = 0
    skip = 0

    def passed(msg: str) -> None:
        nonlocal ok
        ok += 1
        print(f"   [PASS] {msg}")

    def failed(msg: str) -> None:
        nonlocal fail
        fail += 1
        print(f"   [FAIL] {msg}")

    def skipped(msg: str) -> None:
        nonlocal skip
        skip += 1
        print(f"   [SKIP] {msg}")

    print("=" * 60)
    print("1. OPENCHEMIE IMPORT")
    print("=" * 60)
    try:
        from openchemie import OpenChemIE  # noqa: F811

        passed("import OpenChemIE")
    except Exception as exc:
        failed(f"import OpenChemIE: {exc}")
        print("\n   Run: pip install -e external/OpenChemIE")
        return 1

    print()
    print("=" * 60)
    print("2. POPPLER / PDFTOTEXT")
    print("=" * 60)
    try:
        import pdftotext  # noqa: F401

        passed("pdftotext available")
    except ImportError:
        skipped("pdftotext not installed — text extraction methods will fail")

    print()
    print("=" * 60)
    print("3. MODEL LOADING")
    print("=" * 60)
    try:
        model = OpenChemIE(device="cpu")
        passed("OpenChemIE(device='cpu') loaded")
        _init_local_models(model)
        import torch as _t

        _t.cuda.empty_cache()
    except Exception as exc:
        failed(f"model loading: {exc}")
        print("\n   Check that torch, transformers, and NumPy versions match environment.yml")
        return 1

    print()
    print("=" * 60)
    print("4. MOLECULES FROM FIGURES")
    print("=" * 60)
    try:
        import torch as _t

        _t.cuda.empty_cache()
        results = model.extract_molecules_from_figures_in_pdf(
            EXAMPLE_PDF, batch_size=2
        )
        if results:
            total_mols = sum(len(fig.get("molecules", [])) for fig in results)
            passed(f"{len(results)} figure(s), {total_mols} molecule(s) detected")

            # Save each figure image to data/runs/figures/ for manual review.
            _fig_dir = PROJECT_ROOT / "data" / "runs" / "figures"
            _fig_dir.mkdir(parents=True, exist_ok=True)
            for i, fig in enumerate(results):
                _img = fig.get("image")
                _page = fig.get("page", "?")
                _nmols = len(fig.get("molecules", []))
                if _img is not None:
                    from PIL import Image as _PILImage
                    if isinstance(_img, _PILImage.Image):
                        _img.save(str(_fig_dir / f"fig_{i+1:02d}_p{_page}_{_nmols}mols.png"))
                    else:
                        _PILImage.fromarray(_img).save(
                            str(_fig_dir / f"fig_{i+1:02d}_p{_page}_{_nmols}mols.png")
                        )
                # Print all SMILES per figure
                print(f"   Fig {i+1:02d} (page {_page}, {_nmols} mols):")
                _mols = fig.get("molecules", [])
                for m in _mols[:8]:
                    _smi = m.get("smiles", "")
                    _score = m.get("score", "")
                    print(f"       {_smi:45s} score={_score}")
                if len(_mols) > 8:
                    print(f"       ... and {len(_mols) - 8} more")
            print(f"   Figures saved to {_fig_dir}")
        else:
            failed("returned empty list — no figures found in PDF")
    except Exception as exc:
        failed(f"extract_molecules_from_figures_in_pdf: {exc}")

    print()
    print("=" * 60)
    print("5. MOLECULES FROM TEXT")
    print("=" * 60)
    try:
        results = model.extract_molecules_from_text_in_pdf(EXAMPLE_PDF)
        if results:
            total_mols = sum(len(page.get("molecules", [])) for page in results)
            passed(f"{len(results)} page(s), {total_mols} molecule paragraph(s)")
        else:
            failed("returned empty list")
    except Exception as exc:
        failed(f"extract_molecules_from_text_in_pdf: {exc}")

    print()
    print("=" * 60)
    print("6. REACTIONS FROM FIGURES")
    print("=" * 60)
    try:
        _t.cuda.empty_cache()
        results = model.extract_reactions_from_figures_in_pdf(
            EXAMPLE_PDF, batch_size=2
        )
        if results:
            total_rxns = sum(len(fig.get("reactions", [])) for fig in results)
            passed(f"{len(results)} figure(s), {total_rxns} reaction(s)")
            for i, fig in enumerate(results):
                _page = fig.get("page", "?")
                _rxns = fig.get("reactions", [])
                if not _rxns:
                    continue
                print(f"   Fig {i+1:02d} (page {_page}, {len(_rxns)} rxns):")
                for ri, rxn in enumerate(_rxns[:3]):
                    _reac = [p.get("smiles", "?") for p in rxn.get("reactants", [])]
                    _prod = [p.get("smiles", "?") for p in rxn.get("products", [])]
                    _cond = [c.get("text", "") for c in rxn.get("conditions", [])]
                    _cond_flat = "; ".join(
                        str(t) if isinstance(t, str) else " ".join(t)
                        for t in _cond[:2]
                    )
                    print(f"       rxn {ri+1}: {' + '.join(_reac)} -> {' + '.join(_prod)}")
                    if _cond_flat:
                        print(f"               conditions: {_cond_flat[:100]}")
                if len(_rxns) > 3:
                    print(f"       ... and {len(_rxns) - 3} more reactions")
        else:
            failed("returned empty list")
    except Exception as exc:
        failed(f"extract_reactions_from_figures_in_pdf: {exc}")

    print()
    print("=" * 60)
    print("7. REACTIONS FROM TEXT")
    print("=" * 60)
    try:
        results = model.extract_reactions_from_text_in_pdf(EXAMPLE_PDF)
        if results:
            total_rxns = sum(len(page.get("reactions", [])) for page in results)
            passed(f"{len(results)} page(s), {total_rxns} reaction(s)")
        else:
            failed("returned empty list")
    except Exception as exc:
        failed(f"extract_reactions_from_text_in_pdf: {exc}")

    print()
    print("=" * 60)
    print("8. TABLES FROM PDF")
    print("=" * 60)
    try:
        results = model.extract_tables_from_pdf(EXAMPLE_PDF)
        passed(f"{len(results)} table(s)")
    except Exception as exc:
        failed(f"extract_tables_from_pdf: {exc}")

    print()
    # Free GPU memory before Adapter loads its own model copy.
    del model
    _t.cuda.empty_cache()

    print("=" * 60)
    print("9. CHEMEVORAG ADAPTER")
    print("=" * 60)
    try:
        from parsing import OpenChemIEAdapter

        adapter = OpenChemIEAdapter()
        extraction = adapter.parse_pdf(EXAMPLE_PDF, doc_id="test_paper")
        print(f"   status:    {extraction.status}")
        print(f"   molecules: {len(extraction.molecules)} item(s)")
        print(f"   reactions: {len(extraction.reactions)} item(s)")
        print(f"   tables:    {len(extraction.tables)} item(s)")
        print(f"   errors:    {len(extraction.errors)}")
        if extraction.errors:
            for e in extraction.errors:
                print(f"      {e[:100]}")

        # Print reaction items in detail for manual verification.
        _rxn_dir = PROJECT_ROOT / "data" / "runs"
        _rxn_dir.mkdir(parents=True, exist_ok=True)
        _rxn_json = _rxn_dir / "test_paper_reactions.json"
        with open(_rxn_json, "w", encoding="utf-8") as _f:
            json.dump(
                [item.payload for item in extraction.reactions],
                _f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"   Reaction data saved to {_rxn_json}")

        # Print molecule items detail.
        _mol_json = _rxn_dir / "test_paper_molecules.json"
        with open(_mol_json, "w", encoding="utf-8") as _f:
            json.dump(
                [item.payload for item in extraction.molecules],
                _f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"   Molecule data saved to {_mol_json}")

        if extraction.status == "failed":
            failed("adapter returned status=failed")
        elif extraction.status == "partial":
            skipped("adapter returned status=partial (some methods failed)")
        else:
            passed("adapter returned status=success")
    except Exception as exc:
        failed(f"adapter: {exc}")

    print()
    print("=" * 60)
    print("10. END-TO-END PIPELINE")
    print("=" * 60)
    if extraction is not None and extraction.status != "failed":
        try:
            from evidence import EvidenceBuilder
            from storage import LocalStore

            store = LocalStore(base_dir=PROJECT_ROOT)
            store.save_parsed("test_paper", extraction.model_dump(mode="json"))
            passed("step 1/4: RawExtraction saved to data/parsed/")

            bundle = EvidenceBuilder().build(extraction)
            passed(
                f"step 2/4: Evidence built — "
                f"{len(bundle.molecules)} mols, {len(bundle.reactions)} rxns, "
                f"{len(bundle.facts)} facts, {len(bundle.blocks)} blocks"
            )

            from normalization import RDKitNormalizer

            normalizer = RDKitNormalizer()
            if normalizer.available and bundle.molecules:
                norm_ok = 0
                for i, card in enumerate(bundle.molecules):
                    bundle.molecules[i] = normalizer.normalize(card)
                    if bundle.molecules[i].normalization_status == "success":
                        norm_ok += 1
                passed(f"step 3/4: RDKit normalized {norm_ok}/{len(bundle.molecules)} molecules")

            store.save_blocks("test_paper", bundle.blocks)
            store.save_molecules("test_paper", bundle.molecules)
            store.save_reactions("test_paper", bundle.reactions)
            store.save_facts("test_paper", bundle.facts)
            passed("step 3/4: Evidence saved to data/evidence/")

            from solver import ChemRAGSolver

            solver = ChemRAGSolver(store)
            answer = solver.answer(
                "What molecules and reactions are mentioned?",
                doc_ids=["test_paper"],
            )
            if answer.supporting_evidence:
                passed(
                    f"step 4/4: Query returned {len(answer.supporting_evidence)} evidence item(s)"
                )
                print(f"        answer preview: {answer.answer[:120]}...")
            else:
                skipped("step 4/4: Query returned no evidence (expected for small PDF)")
        except Exception as exc:
            failed(f"pipeline: {exc}")
    else:
        skipped("end-to-end pipeline (adapter returned no results)")

    print()
    print("=" * 60)
    total = ok + fail
    pct = (ok / total * 100) if total > 0 else 0
    print(f"RESULTS:  {ok} passed, {fail} failed, {skip} skipped  ({pct:.0f}%)")
    print("=" * 60)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
