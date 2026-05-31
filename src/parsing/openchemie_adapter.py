"""OpenChemIE parser adapter.

The adapter imports OpenChemIE lazily so the rest of the ingestion pipeline can
run even when model dependencies or weights are not installed yet.

Model weights are loaded from ``data/models/`` if present; otherwise they fall
back to HuggingFace download at runtime.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

# Redirect model caches to project-local directory so downloads
# never fill up the system drive.
_MODEL_DIR = str(Path(__file__).resolve().parents[2] / "data" / "models")
_HF_HOME = os.path.join(_MODEL_DIR, "huggingface")
_TORCH_HOME = os.path.join(_MODEL_DIR, "torch")
if not os.environ.get("HF_HOME"):
    os.environ["HF_HOME"] = _HF_HOME
if not os.environ.get("TORCH_HOME"):
    os.environ["TORCH_HOME"] = _TORCH_HOME
os.makedirs(_HF_HOME, exist_ok=True)
os.makedirs(_TORCH_HOME, exist_ok=True)

# Globally intercept hf_hub_download so that ANY call — from RxnScribe,
# MolDetect, ChemNER, or any third-party package — checks data/models/
# first.  If the file exists locally it is returned immediately without
# touching the network.
import shutil as _shutil

import huggingface_hub.file_download as _hf_dl

_original_hf_download = _hf_dl.hf_hub_download


def _patched_hf_download(
    repo_id: str, filename: str, *, cache_dir: str | None = None, **kwargs: Any
) -> str:
    local = os.path.join(_MODEL_DIR, filename)
    if os.path.isfile(local):
        # Copy into the HF cache structure so downstream loaders find it.
        _hf_cache = os.path.join(cache_dir or _HF_HOME, repo_id)
        os.makedirs(_hf_cache, exist_ok=True)
        _cached = os.path.join(_hf_cache, filename)
        if not os.path.exists(_cached):
            _shutil.copy2(local, _cached)
        return _cached
    # Snapshot downloads (directories) — check local model dir.
    # hf_hub_download is only for single files; snapshot_download is
    # handled separately below.
    return _original_hf_download(
        repo_id, filename, cache_dir=cache_dir, **kwargs
    )


_hf_dl.hf_hub_download = _patched_hf_download

# snapshot_download — used by ChemRxnExtractor to download a whole repo.
try:
    import huggingface_hub._snapshot_download as _hf_snap

    _original_snapshot = _hf_snap.snapshot_download

    def _patched_snapshot(repo_id: str, **kwargs: Any) -> str:
        _repo_name = repo_id.replace("/", "--")
        for _c in [
            os.path.join(_MODEL_DIR, _repo_name),
            os.path.join(_MODEL_DIR, repo_id.split("/")[-1]),
        ]:
            if os.path.isdir(_c) and os.listdir(_c):
                return _c
        return _original_snapshot(repo_id, **kwargs)

    _hf_snap.snapshot_download = _patched_snapshot
except ImportError:
    pass

from evidence import RawExtraction, RawExtractionItem, SourceProvenance


class OpenChemIEAdapter:
    """Wrap OpenChemIE PDF extraction behind ChemEvoRAG's RawExtraction schema."""

    parser_name = "openchemie"

    def __init__(
        self,
        external_path: str | Path | None = None,
        device: str | None = "cpu",
        model_dir: str | Path | None = None,
        enabled_methods: list[str] | None = None,
    ) -> None:
        self.external_path = Path(external_path) if external_path else _external_openchemie_path()
        self.device = device
        self.model_dir = Path(model_dir) if model_dir else _default_model_dir()
        self.enabled_methods = enabled_methods or [
            "extract_molecules_from_figures_in_pdf",
            "extract_molecules_from_text_in_pdf",
            "extract_reactions_from_figures_in_pdf",
            "extract_reactions_from_text_in_pdf",
            "extract_tables_from_pdf",
        ]

    def parse_pdf(self, pdf_path: str | Path, doc_id: str) -> RawExtraction:
        pdf = Path(pdf_path)
        try:
            model = self._build_model()
        except Exception as exc:
            return RawExtraction(
                doc_id=doc_id,
                source_file=str(pdf),
                parser_name=self.parser_name,
                status="failed",
                errors=[f"OpenChemIE initialization failed: {exc}"],
                raw_payload={"external_path": str(self.external_path)},
            )

        raw_payload: dict[str, Any] = {}
        errors: list[str] = []

        for method_name in self.enabled_methods:
            method = getattr(model, method_name, None)
            if method is None:
                errors.append(f"OpenChemIE method not found: {method_name}")
                continue

            try:
                raw_payload[method_name] = method(str(pdf))
            except Exception as exc:
                errors.append(f"{method_name} failed: {exc}")

        return self._to_raw_extraction(
            doc_id=doc_id,
            source_file=str(pdf),
            raw_payload=raw_payload,
            errors=errors,
        )

    def _build_model(self) -> Any:
        if not self.external_path.exists():
            raise FileNotFoundError(f"OpenChemIE path not found: {self.external_path}")

        external_path = str(self.external_path)
        inserted = False
        if external_path not in sys.path:
            sys.path.insert(0, external_path)
            inserted = True

        try:
            from openchemie import OpenChemIE  # type: ignore
        finally:
            if inserted:
                try:
                    sys.path.remove(external_path)
                except ValueError:
                    pass

        model = OpenChemIE(device=self.device)
        self._load_local_weights(model)
        return model

    def _load_local_weights(self, model: Any) -> None:
        """Load model weights from ``self.model_dir`` if present.

        Each model checks ``data/models/<filename>`` first.  If the file
        exists it is passed to the corresponding ``init_*(path)`` method,
        bypassing HuggingFace / Dropbox downloads.  If missing, the default
        download behavior applies at first use.

        ====================  ==========================================  ==========
        Model                  File (in data/models/)                     Phase 1
        ====================  ==========================================  ==========
        MolScribe              swin_base_char_aux_1m680k.pth              required
        EfficientDet           publaynet-tf_efficientdet_d1.pth.tar       required
        RxnScribe              pix2seq_reaction_full.ckpt                 suggested
        MolDetect              MolDetect_best_hf.ckpt                     optional
        Coref                  MolDetect_coref_best_hf.ckpt               optional
        ChemNER                ChemNER_best.ckpt                          optional
        ChemRxnExtractor       chemrxnextractor-training-modules/         optional
        ====================  ==========================================  ==========
        """
        _local_weights: list[tuple[str, str]] = [
            # (init_method_suffix, filename)  — filename matches HuggingFace
            ("molscribe",        "swin_base_char_aux_1m680k.pth"),
            ("rxnscribe",        "pix2seq_reaction_full.ckpt"),
            ("pdfparser",        "publaynet-tf_efficientdet_d1.pth.tar"),
            ("moldet",           "best_hf.ckpt"),
            ("coref",            "coref_best_hf.ckpt"),
            ("chemner",          "best.ckpt"),
        ]

        # RxnScribe and MolDetect internally create MolScribe via
        # hf_hub_download in their __init__.  Patch the class-level
        # get_molscribe BEFORE any instance is created so the local
        # file is used from the start.
        _ms_local = self.model_dir / "swin_base_char_aux_1m.pth"
        _ms_680k = self.model_dir / "swin_base_char_aux_1m680k.pth"
        if _ms_680k.exists() or _ms_local.exists():
            _ms_path = str(_ms_680k if _ms_680k.exists() else _ms_local)
            try:
                from molscribe import MolScribe as _MS

                import rxnscribe.interface as _rxn_iface

                def _make_local_ms(self):
                    return _MS(_ms_path, device=self.device)

                _rxn_iface.RxnScribe.get_molscribe = _make_local_ms
                _rxn_iface.MolDetect.get_molscribe = _make_local_ms
            except Exception:
                pass

        for suffix, filename in _local_weights:
            candidate = self.model_dir / filename
            if not candidate.exists():
                continue
            init = getattr(model, f"init_{suffix}", None)
            if init is None:
                continue
            try:
                init(str(candidate))
            except Exception:
                pass

        # ChemRxnExtractor expects a directory (snapshot), not a single file.
        _cre_dir = self.model_dir / "chemrxnextractor-training-modules"
        if _cre_dir.exists():
            try:
                model.init_chemrxnextractor(str(_cre_dir))
            except Exception:
                pass

    def _to_raw_extraction(
        self,
        *,
        doc_id: str,
        source_file: str,
        raw_payload: dict[str, Any],
        errors: list[str],
    ) -> RawExtraction:
        molecules = self._items_from_method(
            doc_id,
            source_file,
            raw_payload,
            method_name="extract_molecules_from_figures_in_pdf",
            item_type="molecule",
        )
        molecules.extend(
            self._items_from_method(
                doc_id,
                source_file,
                raw_payload,
                method_name="extract_molecules_from_text_in_pdf",
                item_type="molecule",
            )
        )

        reactions = self._items_from_method(
            doc_id,
            source_file,
            raw_payload,
            method_name="extract_reactions_from_figures_in_pdf",
            item_type="reaction",
        )
        reactions.extend(
            self._items_from_method(
                doc_id,
                source_file,
                raw_payload,
                method_name="extract_reactions_from_text_in_pdf",
                item_type="reaction",
            )
        )

        tables = self._items_from_method(
            doc_id,
            source_file,
            raw_payload,
            method_name="extract_tables_from_pdf",
            item_type="table",
        )

        status = "success"
        if errors and raw_payload:
            status = "partial"
        elif errors:
            status = "failed"

        return RawExtraction(
            doc_id=doc_id,
            source_file=source_file,
            parser_name=self.parser_name,
            status=status,
            molecules=molecules,
            reactions=reactions,
            tables=tables,
            raw_payload=_json_safe(raw_payload),
            errors=errors,
        )

    def _items_from_method(
        self,
        doc_id: str,
        source_file: str,
        raw_payload: dict[str, Any],
        *,
        method_name: str,
        item_type: str,
    ) -> list[RawExtractionItem]:
        payload = raw_payload.get(method_name)
        if payload is None:
            return []

        if not isinstance(payload, list):
            payloads = [payload]
        else:
            payloads = payload

        items: list[RawExtractionItem] = []
        for index, item in enumerate(payloads):
            page = item.get("page") if isinstance(item, dict) else None
            items.append(
                RawExtractionItem(
                    item_id=f"{method_name}_{index:04d}",
                    doc_id=doc_id,
                    item_type=item_type,  # type: ignore[arg-type]
                    source=SourceProvenance(
                        doc_id=doc_id,
                        source_file=source_file,
                        page=page if isinstance(page, int) else None,
                    ),
                    payload=_json_safe(item),
                )
            )
        return items


def _external_openchemie_path() -> Path:
    return Path(__file__).resolve().parents[2] / "external" / "OpenChemIE"


def _default_model_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "models"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    return repr(value)

