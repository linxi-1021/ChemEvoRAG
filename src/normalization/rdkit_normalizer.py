"""RDKit molecule normalization — canonical SMILES, InChIKey, and structure validation.

Core normalization module for ChemEvoRAG Phase 1.  Takes a MoleculeCard with
raw or non-canonical SMILES and produces a validated card with canonical SMILES,
InChIKey (standard chemical fingerprint), anda a normalization status.

RDKit is imported lazily so the rest of the pipeline can function when it is
not installed.  Call ``RDKitNormalizer.available`` before invoking normalize
to decide whether to fall back to alias-only matching.

Usage::

    normalizer = RDKitNormalizer()
    if normalizer.available:
        card = normalizer.normalize(card)
    cards = normalizer.normalize_batch(cards)
"""

from __future__ import annotations

from typing import Any

from evidence import MoleculeCard


class RDKitNormalizer:
    """Generate canonical SMILES and InChIKey, validate SMILES, and detect
    structural issues for a ``MoleculeCard``."""

    def __init__(self) -> None:
        self._rdkit: Any = None
        self._import_error: str | None = None

    @property
    def available(self) -> bool:
        """True when RDKit was imported successfully."""
        if self._rdkit is None and self._import_error is None:
            try:
                from rdkit import Chem  # noqa: F401
                from rdkit.Chem import inchi  # noqa: F401

                self._rdkit = (Chem, inchi)
            except ModuleNotFoundError as exc:
                self._import_error = (
                    "RDKit is not installed. Install with: conda install -c conda-forge rdkit"
                )
        return self._rdkit is not None and self._import_error is None

    def normalize(self, card: MoleculeCard) -> MoleculeCard:
        """Return a new ``MoleculeCard`` with normalization fields populated.

        The input card is never mutated — a copy is returned.
        """

        if not self.available:
            return _normalization_failed(
                card,
                self._import_error or "RDKit unavailable",
            )

        rdkit_chem, rdkit_inchi = self._rdkit
        smiles_source = card.canonical_smiles or card.raw_smiles

        if not smiles_source or not isinstance(smiles_source, str):
            return _normalization_failed(
                card,
                "No SMILES available for RDKit normalization.",
            )

        # ── validate ────────────────────────────────────────────────
        mol = self._mol_from_smiles(smiles_source, rdkit_chem)
        if mol is None:
            return _normalization_failed(
                card,
                f"RDKit could not parse SMILES: {smiles_source}",
            )

        # ── canonical SMILES ─────────────────────────────────────────
        try:
            canonical = rdkit_chem.MolToSmiles(mol, canonical=True)
        except Exception as exc:
            return _normalization_failed(
                card,
                f"Canonical SMILES generation failed: {exc}",
            )

        # ── InChIKey ─────────────────────────────────────────────────
        try:
            inchi_key = rdkit_inchi.MolToInchiKey(mol)
        except Exception as exc:
            inchi_key = None
            status = "partial"
            errors = [f"InChIKey generation failed: {exc}"]
        else:
            status = "success"
            errors = []

        updates: dict[str, Any] = {
            "canonical_smiles": canonical,
            "inchi_key": inchi_key or card.inchi_key,
            "normalization_status": status,
            "errors": card.errors + errors,
        }

        # Keep the original raw_smiles if we didn't have a canonical_smiles yet
        if not card.canonical_smiles:
            updates["raw_smiles"] = card.raw_smiles or smiles_source

        return card.model_copy(update=updates, deep=True)

    def normalize_batch(self, cards: list[MoleculeCard]) -> list[MoleculeCard]:
        return [self.normalize(card) for card in cards]

    def validate_smiles(self, smiles: str) -> bool:
        """Return True when *smiles* is a syntactically valid SMILES string."""
        if not self.available:
            return True  # cannot judge — let it through
        mol = self._mol_from_smiles(smiles, self._rdkit[0])
        return mol is not None

    def to_canonical_smiles(self, smiles: str) -> str | None:
        """Convert *smiles* to canonical form, or return None."""
        if not self.available:
            return None
        mol = self._mol_from_smiles(smiles, self._rdkit[0])
        if mol is None:
            return None
        try:
            return self._rdkit[0].MolToSmiles(mol, canonical=True)
        except Exception:
            return None

    def to_inchikey(self, smiles: str) -> str | None:
        """Generate InChIKey from *smiles*, or return None."""
        if not self.available:
            return None
        mol = self._mol_from_smiles(smiles, self._rdkit[0])
        if mol is None:
            return None
        try:
            return self._rdkit[1].MolToInchiKey(mol)
        except Exception:
            return None

    @staticmethod
    def _mol_from_smiles(smiles: str, rdkit_chem: Any) -> Any:
        if not smiles or not isinstance(smiles, str) or not smiles.strip():
            return None
        try:
            mol = rdkit_chem.MolFromSmiles(smiles)
        except Exception:
            return None
        if mol is None:
            return None
        # Detect explicit hydrogen issues early
        try:
            rdkit_chem.SanitizeMol(mol)
        except Exception:
            pass
        return mol


# ── helpers ────────────────────────────────────────────────────────────────


def _normalization_failed(card: MoleculeCard, reason: str) -> MoleculeCard:
    errors = list(card.errors) + [reason]
    return card.model_copy(
        update={
            "normalization_status": "failed",
            "errors": errors,
        },
        deep=True,
    )
