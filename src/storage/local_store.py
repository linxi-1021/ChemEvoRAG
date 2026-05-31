"""Local JSON storage for parsed outputs and evidence objects."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeVar

from pydantic import TypeAdapter

from evidence import DocumentBlock, FactCard, MoleculeCard, ReactionEventCard


T = TypeVar("T")


class LocalStore:
    """Read and write ChemEvoRAG Phase 1 artifacts as JSON files."""

    def __init__(self, base_dir: str | Path = ".") -> None:
        self.base_dir = Path(base_dir)
        self.parsed_dir = self.base_dir / "data" / "parsed"
        self.evidence_dir = self.base_dir / "data" / "evidence"

    def save_parsed(self, doc_id: str, payload: dict[str, Any]) -> Path:
        return self._write_json(self._parsed_path(doc_id), payload)

    def load_parsed(self, doc_id: str) -> dict[str, Any]:
        payload = self._read_json(self._parsed_path(doc_id))
        if not isinstance(payload, dict):
            raise TypeError(f"Parsed payload for {doc_id!r} must be a JSON object.")
        return payload

    def save_blocks(self, doc_id: str, blocks: list[DocumentBlock]) -> Path:
        return self._save_models(self._evidence_path(doc_id, "blocks"), blocks)

    def load_blocks(self, doc_id: str) -> list[DocumentBlock]:
        return self._load_models(
            self._evidence_path(doc_id, "blocks"), list[DocumentBlock]
        )

    def save_molecules(self, doc_id: str, molecules: list[MoleculeCard]) -> Path:
        return self._save_models(self._evidence_path(doc_id, "molecules"), molecules)

    def load_molecules(self, doc_id: str) -> list[MoleculeCard]:
        return self._load_models(
            self._evidence_path(doc_id, "molecules"), list[MoleculeCard]
        )

    def save_reactions(self, doc_id: str, reactions: list[ReactionEventCard]) -> Path:
        return self._save_models(self._evidence_path(doc_id, "reactions"), reactions)

    def load_reactions(self, doc_id: str) -> list[ReactionEventCard]:
        return self._load_models(
            self._evidence_path(doc_id, "reactions"), list[ReactionEventCard]
        )

    def save_facts(self, doc_id: str, facts: list[FactCard]) -> Path:
        return self._save_models(self._evidence_path(doc_id, "facts"), facts)

    def load_facts(self, doc_id: str) -> list[FactCard]:
        return self._load_models(self._evidence_path(doc_id, "facts"), list[FactCard])

    def _parsed_path(self, doc_id: str) -> Path:
        return self.parsed_dir / f"{doc_id}.raw_extraction.json"

    def _evidence_path(self, doc_id: str, kind: str) -> Path:
        return self.evidence_dir / f"{doc_id}.{kind}.json"

    def _save_models(self, path: Path, models: list[Any]) -> Path:
        payload = [model.model_dump(mode="json") for model in models]
        return self._write_json(path, payload)

    def _load_models(self, path: Path, model_type: Any) -> Any:
        payload = self._read_json(path)
        return TypeAdapter(model_type).validate_python(payload)

    def _write_json(self, path: Path, payload: Any) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        return path

    def _read_json(self, path: Path) -> Any:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

