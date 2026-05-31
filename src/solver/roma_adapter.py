"""ROMA solver boundary plus deterministic local answer synthesis."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from evidence import EvidencePackage, GroundedAnswer, SupportingEvidence
from retrieval import RetrievalRouter
from storage import LocalStore


class ROMAAdapter:
    """Optional adapter for the external ROMA project."""

    def __init__(self, roma_path: str | Path = "/root/ROMA") -> None:
        self.roma_path = Path(roma_path)

    def available(self) -> bool:
        return (self.roma_path / "src" / "roma_dspy").exists()

    def solve(self, task: str) -> Any:
        if not self.available():
            raise RuntimeError(f"ROMA source not found at {self.roma_path}")
        src_path = str(self.roma_path / "src")
        inserted = False
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
            inserted = True
        try:
            from roma_dspy.core.engine.solve import solve  # type: ignore

            return solve(task)
        finally:
            if inserted:
                try:
                    sys.path.remove(src_path)
                except ValueError:
                    pass


class ChemRAGSolver:
    """Grounded local solver using retrieval evidence, with optional ROMA boundary."""

    def __init__(
        self,
        store: LocalStore,
        *,
        router: RetrievalRouter | None = None,
        roma_adapter: ROMAAdapter | None = None,
        use_roma: bool = False,
    ) -> None:
        self.store = store
        self.router = router or RetrievalRouter(store)
        self.roma_adapter = roma_adapter or ROMAAdapter()
        self.use_roma = use_roma

    def answer(
        self,
        query: str,
        *,
        doc_ids: list[str] | None = None,
        top_k: int = 8,
    ) -> GroundedAnswer:
        package = self.router.retrieve(query, doc_ids=doc_ids, top_k=top_k)
        return self.answer_from_package(package)

    def answer_from_package(self, package: EvidencePackage) -> GroundedAnswer:
        if not package.candidate_evidence:
            return GroundedAnswer(
                answer="No supported answer could be produced from the available evidence.",
                supporting_evidence=[],
                provenance=[],
                involved_entities=[],
                retrieval_path=package.retrieval_path,
                confidence=0.0,
                uncertainty="No candidate evidence was retrieved for the query.",
                raw_payload={"evidence_package": package.model_dump(mode="json")},
            )

        supporting = [
            SupportingEvidence(
                evidence_id=candidate.evidence_id,
                evidence_type=candidate.evidence_type,
                evidence=candidate.summary or "",
                source=candidate.source,
                confidence=candidate.confidence,
            )
            for candidate in package.candidate_evidence
        ]
        confidence_values = [
            evidence.confidence for evidence in supporting if evidence.confidence is not None
        ]
        confidence = (
            sum(confidence_values) / len(confidence_values)
            if confidence_values
            else 0.5
        )

        return GroundedAnswer(
            answer=_synthesize_answer(package),
            supporting_evidence=supporting,
            provenance=package.provenance,
            involved_entities=[
                candidate.evidence_id
                for candidate in package.candidate_evidence
                if candidate.evidence_type == "molecule"
            ],
            retrieval_path=package.retrieval_path,
            confidence=round(confidence, 3),
            uncertainty=_uncertainty(package),
            raw_payload={"evidence_package": package.model_dump(mode="json")},
        )


def _synthesize_answer(package: EvidencePackage) -> str:
    first = package.candidate_evidence[0]
    if package.intent in {"reaction_comparison", "reaction_condition_query"}:
        return f"Retrieved reaction evidence: {first.summary or first.evidence_id}"
    if package.intent in {"alias_resolution", "entity_lookup"}:
        if first.evidence_type == "molecule":
            return _synthesize_molecule_identity(first)
        return f"Retrieved molecule identity evidence: {first.summary or first.evidence_id}"
    return f"Retrieved supporting evidence: {first.summary or first.evidence_id}"


def _synthesize_molecule_identity(candidate) -> str:
    slots = candidate.structured_slots
    names = slots.get("names") or []
    aliases = slots.get("aliases") or []
    smiles = slots.get("canonical_smiles") or slots.get("raw_smiles")
    inchi_key = slots.get("inchi_key")
    source_id = slots.get("linked_elementkg_id")

    display_name = names[0] if names else candidate.evidence_id
    alias_text = ", ".join(alias for alias in aliases if alias != display_name)
    parts = [f"{display_name} is the best local molecule identity match."]
    if alias_text:
        parts.append(f"Aliases: {alias_text}.")
    if smiles:
        parts.append(f"SMILES: {smiles}.")
    if inchi_key:
        parts.append(f"InChIKey: {inchi_key}.")
    if source_id:
        parts.append(f"Source id: {source_id}.")
    return " ".join(parts)


def _uncertainty(package: EvidencePackage) -> str | None:
    if package.conflicts:
        return "Potential evidence conflicts were detected."
    if package.missing_slots:
        return f"Missing evidence slots: {', '.join(package.missing_slots)}"
    return "Answer is based on local retrieved evidence only; no LLM synthesis was used."
