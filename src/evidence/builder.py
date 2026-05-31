"""Build first-pass evidence objects from parser raw extraction output."""

from __future__ import annotations

import re
from typing import Any

from pydantic import TypeAdapter

from .schemas import (
    ChemEvoBaseModel,
    DocumentBlock,
    FactCard,
    MoleculeCard,
    RawExtraction,
    RawExtractionItem,
    ReactionEventCard,
    ReactionParticipant,
    SourceProvenance,
    YieldValue,
)


class EvidenceBundle(ChemEvoBaseModel):
    """Evidence objects generated for one document."""

    doc_id: str
    blocks: list[DocumentBlock]
    molecules: list[MoleculeCard]
    reactions: list[ReactionEventCard]
    facts: list[FactCard]
    errors: list[str] = []


class EvidenceBuilder:
    """Convert RawExtraction into evidence-level representations."""

    def build(self, extraction: RawExtraction | dict[str, Any]) -> EvidenceBundle:
        raw = self._coerce_extraction(extraction)
        blocks = self._build_blocks(raw)
        molecules = self._build_molecules(raw)
        reactions = self._build_reactions(raw)
        facts = self._build_facts(raw, blocks, reactions)
        return EvidenceBundle(
            doc_id=raw.doc_id,
            blocks=blocks,
            molecules=molecules,
            reactions=reactions,
            facts=facts,
            errors=list(raw.errors),
        )

    def _coerce_extraction(self, extraction: RawExtraction | dict[str, Any]) -> RawExtraction:
        if isinstance(extraction, RawExtraction):
            return extraction
        return TypeAdapter(RawExtraction).validate_python(extraction)

    def _build_blocks(self, extraction: RawExtraction) -> list[DocumentBlock]:
        items = [*extraction.blocks, *extraction.tables, *extraction.figures]
        blocks = [
            self._block_from_item(extraction, item, index)
            for index, item in enumerate(items)
        ]
        for index, block in enumerate(blocks):
            prev_id = blocks[index - 1].block_id if index > 0 else None
            next_id = blocks[index + 1].block_id if index + 1 < len(blocks) else None
            blocks[index] = block.model_copy(
                update={"prev_block_id": prev_id, "next_block_id": next_id}
            )
        return blocks

    def _block_from_item(
        self, extraction: RawExtraction, item: RawExtractionItem, index: int
    ) -> DocumentBlock:
        source = item.source
        block_type = _document_block_type(item.item_type)
        return DocumentBlock(
            block_id=f"block_{index + 1:04d}",
            doc_id=extraction.doc_id,
            block_type=block_type,  # type: ignore[arg-type]
            text=item.text or _extract_text(item.payload),
            page=source.page if source else None,
            bbox=source.bbox if source else None,
            section=source.section if source else None,
            source_file=source.source_file if source else extraction.source_file,
            confidence=item.confidence,
            raw_payload=item.payload,
            errors=list(item.errors),
        )

    def _build_molecules(self, extraction: RawExtraction) -> list[MoleculeCard]:
        cards: list[MoleculeCard] = []
        for item_index, item in enumerate(extraction.molecules):
            candidates = _extract_molecule_candidates(item.payload)
            if not candidates:
                candidates = [{"raw_payload": item.payload}]

            # Collect raw scores for this group; normalize per-item so
            # candidates from the same figure are comparable to each other.
            raw_scores = [
                _first_float(c, ["score", "confidence"]) or 0.0
                for c in candidates
            ]

            for candidate_index, candidate in enumerate(candidates):
                card_id = f"mol_card_{item_index + 1:04d}_{candidate_index + 1:02d}"
                mention = _first_str(candidate, ["text", "name", "label", "mention"])
                smiles = _first_str(candidate, ["smiles", "SMILES", "canonical_smiles"])
                aliases = [mention] if mention else []
                names = [mention] if mention else []
                errors = list(item.errors)
                if not mention and not smiles:
                    errors.append("No molecule name or SMILES found in raw item.")

                confidence = _normalize_confidence(
                    raw_scores[candidate_index], raw_scores
                )

                cards.append(
                    MoleculeCard(
                        molecule_card_id=card_id,
                        doc_id=extraction.doc_id,
                        names=_dedupe(names),
                        raw_smiles=smiles,
                        aliases=_dedupe(aliases),
                        source_mentions=[],
                        source_images=[],
                        confidence=confidence,
                        raw_payload=candidate,
                        errors=errors,
                    )
                )
        return cards

    def _build_reactions(self, extraction: RawExtraction) -> list[ReactionEventCard]:
        cards: list[ReactionEventCard] = []
        for item_index, item in enumerate(extraction.reactions):
            candidates = _extract_reaction_candidates(item.payload)
            if not candidates:
                candidates = [{"raw_payload": item.payload}]

            for candidate_index, candidate in enumerate(candidates):
                reaction_id = f"rxn_event_{item_index + 1:04d}_{candidate_index + 1:02d}"
                source = item.source or SourceProvenance(
                    doc_id=extraction.doc_id,
                    source_file=extraction.source_file,
                )
                reactants = _participants(candidate.get("reactants"), role="reactant")
                products = _participants(candidate.get("products"), role="product")
                conditions = _condition_strings(candidate)
                yield_value = _extract_yield(candidate)
                procedure_text = item.text or _extract_text(candidate)

                cards.append(
                    ReactionEventCard(
                        reaction_event_id=reaction_id,
                        doc_id=extraction.doc_id,
                        reaction_smiles=_first_str(
                            candidate, ["reaction_smiles", "rxn_smiles"]
                        ),
                        reactants=reactants,
                        products=products,
                        reagents=conditions.get("reagents", []),
                        catalysts=conditions.get("catalysts", []),
                        solvents=conditions.get("solvents", []),
                        temperature=_first_or_none(conditions.get("temperature", [])),
                        time=_first_or_none(conditions.get("time", [])),
                        yield_value=yield_value,
                        procedure_text=procedure_text,
                        source=source,
                        confidence=item.confidence,
                        evidence_completeness=_reaction_completeness(
                            reactants, products, conditions, yield_value, procedure_text
                        ),
                        raw_payload=candidate,
                        errors=list(item.errors),
                    )
                )
        return cards

    def _build_facts(
        self,
        extraction: RawExtraction,
        blocks: list[DocumentBlock],
        reactions: list[ReactionEventCard],
    ) -> list[FactCard]:
        facts: list[FactCard] = []
        for block in blocks:
            if not block.text:
                continue
            facts.append(
                FactCard(
                    fact_card_id=f"fact_block_{len(facts) + 1:04d}",
                    doc_id=extraction.doc_id,
                    fact_type="procedure"
                    if block.block_type == "procedure"
                    else "claim",
                    claim=block.text,
                    supporting_blocks=[block.block_id],
                    source=SourceProvenance(
                        doc_id=block.doc_id,
                        source_file=block.source_file,
                        page=block.page,
                        bbox=block.bbox,
                        block_id=block.block_id,
                        section=block.section,
                    ),
                    confidence=block.confidence,
                    raw_payload=block.raw_payload,
                    errors=list(block.errors),
                )
            )

        for reaction in reactions:
            if reaction.procedure_text:
                facts.append(
                    FactCard(
                        fact_card_id=f"fact_rxn_{len(facts) + 1:04d}",
                        doc_id=extraction.doc_id,
                        fact_type="reaction_condition",
                        claim=reaction.procedure_text,
                        entities=[reaction.reaction_event_id],
                        source=reaction.source,
                        confidence=reaction.confidence,
                        raw_payload=reaction.raw_payload,
                        errors=list(reaction.errors),
                    )
                )
        return facts


_DOCUMENT_BLOCK_TYPES = {
    "table",
    "figure",
    "caption",
    "procedure",
    "unknown",
}


def _document_block_type(item_type: str) -> str:
    if item_type == "block":
        return "paragraph"
    if item_type in _DOCUMENT_BLOCK_TYPES:
        return item_type
    return "unknown"


def _first_or_none(values: list[str]) -> str | None:
    return values[0] if values else None


def _extract_molecule_candidates(payload: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for key in ("molecules", "molecule", "labels"):
        value = payload.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    candidates.append(item)
                elif isinstance(item, (list, tuple)) and item:
                    candidates.append({"text": str(item[0]), "raw_value": list(item)})
                else:
                    candidates.append({"text": str(item)})
        elif isinstance(value, dict):
            candidates.append(value)
    if not candidates and any(key in payload for key in ("smiles", "SMILES", "text")):
        candidates.append(payload)
    return candidates


def _extract_reaction_candidates(payload: dict[str, Any]) -> list[dict[str, Any]]:
    value = payload.get("reactions")
    if isinstance(value, list):
        return [item if isinstance(item, dict) else {"raw_value": item} for item in value]
    if isinstance(value, dict):
        return [value]
    if any(key in payload for key in ("reactants", "products", "conditions", "tokens")):
        return [payload]
    return []


def _participants(value: Any, *, role: str) -> list[ReactionParticipant]:
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    participants: list[ReactionParticipant] = []
    for item in values:
        if isinstance(item, dict):
            participants.append(
                ReactionParticipant(
                    name=_first_str(item, ["name", "text", "label", "category"]),
                    smiles=_first_str(item, ["smiles", "SMILES"]),
                    role=role,  # type: ignore[arg-type]
                    confidence=_first_float(item, ["score", "confidence"]),
                )
            )
        else:
            participants.append(ReactionParticipant(name=str(item), role=role))  # type: ignore[arg-type]
    return participants


def _condition_strings(candidate: dict[str, Any]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {
        "reagents": [],
        "catalysts": [],
        "solvents": [],
        "temperature": [],
        "time": [],
        "other": [],
    }
    conditions = candidate.get("conditions", [])
    if isinstance(conditions, dict):
        conditions = [conditions]
    if not isinstance(conditions, list):
        conditions = [conditions]

    for condition in conditions:
        text = _condition_text(condition)
        if not text:
            continue
        lowered = text.lower()
        if "solvent" in lowered:
            result["solvents"].append(text)
        elif "catalyst" in lowered or "catalyst" in _condition_category(condition):
            result["catalysts"].append(text)
        elif "temp" in lowered or "°" in text:
            result["temperature"].append(text)
        elif "time" in lowered or re.search(r"\b\d+\s*(h|hr|min)\b", lowered):
            result["time"].append(text)
        elif "reagent" in lowered:
            result["reagents"].append(text)
        else:
            result["other"].append(text)
    return result


def _condition_category(condition: Any) -> str:
    if isinstance(condition, dict):
        return str(condition.get("category", ""))
    return ""


def _condition_text(condition: Any) -> str | None:
    if isinstance(condition, dict):
        text = condition.get("text")
        if isinstance(text, list):
            return " ".join(str(part) for part in text)
        if text is not None:
            return str(text)
        return _extract_text(condition)
    if condition is None:
        return None
    return str(condition)


def _extract_yield(candidate: dict[str, Any]) -> YieldValue | None:
    for key in ("yield", "yield_value"):
        value = candidate.get(key)
        if value is None:
            continue
        if isinstance(value, dict):
            raw = value.get("raw_text") or value.get("text") or value.get("value")
            number = _to_float(value.get("value"))
            unit = str(value.get("unit", "%"))
        else:
            raw = str(value)
            number = _to_float(value)
            unit = "%"
        normalized = number / 100 if number is not None and unit == "%" else None
        return YieldValue(value=number, unit=unit, normalized_value=normalized, raw_text=str(raw))

    text = _extract_text(candidate)
    if not text:
        return None
    match = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
    if not match:
        return None
    value = float(match.group(1))
    return YieldValue(value=value, unit="%", normalized_value=value / 100, raw_text=match.group(0))


def _reaction_completeness(
    reactants: list[ReactionParticipant],
    products: list[ReactionParticipant],
    conditions: dict[str, list[str]],
    yield_value: YieldValue | None,
    procedure_text: str | None,
) -> float:
    slots = [
        bool(reactants),
        bool(products),
        any(conditions.values()),
        yield_value is not None,
        bool(procedure_text),
    ]
    return sum(slots) / len(slots)


def _extract_text(payload: dict[str, Any]) -> str | None:
    for key in ("text", "caption", "procedure_text", "claim"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    tokens = payload.get("tokens")
    if isinstance(tokens, list):
        return " ".join(str(token) for token in tokens)
    return None


def _first_str(payload: dict[str, Any], keys: list[str]) -> str | None:
    for key in keys:
        value = payload.get(key)
        if value not in (None, ""):
            return str(value)
    return None


def _first_float(payload: dict[str, Any], keys: list[str]) -> float | None:
    for key in keys:
        value = _to_float(payload.get(key))
        if value is not None:
            return value
    return None


def _normalize_confidence(value: float, values_in_group: list[float]) -> float:
    """Normalize a raw parser score into [0, 1].

    When there is a single candidate and the score is already in [0, 1]
    it is returned as-is.  Otherwise scores are softmin-normalised within
    the group so that the best candidate (lowest raw score) gets a
    confidence near 1.
    """
    if not values_in_group:
        return 0.5

    # Single candidate with a score already in [0, 1] — trust it.
    if len(values_in_group) == 1 and 0 <= value <= 1:
        return value

    # Softmin: lower raw score → higher confidence.
    min_val = min(values_in_group)
    shifted = [v - min_val for v in values_in_group]
    exp_vals = [2.718281828 ** (-s) for s in shifted]
    total = sum(exp_vals)
    if total <= 0:
        return 1.0 / max(1, len(values_in_group))
    idx = values_in_group.index(value)
    return max(0.0, min(1.0, exp_vals[idx] / total))


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"\d+(?:\.\d+)?", str(value))
    if not match:
        return None
    return float(match.group(0))


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result
