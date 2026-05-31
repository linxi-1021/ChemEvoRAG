"""Shared helpers for local evidence retrieval."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterable, TypeVar

from evidence import (
    CandidateEvidence,
    DocumentBlock,
    FactCard,
    MoleculeCard,
    ReactionEventCard,
    SourceProvenance,
)
from storage import LocalStore


T = TypeVar("T")


@dataclass
class ScoredCandidate:
    score: float
    evidence: CandidateEvidence


def discover_doc_ids(store: LocalStore) -> list[str]:
    doc_ids: set[str] = set()
    for path in store.evidence_dir.glob("*.json"):
        name = path.name
        for suffix in (
            ".blocks.json",
            ".molecules.json",
            ".reactions.json",
            ".facts.json",
        ):
            if name.endswith(suffix):
                doc_ids.add(name[: -len(suffix)])
    return sorted(doc_ids)


def resolve_doc_ids(store: LocalStore, doc_ids: Iterable[str] | None) -> list[str]:
    return sorted(set(doc_ids)) if doc_ids is not None else discover_doc_ids(store)


def load_or_empty(loader: Callable[[str], list[T]], doc_id: str) -> list[T]:
    try:
        return loader(doc_id)
    except FileNotFoundError:
        return []


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_+.%]+", text.lower())


def text_score(query: str, haystack: str) -> float:
    terms = tokenize(query)
    if not terms:
        return 0.0
    haystack_lower = haystack.lower()
    haystack_terms = tokenize(haystack)
    score = 0.0
    # Full-phrase match
    if query.lower() in haystack_lower:
        score += 2.0
    # Per-term matching
    for term in terms:
        if term in haystack_lower:
            # Boost rare/chemical tokens (containing digits or special chars)
            weight = 1.5 if re.search(r"\d", term) and len(term) <= 6 else 1.0
            score += weight
        else:
            for ht in haystack_terms:
                if ht.startswith(term) or term.startswith(ht):
                    score += 0.6
                    break
    # Numeric value matching — if query has numbers, boost exact matches
    query_numbers = set(re.findall(r"\d+(?:\.\d+)?%?", query))
    for num in query_numbers:
        if num in haystack:
            score += 1.5
    return score


def sort_candidates(candidates: list[ScoredCandidate], top_k: int) -> list[CandidateEvidence]:
    candidates.sort(key=lambda item: item.score, reverse=True)
    return [item.evidence for item in candidates[:top_k]]


def confidence_from_score(score: float, query: str) -> float:
    terms = max(5, len(tokenize(query)))
    return min(1.0, score / (terms + 2))


def block_to_candidate(block: DocumentBlock, *, score: float, query: str) -> CandidateEvidence:
    source = source_from_block(block)
    return CandidateEvidence(
        evidence_id=block.block_id,
        evidence_type="document_block",
        summary=block.text,
        structured_slots={
            "doc_id": block.doc_id,
            "block_type": block.block_type,
            "score": score,
        },
        source=source,
        confidence=confidence_from_score(score, query),
    )


def molecule_to_candidate(
    molecule: MoleculeCard, *, score: float, query: str
) -> CandidateEvidence:
    summary_parts = [
        *molecule.names,
        *molecule.aliases,
        molecule.canonical_smiles,
        molecule.raw_smiles,
        molecule.inchi_key,
    ]
    return CandidateEvidence(
        evidence_id=molecule.molecule_card_id,
        evidence_type="molecule",
        summary=" | ".join(part for part in summary_parts if part),
        structured_slots={
            "doc_id": molecule.doc_id,
            "names": molecule.names,
            "aliases": molecule.aliases,
            "raw_smiles": molecule.raw_smiles,
            "canonical_smiles": molecule.canonical_smiles,
            "inchi_key": molecule.inchi_key,
            "linked_elementkg_id": molecule.linked_elementkg_id,
            "score": score,
        },
        source=source_from_molecule(molecule),
        confidence=confidence_from_score(score, query),
    )


def reaction_to_candidate(
    reaction: ReactionEventCard, *, score: float, query: str
) -> CandidateEvidence:
    return CandidateEvidence(
        evidence_id=reaction.reaction_event_id,
        evidence_type="reaction_event",
        summary=reaction_summary(reaction),
        structured_slots={
            "doc_id": reaction.doc_id,
            "reactants": [participant.model_dump(mode="json") for participant in reaction.reactants],
            "products": [participant.model_dump(mode="json") for participant in reaction.products],
            "reagents": reaction.reagents,
            "catalysts": reaction.catalysts,
            "solvents": reaction.solvents,
            "temperature": reaction.temperature,
            "time": reaction.time,
            "yield": reaction.yield_value.model_dump(mode="json")
            if reaction.yield_value
            else None,
            "score": score,
        },
        source=reaction.source,
        confidence=confidence_from_score(score, query),
    )


def fact_to_candidate(fact: FactCard, *, score: float, query: str) -> CandidateEvidence:
    return CandidateEvidence(
        evidence_id=fact.fact_card_id,
        evidence_type="fact",
        summary=fact.claim,
        structured_slots={
            "doc_id": fact.doc_id,
            "fact_type": fact.fact_type,
            "entities": fact.entities,
            "supporting_blocks": fact.supporting_blocks,
            "supporting_figures": fact.supporting_figures,
            "score": score,
        },
        source=fact.source,
        confidence=confidence_from_score(score, query),
    )


def source_from_block(block: DocumentBlock) -> SourceProvenance:
    return SourceProvenance(
        doc_id=block.doc_id,
        source_file=block.source_file,
        page=block.page,
        bbox=block.bbox,
        block_id=block.block_id,
        section=block.section,
    )


def source_from_molecule(molecule: MoleculeCard) -> SourceProvenance:
    if molecule.source_mentions:
        return molecule.source_mentions[0].provenance
    if molecule.source_images:
        return molecule.source_images[0].provenance
    return SourceProvenance(doc_id=molecule.doc_id)


def reaction_summary(reaction: ReactionEventCard) -> str:
    parts: list[str] = []
    parts.extend(_participant_text(participant) for participant in reaction.reactants)
    parts.extend(_participant_text(participant) for participant in reaction.products)
    parts.extend(reaction.reagents)
    parts.extend(reaction.catalysts)
    parts.extend(reaction.solvents)
    if reaction.temperature:
        parts.append(f"temperature: {reaction.temperature}")
    if reaction.time:
        parts.append(f"time: {reaction.time}")
    if reaction.yield_value:
        parts.append(
            f"yield: {reaction.yield_value.raw_text or reaction.yield_value.value}"
        )
    if reaction.procedure_text:
        parts.append(reaction.procedure_text)
    # Include raw condition text from the payload for semantic search
    raw_conds = reaction.raw_payload.get("conditions", [])
    for c in raw_conds if isinstance(raw_conds, list) else []:
        if isinstance(c, dict):
            _txt = c.get("text", "")
            if isinstance(_txt, list):
                parts.append(" ".join(str(t) for t in _txt))
            elif _txt:
                parts.append(str(_txt))
    return " | ".join(part for part in parts if part)


def molecule_search_text(molecule: MoleculeCard) -> str:
    values = [
        *molecule.local_ids,
        *molecule.names,
        *molecule.aliases,
        molecule.raw_smiles,
        molecule.canonical_smiles,
        molecule.inchi_key,
        molecule.iupac_name,
        molecule.linked_elementkg_id,
    ]
    return " | ".join(value for value in values if value)


def block_search_text(block: DocumentBlock) -> str:
    values = [block.text, block.section, block.block_type]
    return " | ".join(value for value in values if value)


def fact_search_text(fact: FactCard) -> str:
    values = [
        fact.claim,
        fact.fact_type,
        *fact.entities,
        *fact.supporting_blocks,
        *fact.supporting_figures,
    ]
    return " | ".join(value for value in values if value)


def _participant_text(participant) -> str:
    return " ".join(value for value in [participant.name, participant.smiles] if value)

