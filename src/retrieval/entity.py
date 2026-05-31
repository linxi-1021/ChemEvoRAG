"""Entity retrieval over local MoleculeCard evidence."""

from __future__ import annotations

import re

from storage import LocalStore

from .common import (
    ScoredCandidate,
    load_or_empty,
    molecule_search_text,
    molecule_to_candidate,
    resolve_doc_ids,
    sort_candidates,
    text_score,
)


def entity_search(
    name_or_smiles_or_inchikey: str,
    store: LocalStore,
    *,
    doc_ids: list[str] | None = None,
    top_k: int = 10,
):
    candidates: list[ScoredCandidate] = []
    for doc_id in resolve_doc_ids(store, doc_ids):
        for molecule in load_or_empty(store.load_molecules, doc_id):
            score = text_score(name_or_smiles_or_inchikey, molecule_search_text(molecule))
            if _exact_entity_match(name_or_smiles_or_inchikey, molecule):
                score += 5.0
            else:
                overlap = _entity_token_overlap(name_or_smiles_or_inchikey, molecule)
                if overlap >= 0.5:
                    score += 2.0
            if score > 0:
                candidates.append(
                    ScoredCandidate(
                        score,
                        molecule_to_candidate(
                            molecule,
                            score=score,
                            query=name_or_smiles_or_inchikey,
                        ),
                    )
                )
    return sort_candidates(candidates, top_k)


def _exact_entity_match(query: str, molecule) -> bool:
    query_lower = query.lower()
    fields = [
        *molecule.local_ids,
        *molecule.names,
        *molecule.aliases,
        molecule.raw_smiles,
        molecule.canonical_smiles,
        molecule.inchi_key,
        molecule.iupac_name,
        molecule.linked_elementkg_id,
    ]
    return any(value and str(value).lower() == query_lower for value in fields)


def _entity_token_overlap(query: str, molecule) -> float:
    """Return overlap ratio between query tokens and molecule field tokens."""
    query_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
    if not query_tokens:
        return 0.0
    field_values = [
        *molecule.local_ids,
        *molecule.names,
        *molecule.aliases,
        molecule.raw_smiles or "",
        molecule.canonical_smiles or "",
        molecule.iupac_name or "",
    ]
    field_tokens = set()
    for v in field_values:
        field_tokens.update(re.findall(r"[a-z0-9]+", str(v).lower()))
    if not field_tokens:
        return 0.0
    overlap = query_tokens & field_tokens
    return len(overlap) / len(query_tokens)

