"""Lexical retrieval over local evidence JSON files."""

from __future__ import annotations

from storage import LocalStore

from .common import (
    ScoredCandidate,
    block_search_text,
    block_to_candidate,
    fact_search_text,
    fact_to_candidate,
    load_or_empty,
    molecule_search_text,
    molecule_to_candidate,
    reaction_summary,
    reaction_to_candidate,
    resolve_doc_ids,
    sort_candidates,
    text_score,
)


def lexical_search(
    query: str,
    store: LocalStore,
    *,
    doc_ids: list[str] | None = None,
    top_k: int = 20,
):
    candidates: list[ScoredCandidate] = []
    for doc_id in resolve_doc_ids(store, doc_ids):
        for block in load_or_empty(store.load_blocks, doc_id):
            score = text_score(query, block_search_text(block))
            if score > 0:
                candidates.append(
                    ScoredCandidate(score, block_to_candidate(block, score=score, query=query))
                )

        for molecule in load_or_empty(store.load_molecules, doc_id):
            score = text_score(query, molecule_search_text(molecule))
            if score > 0:
                candidates.append(
                    ScoredCandidate(score, molecule_to_candidate(molecule, score=score, query=query))
                )

        for reaction in load_or_empty(store.load_reactions, doc_id):
            score = text_score(query, reaction_summary(reaction))
            if score > 0:
                candidates.append(
                    ScoredCandidate(score, reaction_to_candidate(reaction, score=score, query=query))
                )

        for fact in load_or_empty(store.load_facts, doc_id):
            score = text_score(query, fact_search_text(fact))
            if score > 0:
                candidates.append(
                    ScoredCandidate(score, fact_to_candidate(fact, score=score, query=query))
                )

    return sort_candidates(candidates, top_k)

