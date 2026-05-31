"""Reaction-event retrieval over local ReactionEventCard evidence."""

from __future__ import annotations

from typing import Any

from storage import LocalStore

from .common import (
    ScoredCandidate,
    load_or_empty,
    reaction_summary,
    reaction_to_candidate,
    resolve_doc_ids,
    sort_candidates,
    text_score,
)


def reaction_event_search(
    filters: dict[str, Any] | str,
    store: LocalStore,
    *,
    doc_ids: list[str] | None = None,
    top_k: int = 10,
):
    normalized_filters = {"query": filters} if isinstance(filters, str) else filters
    query = str(normalized_filters.get("query", ""))
    candidates: list[ScoredCandidate] = []

    for doc_id in resolve_doc_ids(store, doc_ids):
        for reaction in load_or_empty(store.load_reactions, doc_id):
            score = text_score(query, reaction_summary(reaction)) if query else 0.0
            score += _structured_filter_score(normalized_filters, reaction)
            if score > 0 or not normalized_filters:
                candidates.append(
                    ScoredCandidate(
                        score,
                        reaction_to_candidate(reaction, score=score, query=query or "reaction"),
                    )
                )

    return sort_candidates(candidates, top_k)


def _structured_filter_score(filters: dict[str, Any], reaction) -> float:
    score = 0.0
    score += _list_filter_score(filters.get("reactants"), _participant_values(reaction.reactants))
    score += _list_filter_score(filters.get("products"), _participant_values(reaction.products))
    score += _list_filter_score(filters.get("reagents"), reaction.reagents)
    score += _list_filter_score(filters.get("catalysts"), reaction.catalysts)
    score += _list_filter_score(filters.get("solvents"), reaction.solvents)
    if filters.get("yield_min") is not None and reaction.yield_value and reaction.yield_value.value is not None:
        if reaction.yield_value.value >= float(filters["yield_min"]):
            score += 2.0
    return score


def _participant_values(participants) -> list[str]:
    values: list[str] = []
    for participant in participants:
        values.extend(value for value in [participant.name, participant.smiles] if value)
    return values


def _list_filter_score(expected: Any, values: list[str]) -> float:
    if expected is None:
        return 0.0
    expected_values = expected if isinstance(expected, list) else [expected]
    haystack = " | ".join(values).lower()
    return sum(2.0 for value in expected_values if str(value).lower() in haystack)

