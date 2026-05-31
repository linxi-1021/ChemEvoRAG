"""Provenance lookup for local evidence ids."""

from __future__ import annotations

from evidence import SourceProvenance
from storage import LocalStore

from .common import (
    load_or_empty,
    resolve_doc_ids,
    source_from_block,
    source_from_molecule,
)


def provenance_backtrack(
    evidence_id: str,
    store: LocalStore,
    *,
    doc_ids: list[str] | None = None,
) -> SourceProvenance | None:
    for doc_id in resolve_doc_ids(store, doc_ids):
        for block in load_or_empty(store.load_blocks, doc_id):
            if block.block_id == evidence_id:
                return source_from_block(block)

        for molecule in load_or_empty(store.load_molecules, doc_id):
            if molecule.molecule_card_id == evidence_id:
                return source_from_molecule(molecule)

        for reaction in load_or_empty(store.load_reactions, doc_id):
            if reaction.reaction_event_id == evidence_id:
                return reaction.source

        for fact in load_or_empty(store.load_facts, doc_id):
            if fact.fact_card_id == evidence_id:
                return fact.source

    return None

