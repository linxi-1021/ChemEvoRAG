"""Dense (semantic) retrieval over local evidence via vector embeddings.

Indexes evidence text into a Qdrant collection using SentenceTransformer,
then searches by cosine similarity at query time.
"""

from __future__ import annotations

import os

# Force offline mode to avoid HuggingFace network calls
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
from sentence_transformers import SentenceTransformer

from evidence import CandidateEvidence
from storage.local_store import LocalStore
from storage.qdrant_store import QdrantStore

from .common import (
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
)

DEFAULT_MODEL = "all-MiniLM-L6-v2"


class DenseRetriever:
    """Embed evidence text and search by semantic similarity."""

    def __init__(
        self,
        qdrant_store: QdrantStore,
        model_name: str = DEFAULT_MODEL,
    ) -> None:
        self._qdrant = qdrant_store
        self._model = SentenceTransformer(model_name)
        self._qdrant.ensure_collection(
            vector_size=self._model.get_sentence_embedding_dimension()
        )

    def index(
        self, evidence_store: LocalStore, doc_ids: list[str] | None = None
    ) -> int:
        """Embed all evidence from *evidence_store* and upsert into Qdrant.

        Returns the number of points indexed.
        """
        chunks = self._build_chunks(evidence_store, doc_ids)
        if not chunks:
            return 0

        texts = [c["text"] for c in chunks]
        vectors = self._model.encode(texts, show_progress_bar=True)

        items = [
            (c["evidence_id"], vectors[i].astype(np.float32), c["payload"])
            for i, c in enumerate(chunks)
        ]
        self._qdrant.upsert(items)
        return len(items)

    def search(self, query: str, top_k: int = 10) -> list[CandidateEvidence]:
        """Embed *query* and return the top-k most similar evidence items."""
        q_vec = self._model.encode([query])[0].astype(np.float32)
        hits = self._qdrant.search(q_vec, top_k=top_k)

        results: list[CandidateEvidence] = []
        for ev_id, score, payload in hits:
            etype = payload.get("evidence_type", "document_block")
            summary = payload.get("text", "")
            doc_id = payload.get("doc_id", "")
            results.append(
                CandidateEvidence(
                    evidence_id=payload.get("evidence_id", ev_id),
                    evidence_type=etype,  # type: ignore[arg-type]
                    summary=summary,
                    structured_slots={"doc_id": doc_id, "score": score},
                    confidence=min(1.0, max(0.0, score)),
                )
            )
        return results

    def _build_chunks(
        self, store: LocalStore, doc_ids: list[str] | None
    ) -> list[dict]:
        chunks: list[dict] = []
        for doc_id in resolve_doc_ids(store, doc_ids):
            for block in load_or_empty(store.load_blocks, doc_id):
                text = block_search_text(block)
                if text:
                    chunks.append({
                        "evidence_id": block.block_id,
                        "text": text,
                        "payload": {
                            "evidence_id": block.block_id,
                            "evidence_type": "document_block",
                            "doc_id": doc_id,
                            "text": text,
                        },
                    })
            for mol in load_or_empty(store.load_molecules, doc_id):
                text = molecule_search_text(mol)
                if text:
                    chunks.append({
                        "evidence_id": mol.molecule_card_id,
                        "text": text,
                        "payload": {
                            "evidence_id": mol.molecule_card_id,
                            "evidence_type": "molecule",
                            "doc_id": doc_id,
                            "text": text,
                        },
                    })
            for rxn in load_or_empty(store.load_reactions, doc_id):
                text = reaction_summary(rxn)
                if text:
                    chunks.append({
                        "evidence_id": rxn.reaction_event_id,
                        "text": text,
                        "payload": {
                            "evidence_id": rxn.reaction_event_id,
                            "evidence_type": "reaction_event",
                            "doc_id": doc_id,
                            "text": text,
                        },
                    })
            for fact in load_or_empty(store.load_facts, doc_id):
                text = fact_search_text(fact)
                if text:
                    chunks.append({
                        "evidence_id": fact.fact_card_id,
                        "text": text,
                        "payload": {
                            "evidence_id": fact.fact_card_id,
                            "evidence_type": "fact",
                            "doc_id": doc_id,
                            "text": text,
                        },
                    })
        return chunks


def dense_search(
    query: str,
    evidence_store: LocalStore | None = None,
    *,
    doc_ids: list[str] | None = None,
    top_k: int = 10,
    qdrant_store: QdrantStore | None = None,
    model_name: str = DEFAULT_MODEL,
) -> list[CandidateEvidence]:
    """Convenience wrapper: embed *query* and search Qdrant.

    If *qdrant_store* is not supplied a default local-mode instance at
    ``data/indexes/qdrant`` is created on the fly.
    """
    if qdrant_store is None:
        qdrant_store = QdrantStore()
    retriever = DenseRetriever(qdrant_store, model_name=model_name)
    results = retriever.search(query, top_k=top_k)
    if doc_ids:
        results = [
            r
            for r in results
            if r.structured_slots.get("doc_id") in doc_ids
        ]
    return results
