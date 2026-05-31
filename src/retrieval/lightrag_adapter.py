"""LightRAG adapter boundary for evidence recall.

This module keeps a local evidence-document manifest so queries can return
ChemEvoRAG evidence ids even when the real LightRAG runtime is not configured.
If a LightRAG-like client is injected, indexed document text is also passed to it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import Field

from evidence.schemas import ChemEvoBaseModel
from storage import LocalStore

from .common import (
    block_search_text,
    discover_doc_ids,
    fact_search_text,
    load_or_empty,
    molecule_search_text,
    reaction_summary,
    text_score,
)


class LightRAGEvidenceDocument(ChemEvoBaseModel):
    evidence_id: str
    evidence_type: str
    doc_id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class LightRAGAdapter:
    """Index local evidence summaries for optional LightRAG recall."""

    def __init__(
        self,
        store: LocalStore,
        *,
        index_path: str | Path | None = None,
        rag_client: Any | None = None,
    ) -> None:
        self.store = store
        self.index_path = (
            Path(index_path)
            if index_path is not None
            else store.base_dir / "data" / "indexes" / "lightrag_documents.jsonl"
        )
        self.rag_client = rag_client

    def build_documents(
        self, doc_ids: list[str] | None = None
    ) -> list[LightRAGEvidenceDocument]:
        resolved_doc_ids = doc_ids or discover_doc_ids(self.store)
        documents: list[LightRAGEvidenceDocument] = []
        for doc_id in resolved_doc_ids:
            for block in load_or_empty(self.store.load_blocks, doc_id):
                text = block_search_text(block)
                if text:
                    documents.append(
                        LightRAGEvidenceDocument(
                            evidence_id=block.block_id,
                            evidence_type="document_block",
                            doc_id=doc_id,
                            text=text,
                            metadata={"page": block.page, "block_type": block.block_type},
                        )
                    )
            for molecule in load_or_empty(self.store.load_molecules, doc_id):
                text = molecule_search_text(molecule)
                if text:
                    documents.append(
                        LightRAGEvidenceDocument(
                            evidence_id=molecule.molecule_card_id,
                            evidence_type="molecule",
                            doc_id=doc_id,
                            text=text,
                            metadata={"names": molecule.names, "aliases": molecule.aliases},
                        )
                    )
            for reaction in load_or_empty(self.store.load_reactions, doc_id):
                text = reaction_summary(reaction)
                if text:
                    documents.append(
                        LightRAGEvidenceDocument(
                            evidence_id=reaction.reaction_event_id,
                            evidence_type="reaction_event",
                            doc_id=doc_id,
                            text=text,
                            metadata={"source": reaction.source.model_dump(mode="json") if reaction.source else None},
                        )
                    )
            for fact in load_or_empty(self.store.load_facts, doc_id):
                text = fact_search_text(fact)
                if text:
                    documents.append(
                        LightRAGEvidenceDocument(
                            evidence_id=fact.fact_card_id,
                            evidence_type="fact",
                            doc_id=doc_id,
                            text=text,
                            metadata={"fact_type": fact.fact_type},
                        )
                    )
        return documents

    def index(self, doc_ids: list[str] | None = None) -> list[LightRAGEvidenceDocument]:
        documents = self.build_documents(doc_ids)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with self.index_path.open("w", encoding="utf-8") as handle:
            for document in documents:
                handle.write(json.dumps(document.model_dump(mode="json"), ensure_ascii=False))
                handle.write("\n")

        if self.rag_client is not None:
            for document in documents:
                text = _format_lightrag_document(document)
                insert = getattr(self.rag_client, "insert", None)
                if insert is not None:
                    insert(text)
        return documents

    def query(self, query: str, *, top_k: int = 10) -> list[LightRAGEvidenceDocument]:
        documents = self._load_index()
        scored = [
            (text_score(query, document.text), document)
            for document in documents
        ]
        scored = [(score, document) for score, document in scored if score > 0]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [document for _, document in scored[:top_k]]

    def _load_index(self) -> list[LightRAGEvidenceDocument]:
        if not self.index_path.exists():
            return self.index()
        documents: list[LightRAGEvidenceDocument] = []
        with self.index_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    documents.append(LightRAGEvidenceDocument.model_validate_json(line))
        return documents


def _format_lightrag_document(document: LightRAGEvidenceDocument) -> str:
    return (
        f"evidence_id: {document.evidence_id}\n"
        f"evidence_type: {document.evidence_type}\n"
        f"doc_id: {document.doc_id}\n"
        f"text: {document.text}"
    )

