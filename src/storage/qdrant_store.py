"""Qdrant local-mode vector store for evidence embedding search."""

from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


def _to_uuid(raw_id: str) -> str:
    """Convert an arbitrary string ID to a valid UUID v5."""
    return str(uuid.uuid5(uuid.NAMESPACE_OID, raw_id))


class QdrantStore:
    """Local Qdrant collection for evidence vector search.

    Data is persisted as files under *path* — no server needed.

    Usage::

        store = QdrantStore(path="data/indexes/qdrant")
        store.ensure_collection(vector_size=384)
        store.upsert([("id1", vec1, payload1), ...])
        results = store.search(query_vec, top_k=10)
    """

    def __init__(
        self,
        path: str = "data/indexes/qdrant",
        collection: str = "chemevorag",
    ) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        self._client = QdrantClient(path=path)
        self.collection = collection

    def ensure_collection(self, vector_size: int = 384) -> None:
        """Create the collection if it does not exist."""
        if not self._client.collection_exists(self.collection):
            self._client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                ),
            )

    def upsert(
        self,
        items: list[tuple[str, np.ndarray, dict]],
    ) -> None:
        points = [
            PointStruct(id=_to_uuid(id_), vector=vec.tolist(), payload=payload)
            for id_, vec, payload in items
        ]
        self._client.upsert(
            collection_name=self.collection,
            points=points,
            wait=True,
        )

    def search(
        self,
        query_vec: np.ndarray,
        top_k: int = 10,
    ) -> list[tuple[str, float, dict]]:
        hits = self._client.query_points(
            collection_name=self.collection,
            query=query_vec.tolist(),
            limit=top_k,
        )
        return [(hit.id, hit.score, hit.payload or {}) for hit in hits.points]

    def count(self) -> int:
        info = self._client.get_collection(self.collection)
        return info.points_count
