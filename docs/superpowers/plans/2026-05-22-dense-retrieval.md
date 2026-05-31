# Dense Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add semantic vector search (Dense Retrieval) to ChemEvoRAG via SentenceTransformer embeddings + Qdrant local-mode vector index.

**Architecture:** New `QdrantStore` wraps Qdrant local-mode collection. New `DenseRetriever` uses `SentenceTransformer("all-MiniLM-L6-v2")` to embed evidence text into 384-dim vectors, then queries Qdrant by cosine similarity. `RetrievalRouter` merges dense results with lexical for `paper_local_question` intent. A CLI script `index_evidence.py` builds the index from stored evidence in one shot.

**Tech Stack:** `qdrant-client`, `sentence-transformers==2.7.0`, `all-MiniLM-L6-v2` (384 dim), Qdrant local mode (file-based, no Docker).

**Design spec:** `docs/superpowers/specs/2026-05-22-dense-retrieval-design.md`

---

### Task 1: Install qdrant-client

- [ ] **Step 1: Install qdrant-client**

```bash
pip install qdrant-client
```

- [ ] **Step 2: Verify import**

```bash
python -c "import qdrant_client; print(qdrant_client.__version__)"
```

Expected: prints version string (e.g. `1.13.x`)

---

### Task 2: QdrantStore

**Files:**
- Create: `src/storage/qdrant_store.py`
- Test: `tests/test_dense_retrieval.py` (test class written in this task)

- [ ] **Step 1: Write failing test for QdrantStore**

File: `tests/test_dense_retrieval.py`

```python
"""Tests for Dense Retrieval — QdrantStore and DenseRetriever."""

import numpy as np
from storage.qdrant_store import QdrantStore


class TestQdrantStore:
    def test_create_collection_and_upsert_search(self, tmp_path):
        store = QdrantStore(path=str(tmp_path / "qdrant"), collection="test_coll")
        store.ensure_collection(vector_size=4)

        vec1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        vec2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        store.upsert(
            [
                ("id_1", vec1, {"evidence_type": "molecule", "text": "ethanol"}),
                ("id_2", vec2, {"evidence_type": "molecule", "text": "benzene"}),
            ]
        )

        assert store.count() == 2

        results = store.search(np.array([0.99, 0.01, 0.0, 0.0], dtype=np.float32), top_k=2)
        assert results[0][0] == "id_1"
        assert results[0][1] > 0.9  # cosine similarity near 1.0
        assert results[1][0] == "id_2"
```

- [ ] **Step 2: Run test — expected FAIL**

```bash
pytest tests/test_dense_retrieval.py::TestQdrantStore -v
```

Expected: `ModuleNotFoundError: No module named 'storage.qdrant_store'`

- [ ] **Step 3: Implement QdrantStore**

File: `src/storage/qdrant_store.py`

```python
"""Qdrant local-mode vector store for evidence embedding search."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


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
            PointStruct(id=id_, vector=vec.tolist(), payload=payload)
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
        hits = self._client.search(
            collection_name=self.collection,
            query_vector=query_vec.tolist(),
            limit=top_k,
        )
        return [(hit.id, hit.score, hit.payload or {}) for hit in hits]

    def count(self) -> int:
        info = self._client.get_collection(self.collection)
        return info.points_count
```

- [ ] **Step 4: Run test — expected PASS**

```bash
pytest tests/test_dense_retrieval.py::TestQdrantStore -v
```

Expected: PASS

- [ ] **Step 5: Add test for re-creation (idempotent upsert)**

Add to `TestQdrantStore` in `tests/test_dense_retrieval.py`:

```python
    def test_ensure_collection_is_idempotent(self, tmp_path):
        store = QdrantStore(path=str(tmp_path / "qdrant2"))
        store.ensure_collection(vector_size=8)
        store.ensure_collection(vector_size=8)  # does not raise
        assert store.count() == 0

    def test_upsert_updates_existing_point(self, tmp_path):
        store = QdrantStore(path=str(tmp_path / "qdrant3"))
        store.ensure_collection(vector_size=3)
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        store.upsert([("same_id", v, {"v": 1})])
        store.upsert([("same_id", v, {"v": 2})])
        assert store.count() == 1
```

- [ ] **Step 6: Run all QdrantStore tests**

```bash
pytest tests/test_dense_retrieval.py::TestQdrantStore -v
```

Expected: 3 passed

- [ ] **Step 7: Commit**

```bash
git add src/storage/qdrant_store.py tests/test_dense_retrieval.py
git commit -m "feat: add QdrantStore for local-mode vector search"
```

---

### Task 3: DenseRetriever

**Files:**
- Create: `src/retrieval/dense.py`
- Modify: `tests/test_dense_retrieval.py` (add TestDenseRetriever class)

- [ ] **Step 1: Write failing integration test for DenseRetriever**

Add to `tests/test_dense_retrieval.py`:

```python
from evidence import DocumentBlock, MoleculeCard
from retrieval.dense import DenseRetriever
from storage.local_store import LocalStore


class TestDenseRetriever:
    def test_index_and_search_returns_candidate_evidence(self, tmp_path):
        from unittest.mock import patch

        # Seed LocalStore
        store = LocalStore(base_dir=tmp_path)
        store.save_molecules(
            "paper_001",
            [
                MoleculeCard(
                    molecule_card_id="mol_001",
                    doc_id="paper_001",
                    names=["ethanol"],
                    raw_smiles="CCO",
                ),
                MoleculeCard(
                    molecule_card_id="mol_002",
                    doc_id="paper_001",
                    names=["benzene"],
                    raw_smiles="c1ccccc1",
                ),
            ],
        )

        qdrant = QdrantStore(path=str(tmp_path / "qdrant"), collection="test_dense")
        retriever = DenseRetriever(qdrant)

        # Index
        retriever.index(store)
        assert qdrant.count() == 2

        # Search for something semantically close to "alcohol" → should hit ethanol
        results = retriever.search("alcohol", top_k=2)
        assert len(results) >= 1
        assert results[0].evidence_type == "molecule"
        assert "ethanol" in (results[0].summary or "")
```

- [ ] **Step 2: Run test — expected FAIL**

```bash
pytest tests/test_dense_retrieval.py::TestDenseRetriever -v
```

Expected: `ModuleNotFoundError: No module named 'retrieval.dense'`

- [ ] **Step 3: Implement DenseRetriever**

File: `src/retrieval/dense.py`

```python
"""Dense (semantic) retrieval over local evidence via vector embeddings.

Indexes evidence text into a Qdrant collection using SentenceTransformer,
then searches by cosine similarity at query time.
"""

from __future__ import annotations

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
        self._qdrant.ensure_collection(vector_size=self._model.get_sentence_embedding_dimension())

    def index(self, evidence_store: LocalStore, doc_ids: list[str] | None = None) -> int:
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

            if etype == "molecule":
                results.append(
                    CandidateEvidence(
                        evidence_id=ev_id,
                        evidence_type="molecule",
                        summary=summary,
                        structured_slots={"doc_id": doc_id, "score": score},
                        confidence=min(1.0, max(0.0, score)),
                    )
                )
            elif etype == "reaction_event":
                results.append(
                    CandidateEvidence(
                        evidence_id=ev_id,
                        evidence_type="reaction_event",
                        summary=summary,
                        structured_slots={"doc_id": doc_id, "score": score},
                        confidence=min(1.0, max(0.0, score)),
                    )
                )
            elif etype == "fact":
                results.append(
                    CandidateEvidence(
                        evidence_id=ev_id,
                        evidence_type="fact",
                        summary=summary,
                        structured_slots={"doc_id": doc_id, "score": score},
                        confidence=min(1.0, max(0.0, score)),
                    )
                )
            else:
                results.append(
                    CandidateEvidence(
                        evidence_id=ev_id,
                        evidence_type="document_block",
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
```

- [ ] **Step 4: Run DenseRetriever test**

```bash
pytest tests/test_dense_retrieval.py::TestDenseRetriever -v -s
```

`s` = print model loading output. Expected: PASS (may take 5-15 seconds first run to load the embedding model)

**Note:** First run downloads `all-MiniLM-L6-v2` (~80MB) to `HF_HOME`. Ensure `HF_HOME` is set per our existing environment config.

- [ ] **Step 5: Commit**

```bash
git add src/retrieval/dense.py tests/test_dense_retrieval.py
git commit -m "feat: add DenseRetriever for semantic vector search"
```

---

### Task 4: dense_search convenience function

**Files:**
- Modify: `src/retrieval/dense.py` (add function at bottom)
- Modify: `tests/test_dense_retrieval.py` (add test)

- [ ] **Step 1: Add dense_search function to dense.py**

Append to `src/retrieval/dense.py`:

```python
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
    return retriever.search(query, top_k=top_k)
```

- [ ] **Step 2: Write test for dense_search**

Add to `TestDenseRetriever`:

```python
    def test_dense_search_convenience_function(self, tmp_path):
        store = LocalStore(base_dir=tmp_path)
        store.save_molecules(
            "paper_001",
            [
                MoleculeCard(
                    molecule_card_id="mol_001",
                    doc_id="paper_001",
                    names=["tetrahydrofuran"],
                    raw_smiles="C1CCOC1",
                ),
            ],
        )
        from retrieval.dense import dense_search

        qdrant = QdrantStore(path=str(tmp_path / "qdrant_conv"), collection="test_conv")
        retriever = DenseRetriever(qdrant)
        retriever.index(store)

        results = dense_search(
            "solvent", top_k=3,
            qdrant_store=qdrant,
        )
        assert len(results) >= 1
        assert "tetrahydrofuran" in (results[0].summary or "")

---

### Task 5: Integrate dense into RetrievalRouter

**Files:**
- Modify: `src/retrieval/router.py`
- Modify: `src/retrieval/__init__.py`
- Modify: `tests/test_retrieval.py` (add test)

- [ ] **Step 1: Write test for router dense channel**

Add to `tests/test_retrieval.py`:

```python
def test_router_includes_dense_channel_for_paper_local_question(tmp_path):
    store = seed_store(tmp_path)
    router = RetrievalRouter(store)
    package = router.retrieve("What is the reaction solvent?")
    # Paper local question → lexical + dense + provenance
    assert "lexical_retrieval" in package.retrieval_path
    # dense_retrieval should be in path even if no dense index yet
    # (the router adds it; actual results may be empty if index not built)
    assert "dense_retrieval" in package.retrieval_path
```

- [ ] **Step 2: Run test — expected FAIL**

```bash
pytest tests/test_retrieval.py::test_router_includes_dense_channel_for_paper_local_question -v
```

Expected: `AssertionError: assert 'dense_retrieval' in ['lexical_retrieval', 'provenance_backtracking']`

- [ ] **Step 3: Modify router.py**

Replace the existing `else` block in `retrieve()` (lines 46-48) with:

```python
        else:
            path = ["lexical_retrieval", "dense_retrieval", "provenance_backtracking"]
            candidates = lexical_search(query, self.store, doc_ids=doc_ids, top_k=limit)
            try:
                from .dense import dense_search
                dense_candidates = dense_search(query, self.store, doc_ids=doc_ids, top_k=limit)
                candidates = _dedupe_candidates(list(candidates) + dense_candidates)[:limit]
            except Exception:
                pass  # dense unavailable (no index / model not loaded) — fall back to lexical only
```

The `try/except` ensures dense is optional — the router works with only lexical if the vector index hasn't been built yet.

- [ ] **Step 4: Run router test**

```bash
pytest tests/test_retrieval.py::test_router_includes_dense_channel_for_paper_local_question -v
```

Expected: PASS (dense channel in path, even if no dense results returned)

- [ ] **Step 5: Run all retrieval tests to check no regressions**

```bash
pytest tests/test_retrieval.py -v
```

Expected: all 6 existing tests pass, plus the new one = 7 passed

- [ ] **Step 6: Update __init__.py**

Add the dense export to `src/retrieval/__init__.py`:

Replace the file with:

```python
"""Local retrieval utilities for ChemEvoRAG Phase 1."""

from .dense import DenseRetriever, dense_search
from .entity import entity_search
from .lexical import lexical_search
from .lightrag_adapter import LightRAGAdapter, LightRAGEvidenceDocument
from .provenance import provenance_backtrack
from .reaction import reaction_event_search
from .router import RetrievalRouter

__all__ = [
    "RetrievalRouter",
    "LightRAGAdapter",
    "LightRAGEvidenceDocument",
    "DenseRetriever",
    "dense_search",
    "entity_search",
    "lexical_search",
    "provenance_backtrack",
    "reaction_event_search",
]
```

- [ ] **Step 7: Commit**

```bash
git add src/retrieval/router.py src/retrieval/__init__.py tests/test_retrieval.py
git commit -m "feat: integrate dense retrieval channel into RetrievalRouter"
```

---

### Task 6: Indexing script

**Files:**
- Create: `scripts/index_evidence.py`

- [ ] **Step 1: Write index_evidence.py**

```python
#!/usr/bin/env python
"""Build the Qdrant vector index from stored evidence JSON files.

Usage:
    conda activate chemevorag
    cd ChemEvoRAG_Phase1-main
    python scripts/index_evidence.py                    # all documents
    python scripts/index_evidence.py --doc-id paper_001  # single document
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from retrieval import DenseRetriever
from storage import LocalStore
from storage.qdrant_store import QdrantStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc-id", default=None, help="Index a single document instead of all.")
    parser.add_argument(
        "--base-dir",
        default=str(PROJECT_ROOT),
        help="Project base directory.",
    )
    parser.add_argument(
        "--qdrant-path",
        default=None,
        help="Qdrant data directory. Defaults to data/indexes/qdrant/",
    )
    args = parser.parse_args()

    base = Path(args.base_dir)
    qdrant_path = str(base / "data" / "indexes" / "qdrant") if args.qdrant_path is None else args.qdrant_path

    store = LocalStore(base_dir=base)
    qdrant = QdrantStore(path=qdrant_path)
    retriever = DenseRetriever(qdrant)

    doc_ids = [args.doc_id] if args.doc_id else None
    count = retriever.index(store, doc_ids=doc_ids)
    print(f"Indexed {count} evidence points into {qdrant_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Test indexing script with pubchem_seed**

```bash
python scripts/index_evidence.py --doc-id pubchem_seed
```

Expected: `Indexed 10 evidence points into data/indexes/qdrant` (5 molecules + 5 facts)

- [ ] **Step 3: Test indexing script with all docs**

```bash
python scripts/index_evidence.py
```

Expected: prints count > 0 (total evidence from pubchem_seed + any other docs)

- [ ] **Step 4: Commit**

```bash
git add scripts/index_evidence.py
git commit -m "feat: add index_evidence.py CLI to build Qdrant vector index"
```

---

### Task 7: Update environment.yml

- [ ] **Step 1: Add qdrant-client to environment.yml**

Edit `environment.yml`, add under pip packages:

```yaml
      - qdrant-client
```

- [ ] **Step 2: Commit**

```bash
git add environment.yml
git commit -m "chore: add qdrant-client to environment.yml dependencies"
```

---

### Task 8: Full integration verification

- [ ] **Step 1: Run all existing tests to verify no regressions**

```bash
pytest tests/ --ignore=tests/test_neo4j_identity_client.py -k "not build_evidence_script and not run_query_script and not roma_adapter_reports_availability" -v
```

Expected: all previously passing tests still pass, plus new dense tests.

- [ ] **Step 2: End-to-end dense search demo**

```bash
# Index pubchem seed evidence
python scripts/index_evidence.py --doc-id pubchem_seed

# Query via run_query (now uses dense channel automatically)
python scripts/run_query.py "What solvent was used?" --doc-id pubchem_seed
```

Expected: returns GroundedAnswer with ethanol or similar solvent evidence. `retrieval_path` includes `dense_retrieval`.

- [ ] **Step 3: Commit final integration checkpoint**

```bash
git add -A
git commit -m "test: full integration verification for dense retrieval"
```
