"""Tests for Dense Retrieval — QdrantStore and DenseRetriever."""

import numpy as np
from evidence import MoleculeCard
from storage.local_store import LocalStore
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
        # IDs are UUID5-hashed from the string keys
        assert results[0][1] > 0.9  # cosine similarity near 1.0
        assert results[0][2]["text"] == "ethanol"
        assert results[1][2]["text"] == "benzene"

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


class TestDenseRetriever:
    def test_index_and_search_returns_candidate_evidence(self, tmp_path):
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

        qdrant = QdrantStore(path=str(tmp_path / "qdrant_dr"), collection="test_dr")
        from retrieval.dense import DenseRetriever

        retriever = DenseRetriever(qdrant)
        retriever.index(store)
        assert qdrant.count() == 2

        results = retriever.search("alcohol", top_k=2)
        assert len(results) >= 1
        assert results[0].evidence_type == "molecule"
        assert "ethanol" in (results[0].summary or "")

    def test_dense_search_convenience_function(self, tmp_path):
        store = LocalStore(base_dir=tmp_path)
        store.save_molecules(
            "paper_001",
            [
                MoleculeCard(
                    molecule_card_id="mol_thf",
                    doc_id="paper_001",
                    names=["tetrahydrofuran"],
                    raw_smiles="C1CCOC1",
                ),
            ],
        )
        from retrieval.dense import dense_search

        qdrant = QdrantStore(path=str(tmp_path / "qdrant_conv"), collection="test_conv")
        from retrieval.dense import DenseRetriever

        retriever = DenseRetriever(qdrant)
        retriever.index(store)

        results = dense_search("solvent", top_k=3, qdrant_store=qdrant)
        assert len(results) >= 1
        assert "tetrahydrofuran" in (results[0].summary or "")
