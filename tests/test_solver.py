import json
import subprocess
from pathlib import Path

from evidence import MoleculeCard
from retrieval import RetrievalRouter
from solver import ChemRAGSolver, LLMChemSolver, ROMAAdapter
from storage import LocalStore


def test_solver_returns_uncertainty_when_no_evidence(tmp_path):
    solver = ChemRAGSolver(LocalStore(base_dir=tmp_path))

    answer = solver.answer("What is DCHA?")

    assert answer.confidence == 0.0
    assert answer.supporting_evidence == []
    assert "No candidate evidence" in answer.uncertainty


def test_solver_answers_from_local_retrieved_evidence(tmp_path):
    store = LocalStore(base_dir=tmp_path)
    store.save_molecules(
        "paper_001",
        [
            MoleculeCard(
                molecule_card_id="mol_001",
                doc_id="paper_001",
                names=["DCHA"],
                aliases=["dicyclohexylamine"],
            )
        ],
    )
    solver = ChemRAGSolver(store)

    answer = solver.answer("What is DCHA?")

    assert answer.supporting_evidence
    assert answer.involved_entities == ["mol_001"]
    assert "DCHA" in answer.answer
    assert "entity_retrieval" in answer.retrieval_path


def test_roma_adapter_reports_availability():
    adapter = ROMAAdapter("/root/ROMA")

    assert adapter.available() is True


class TestLLMChemSolver:
    def test_llm_solver_falls_back_to_deterministic_without_api_key(self, tmp_path):
        from solver import LLMChemSolver

        store = LocalStore(base_dir=tmp_path)
        store.save_molecules(
            "paper_001",
            [
                MoleculeCard(
                    molecule_card_id="mol_001",
                    doc_id="paper_001",
                    names=["ethanol"],
                    canonical_smiles="CCO",
                )
            ],
        )

        solver = LLMChemSolver(api_key="")  # no valid key
        package = RetrievalRouter(store, use_llm_intent=False).retrieve("ethanol")
        answer = solver.answer_from_package(package)

        # Falls back to deterministic — note should mention LLM unavailable
        assert answer.raw_payload.get("note", "").lower().find("llm unavailable") >= 0

    def test_llm_solver_falls_back_with_no_evidence(self, tmp_path):
        from solver import LLMChemSolver

        store = LocalStore(base_dir=tmp_path)
        store.save_molecules(
            "paper_001",
            [
                MoleculeCard(
                    molecule_card_id="mol_001",
                    doc_id="paper_001",
                    names=["DCHA"],
                    aliases=["dicyclohexylamine"],
                )
            ],
        )
        solver = LLMChemSolver(api_key="")
        package = RetrievalRouter(store, use_llm_intent=False).retrieve("What is DCHA?")
        answer = solver.answer_from_package(package)

        assert answer.supporting_evidence
        assert "DCHA" in answer.answer
        # Without API key, falls back to deterministic — answer mentions "local molecule identity"
        assert "molecule identity" in answer.answer.lower()


def test_run_query_script_outputs_grounded_answer_json(tmp_path):
    store = LocalStore(base_dir=tmp_path)
    store.save_molecules(
        "paper_001",
        [
            MoleculeCard(
                molecule_card_id="mol_001",
                doc_id="paper_001",
                names=["DCHA"],
                aliases=["dicyclohexylamine"],
            )
        ],
    )
    project_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [
            "/root/miniconda3/bin/conda",
            "run",
            "-n",
            "chemevorag",
            "python",
            str(project_root / "scripts" / "run_query.py"),
            "What is DCHA?",
            "--base-dir",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["supporting_evidence"][0]["evidence_id"] == "mol_001"
    assert payload["confidence"] > 0
