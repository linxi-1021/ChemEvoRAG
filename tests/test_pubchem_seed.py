import importlib.util
import json
from pathlib import Path

from solver import ChemRAGSolver
from storage import LocalStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "bootstrap_pubchem_seed.py"


def load_seed_module():
    spec = importlib.util.spec_from_file_location("bootstrap_pubchem_seed", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def fake_fetcher(name):
    cids = {
        "dicyclohexylamine": 8082,
        "ethanol": 702,
    }
    return {
        "PropertyTable": {
            "Properties": [
                {
                    "CID": cids[name],
                    "CanonicalSMILES": "C1CCC(CC1)NC2CCCCC2"
                    if name == "dicyclohexylamine"
                    else "CCO",
                    "IsomericSMILES": "C1CCC(CC1)NC2CCCCC2"
                    if name == "dicyclohexylamine"
                    else "CCO",
                    "InChIKey": "DCHAKEY" if name == "dicyclohexylamine" else "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
                    "IUPACName": name,
                    "MolecularFormula": "C12H23N" if name == "dicyclohexylamine" else "C2H6O",
                    "MolecularWeight": 181.32 if name == "dicyclohexylamine" else 46.07,
                }
            ]
        }
    }


def test_pubchem_seed_builds_local_graph_and_evidence(tmp_path):
    module = load_seed_module()
    store = LocalStore(base_dir=tmp_path)

    artifacts = module.bootstrap_pubchem_seed(
        store,
        seeds=module.DEFAULT_SEEDS[:2],
        fetcher=fake_fetcher,
    )

    molecules = store.load_molecules("pubchem_seed")
    facts = store.load_facts("pubchem_seed")
    graph = json.loads(Path(artifacts["kg_path"]).read_text(encoding="utf-8"))

    assert artifacts["molecule_count"] == 2
    assert len(molecules) == 2
    assert len(facts) == 2
    assert molecules[0].linked_elementkg_id == "pubchem:8082"
    assert any(node["id"] == "alias:DCHA" for node in graph["nodes"])
    assert any(edge["type"] == "alias_of" for edge in graph["edges"])


def test_pubchem_seed_can_answer_simple_identity_question(tmp_path):
    module = load_seed_module()
    store = LocalStore(base_dir=tmp_path)
    module.bootstrap_pubchem_seed(
        store,
        seeds=module.DEFAULT_SEEDS[:2],
        fetcher=fake_fetcher,
    )

    answer = ChemRAGSolver(store).answer("What is DCHA?", doc_ids=["pubchem_seed"])

    assert answer.supporting_evidence
    assert answer.involved_entities == ["pubchem_8082"]
    assert "dicyclohexylamine" in answer.answer
    assert "SMILES" in answer.answer
