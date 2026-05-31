import pytest

from evidence import (
    DocumentBlock,
    FactCard,
    MoleculeCard,
    ReactionEventCard,
    SourceProvenance,
)
from storage import LocalStore


def test_save_and_load_parsed_payload(tmp_path):
    store = LocalStore(base_dir=tmp_path)
    payload = {
        "doc_id": "paper_001",
        "source_file": "paper_001.pdf",
        "status": "success",
        "blocks": [{"text": "hello"}],
    }

    path = store.save_parsed("paper_001", payload)
    loaded = store.load_parsed("paper_001")

    assert path == tmp_path / "data" / "parsed" / "paper_001.raw_extraction.json"
    assert loaded == payload


def test_save_and_load_blocks(tmp_path):
    store = LocalStore(base_dir=tmp_path)
    blocks = [
        DocumentBlock(
            block_id="block_001",
            doc_id="paper_001",
            block_type="procedure",
            text="Compound 7 was synthesized.",
            page=6,
        )
    ]

    path = store.save_blocks("paper_001", blocks)
    loaded = store.load_blocks("paper_001")

    assert path == tmp_path / "data" / "evidence" / "paper_001.blocks.json"
    assert loaded == blocks


def test_save_and_load_molecules(tmp_path):
    store = LocalStore(base_dir=tmp_path)
    molecules = [
        MoleculeCard(
            molecule_card_id="mol_card_001",
            doc_id="paper_001",
            names=["DCHA"],
            aliases=["dicyclohexylamine"],
            raw_smiles="C1CCC(CC1)NC2CCCCC2",
        )
    ]

    store.save_molecules("paper_001", molecules)
    loaded = store.load_molecules("paper_001")

    assert loaded == molecules
    assert loaded[0].names == ["DCHA"]


def test_save_and_load_reactions(tmp_path):
    store = LocalStore(base_dir=tmp_path)
    reactions = [
        ReactionEventCard(
            reaction_event_id="rxn_001",
            doc_id="paper_001",
            solvents=["ethanol"],
            source=SourceProvenance(doc_id="paper_001", page=6),
            confidence=0.88,
        )
    ]

    store.save_reactions("paper_001", reactions)
    loaded = store.load_reactions("paper_001")

    assert loaded == reactions
    assert loaded[0].source.page == 6


def test_save_and_load_facts(tmp_path):
    store = LocalStore(base_dir=tmp_path)
    facts = [
        FactCard(
            fact_card_id="fact_001",
            doc_id="paper_001",
            fact_type="reaction_condition",
            claim="The reaction used ethanol.",
            supporting_blocks=["block_001"],
        )
    ]

    store.save_facts("paper_001", facts)
    loaded = store.load_facts("paper_001")

    assert loaded == facts
    assert loaded[0].claim == "The reaction used ethanol."


def test_missing_file_raises_file_not_found(tmp_path):
    store = LocalStore(base_dir=tmp_path)

    with pytest.raises(FileNotFoundError):
        store.load_molecules("missing_doc")


def test_parsed_payload_must_be_json_object(tmp_path):
    store = LocalStore(base_dir=tmp_path)
    path = tmp_path / "data" / "parsed" / "paper_001.raw_extraction.json"
    path.parent.mkdir(parents=True)
    path.write_text("[]\n", encoding="utf-8")

    with pytest.raises(TypeError, match="must be a JSON object"):
        store.load_parsed("paper_001")

