import json
import subprocess
from pathlib import Path

from evidence import EvidenceBuilder, RawExtraction, RawExtractionItem, SourceProvenance
from storage import LocalStore


def test_evidence_builder_handles_empty_failed_extraction():
    extraction = RawExtraction(
        doc_id="paper_001",
        source_file="paper_001.pdf",
        parser_name="openchemie",
        status="failed",
        errors=["OpenChemIE unavailable"],
    )

    bundle = EvidenceBuilder().build(extraction)

    assert bundle.doc_id == "paper_001"
    assert bundle.blocks == []
    assert bundle.molecules == []
    assert bundle.reactions == []
    assert bundle.facts == []
    assert bundle.errors == ["OpenChemIE unavailable"]


def test_evidence_builder_builds_molecule_cards_from_raw_items():
    extraction = RawExtraction(
        doc_id="paper_001",
        source_file="paper_001.pdf",
        parser_name="openchemie",
        status="success",
        molecules=[
            RawExtractionItem(
                item_id="mol_item_001",
                doc_id="paper_001",
                item_type="molecule",
                source=SourceProvenance(doc_id="paper_001", page=1),
                payload={
                    "molecules": [
                        {
                            "text": "ethanol",
                            "smiles": "CCO",
                            "score": 0.91,
                        }
                    ]
                },
            )
        ],
    )

    bundle = EvidenceBuilder().build(extraction)

    assert len(bundle.molecules) == 1
    molecule = bundle.molecules[0]
    assert molecule.molecule_card_id == "mol_card_0001_01"
    assert molecule.names == ["ethanol"]
    assert molecule.aliases == ["ethanol"]
    assert molecule.raw_smiles == "CCO"
    assert molecule.confidence == 0.91


def test_evidence_builder_builds_reaction_cards_and_facts():
    extraction = RawExtraction(
        doc_id="paper_001",
        source_file="paper_001.pdf",
        parser_name="openchemie",
        status="success",
        reactions=[
            RawExtractionItem(
                item_id="rxn_item_001",
                doc_id="paper_001",
                item_type="reaction",
                source=SourceProvenance(doc_id="paper_001", page=3),
                payload={
                    "reactions": [
                        {
                            "reactants": [{"name": "benzaldehyde", "smiles": "O=CC1=CC=CC=C1"}],
                            "products": [{"name": "imine"}],
                            "conditions": [
                                {"text": ["solvent ethanol"]},
                                {"text": ["25 °C"]},
                            ],
                            "text": "The reaction gave 92% yield.",
                        }
                    ]
                },
            )
        ],
    )

    bundle = EvidenceBuilder().build(extraction)

    assert len(bundle.reactions) == 1
    reaction = bundle.reactions[0]
    assert reaction.reaction_event_id == "rxn_event_0001_01"
    assert reaction.reactants[0].name == "benzaldehyde"
    assert reaction.products[0].name == "imine"
    assert reaction.solvents == ["solvent ethanol"]
    assert reaction.temperature == "25 °C"
    assert reaction.yield_value.value == 92
    assert reaction.yield_value.normalized_value == 0.92
    assert reaction.evidence_completeness == 1.0
    assert bundle.facts[0].fact_type == "reaction_condition"
    assert bundle.facts[0].entities == ["rxn_event_0001_01"]


def test_evidence_builder_builds_blocks_and_block_facts():
    extraction = RawExtraction(
        doc_id="paper_001",
        source_file="paper_001.pdf",
        parser_name="openchemie",
        status="success",
        blocks=[
            RawExtractionItem(
                item_id="block_item_001",
                doc_id="paper_001",
                item_type="procedure",
                source=SourceProvenance(doc_id="paper_001", page=6),
                text="Compound 7 was synthesized under nitrogen.",
            ),
            RawExtractionItem(
                item_id="block_item_002",
                doc_id="paper_001",
                item_type="block",
                source=SourceProvenance(doc_id="paper_001", page=7),
                text="The product was purified by chromatography.",
            ),
        ],
    )

    bundle = EvidenceBuilder().build(extraction)

    assert [block.block_id for block in bundle.blocks] == ["block_0001", "block_0002"]
    assert bundle.blocks[0].next_block_id == "block_0002"
    assert bundle.blocks[1].prev_block_id == "block_0001"
    assert len(bundle.facts) == 2
    assert bundle.facts[0].fact_type == "procedure"
    assert bundle.facts[0].supporting_blocks == ["block_0001"]


def test_build_evidence_script_writes_all_evidence_files(tmp_path):
    store = LocalStore(base_dir=tmp_path)
    extraction = RawExtraction(
        doc_id="paper_001",
        source_file="paper_001.pdf",
        parser_name="openchemie",
        status="success",
        molecules=[
            RawExtractionItem(
                item_id="mol_item_001",
                doc_id="paper_001",
                item_type="molecule",
                payload={"molecules": [{"text": "ethanol", "smiles": "CCO"}]},
            )
        ],
    )
    store.save_parsed("paper_001", extraction.model_dump(mode="json"))

    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            "/root/miniconda3/bin/conda",
            "run",
            "-n",
            "chemevorag",
            "python",
            str(project_root / "scripts" / "build_evidence.py"),
            "--doc-id",
            "paper_001",
            "--base-dir",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["doc_id"] == "paper_001"
    assert (tmp_path / "data" / "evidence" / "paper_001.blocks.json").exists()
    assert (tmp_path / "data" / "evidence" / "paper_001.molecules.json").exists()
    assert (tmp_path / "data" / "evidence" / "paper_001.reactions.json").exists()
    assert (tmp_path / "data" / "evidence" / "paper_001.facts.json").exists()
    assert store.load_molecules("paper_001")[0].raw_smiles == "CCO"

