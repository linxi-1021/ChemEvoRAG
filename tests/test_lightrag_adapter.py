from evidence import DocumentBlock, FactCard, MoleculeCard, ReactionEventCard, ReactionParticipant, SourceProvenance
from retrieval import LightRAGAdapter
from storage import LocalStore


class FakeRAGClient:
    def __init__(self):
        self.inserted = []

    def insert(self, text):
        self.inserted.append(text)


def seed_store(tmp_path):
    store = LocalStore(base_dir=tmp_path)
    doc_id = "paper_001"
    store.save_blocks(
        doc_id,
        [
            DocumentBlock(
                block_id="block_001",
                doc_id=doc_id,
                block_type="procedure",
                text="Compound 7 was synthesized in ethanol.",
                page=6,
            )
        ],
    )
    store.save_molecules(
        doc_id,
        [
            MoleculeCard(
                molecule_card_id="mol_001",
                doc_id=doc_id,
                names=["DCHA"],
                aliases=["dicyclohexylamine"],
                raw_smiles="C1CCC(CC1)NC2CCCCC2",
            )
        ],
    )
    store.save_reactions(
        doc_id,
        [
            ReactionEventCard(
                reaction_event_id="rxn_001",
                doc_id=doc_id,
                reactants=[ReactionParticipant(name="benzaldehyde", role="reactant")],
                products=[ReactionParticipant(name="imine", role="product")],
                procedure_text="benzaldehyde gave imine in ethanol",
                source=SourceProvenance(doc_id=doc_id, page=8),
            )
        ],
    )
    store.save_facts(
        doc_id,
        [
            FactCard(
                fact_card_id="fact_001",
                doc_id=doc_id,
                fact_type="reaction_condition",
                claim="The reaction used ethanol.",
            )
        ],
    )
    return store


def test_lightrag_adapter_builds_documents_and_writes_manifest(tmp_path):
    store = seed_store(tmp_path)
    fake_rag = FakeRAGClient()
    adapter = LightRAGAdapter(store, rag_client=fake_rag)

    documents = adapter.index()

    assert len(documents) == 4
    assert adapter.index_path.exists()
    assert len(fake_rag.inserted) == 4
    assert "evidence_id: rxn_001" in "\n".join(fake_rag.inserted)


def test_lightrag_adapter_query_returns_matching_documents(tmp_path):
    store = seed_store(tmp_path)
    adapter = LightRAGAdapter(store)
    adapter.index()

    results = adapter.query("ethanol imine")

    assert results
    assert results[0].evidence_id in {"rxn_001", "fact_001", "block_001"}

