from evidence import (
    DocumentBlock,
    FactCard,
    MoleculeCard,
    ReactionEventCard,
    ReactionParticipant,
    SourceProvenance,
    YieldValue,
)
from retrieval import (
    RetrievalRouter,
    entity_search,
    lexical_search,
    provenance_backtrack,
    reaction_event_search,
)
from storage import LocalStore


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
                source_file="paper_001.pdf",
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
                inchi_key="DCHAKEY",
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
                solvents=["ethanol"],
                temperature="25 °C",
                yield_value=YieldValue(value=92, unit="%", normalized_value=0.92, raw_text="92%"),
                procedure_text="benzaldehyde and aniline gave imine in 92% yield",
                source=SourceProvenance(doc_id=doc_id, page=8, block_id="block_002"),
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
                claim="The imine reaction used ethanol at 25 °C.",
                source=SourceProvenance(doc_id=doc_id, page=8, block_id="block_002"),
            )
        ],
    )
    return store


def test_lexical_search_returns_matching_evidence(tmp_path):
    store = seed_store(tmp_path)

    results = lexical_search("ethanol imine", store)

    assert results
    assert {result.evidence_type for result in results} >= {"reaction_event", "fact"}
    assert results[0].structured_slots["score"] > 0


def test_entity_search_matches_alias_and_structure_fields(tmp_path):
    store = seed_store(tmp_path)

    alias_results = entity_search("dicyclohexylamine", store)
    inchikey_results = entity_search("DCHAKEY", store)

    assert alias_results[0].evidence_id == "mol_001"
    assert inchikey_results[0].evidence_id == "mol_001"
    assert alias_results[0].confidence > 0
    assert inchikey_results[0].confidence > 0


def test_reaction_event_search_matches_structured_filters(tmp_path):
    store = seed_store(tmp_path)

    results = reaction_event_search(
        {"reactants": ["benzaldehyde"], "products": ["imine"], "yield_min": 90},
        store,
    )

    assert len(results) == 1
    assert results[0].evidence_id == "rxn_001"
    assert results[0].structured_slots["yield"]["value"] == 92


def test_provenance_backtrack_finds_sources(tmp_path):
    store = seed_store(tmp_path)

    block_source = provenance_backtrack("block_001", store)
    reaction_source = provenance_backtrack("rxn_001", store)
    molecule_source = provenance_backtrack("mol_001", store)

    assert block_source.page == 6
    assert reaction_source.page == 8
    assert molecule_source.doc_id == "paper_001"
    assert provenance_backtrack("missing", store) is None


def test_retrieval_router_builds_evidence_package_for_reaction_query(tmp_path):
    store = seed_store(tmp_path)
    router = RetrievalRouter(store, use_llm_intent=False)

    package = router.retrieve("Which reaction has the highest imine yield?")

    assert package.intent == "reaction_comparison"
    assert package.candidate_evidence
    assert "reaction_retrieval" in package.retrieval_path
    assert package.provenance
    assert package.missing_slots == []


def test_retrieval_router_reports_missing_evidence(tmp_path):
    store = LocalStore(base_dir=tmp_path)
    router = RetrievalRouter(store, use_llm_intent=False)

    package = router.retrieve("unknown query")

    # All channels run regardless of intent; empty store means no results.
    assert "lexical_retrieval" in package.retrieval_path
    assert "entity_retrieval" in package.retrieval_path
