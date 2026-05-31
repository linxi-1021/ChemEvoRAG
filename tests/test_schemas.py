from evidence import (
    EvidencePackage,
    GroundedAnswer,
    MoleculeCard,
    ReactionEventCard,
    SourceProvenance,
)


def test_core_evidence_schemas_can_be_created():
    source = SourceProvenance(
        doc_id="paper_001",
        source_file="data/raw_pdfs/paper_001.pdf",
        page=6,
        block_id="block_0032",
    )

    molecule = MoleculeCard(
        molecule_card_id="mol_card_0001",
        doc_id="paper_001",
        names=["DCHA"],
        aliases=["dicyclohexylamine"],
        confidence=0.92,
    )

    reaction = ReactionEventCard(
        reaction_event_id="rxn_event_0001",
        doc_id="paper_001",
        source=source,
        supporting_block_ids=["block_0032"],
        confidence=0.88,
        evidence_completeness=0.81,
    )

    package = EvidencePackage(
        query="Which reaction has the highest yield?",
        intent="reaction_comparison",
        resolved_entities=[molecule],
        retrieval_path=["reaction_event_retrieval", "provenance_backtracking"],
    )

    answer = GroundedAnswer(
        answer="Reaction rxn_event_0001 has the highest extracted yield.",
        provenance=[source],
        involved_entities=[molecule.molecule_card_id, reaction.reaction_event_id],
        retrieval_path=package.retrieval_path,
        confidence=0.82,
    )

    assert package.intent == "reaction_comparison"
    assert answer.provenance[0].page == 6
    assert reaction.evidence_completeness == 0.81

