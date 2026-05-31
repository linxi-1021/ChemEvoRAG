"""Pydantic schemas for ChemEvoRAG Phase 1 evidence objects."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


BlockType = Literal[
    "title",
    "abstract",
    "section",
    "paragraph",
    "table",
    "figure",
    "caption",
    "procedure",
    "scheme",
    "reference",
    "supplementary",
    "unknown",
]

NormalizationStatus = Literal["success", "partial", "failed", "not_attempted"]

MoleculeMatchType = Literal["name", "alias", "smiles", "inchikey", "unknown"]

FactType = Literal[
    "property",
    "mechanism",
    "reaction_condition",
    "comparison",
    "claim",
    "procedure",
    "identity",
    "unknown",
]

IntentType = Literal[
    "entity_lookup",
    "alias_resolution",
    "property_query",
    "reaction_condition_query",
    "reaction_comparison",
    "synthesis_route_query",
    "mechanism_query",
    "structure_query",
    "substructure_query",
    "similarity_query",
    "paper_local_question",
    "cross_paper_question",
    "evidence_verification",
    "unknown",
]

ParserRunStatus = Literal["success", "partial", "failed"]

RawExtractionItemType = Literal[
    "block",
    "molecule",
    "reaction",
    "table",
    "figure",
    "caption",
    "procedure",
    "unknown",
]


class ChemEvoBaseModel(BaseModel):
    """Base model with strict-ish defaults and JSON-friendly behavior."""

    model_config = ConfigDict(
        populate_by_name=True,
        validate_assignment=True,
        extra="forbid",
    )


class SourceProvenance(ChemEvoBaseModel):
    """Location of evidence in an original document."""

    doc_id: str
    source_file: str | None = None
    page: int | None = None
    bbox: list[float] | None = Field(
        default=None,
        description="Bounding box as [x1, y1, x2, y2] in parser coordinates.",
    )
    block_id: str | None = None
    figure_id: str | None = None
    table_id: str | None = None
    section: str | None = None
    text_span: tuple[int, int] | None = None


class DocumentBlock(ChemEvoBaseModel):
    """Structured block extracted from a PDF or supporting document."""

    block_id: str
    doc_id: str
    block_type: BlockType = "unknown"
    text: str | None = None
    page: int | None = None
    bbox: list[float] | None = None
    section: str | None = None
    prev_block_id: str | None = None
    next_block_id: str | None = None
    source_file: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    raw_payload: dict[str, Any] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)


class RawExtractionItem(ChemEvoBaseModel):
    """A parser-normalized raw extraction item before evidence construction."""

    item_id: str
    doc_id: str
    item_type: RawExtractionItemType = "unknown"
    source: SourceProvenance | None = None
    text: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    errors: list[str] = Field(default_factory=list)


class RawExtraction(ChemEvoBaseModel):
    """Unified output from a parser adapter for one source document."""

    doc_id: str
    source_file: str
    parser_name: str
    status: ParserRunStatus
    blocks: list[RawExtractionItem] = Field(default_factory=list)
    molecules: list[RawExtractionItem] = Field(default_factory=list)
    reactions: list[RawExtractionItem] = Field(default_factory=list)
    tables: list[RawExtractionItem] = Field(default_factory=list)
    figures: list[RawExtractionItem] = Field(default_factory=list)
    raw_payload: dict[str, Any] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)


class MoleculeMention(ChemEvoBaseModel):
    """A molecule mention in text, table, figure, or caption."""

    mention: str
    provenance: SourceProvenance
    role: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class MoleculeImageSource(ChemEvoBaseModel):
    """Image source for a molecule candidate."""

    image_id: str | None = None
    figure_id: str | None = None
    provenance: SourceProvenance
    bbox: list[float] | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class MoleculeCard(ChemEvoBaseModel):
    """Normalized molecule or compound-level evidence object."""

    molecule_card_id: str
    doc_id: str
    local_ids: list[str] = Field(default_factory=list)
    names: list[str] = Field(default_factory=list)
    raw_smiles: str | None = None
    canonical_smiles: str | None = None
    inchi_key: str | None = None
    iupac_name: str | None = None
    aliases: list[str] = Field(default_factory=list)
    source_mentions: list[MoleculeMention] = Field(default_factory=list)
    source_images: list[MoleculeImageSource] = Field(default_factory=list)
    linked_elementkg_id: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    normalization_status: NormalizationStatus = "not_attempted"
    raw_payload: dict[str, Any] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)


class MoleculeIdentityCandidate(ChemEvoBaseModel):
    """Canonical molecule identity returned from an external identity source."""

    elementkg_id: str | None = None
    canonical_name: str | None = None
    aliases: list[str] = Field(default_factory=list)
    smiles: str | None = None
    inchi_key: str | None = None
    match_type: MoleculeMatchType = "unknown"
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    raw_node: dict[str, Any] = Field(default_factory=dict)
    raw_payload: dict[str, Any] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)


class ReactionParticipant(ChemEvoBaseModel):
    """A molecule or named substance participating in a reaction."""

    name: str | None = None
    smiles: str | None = None
    role: Literal["reactant", "product", "reagent", "catalyst", "solvent", "unknown"] = (
        "unknown"
    )
    linked_molecule_card_id: str | None = None
    amount: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class YieldValue(ChemEvoBaseModel):
    """Yield value extracted from reaction evidence."""

    value: float | None = None
    unit: str | None = "%"
    normalized_value: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Fractional yield, for example 0.92 for 92%.",
    )
    raw_text: str | None = None


class ReactionEventCard(ChemEvoBaseModel):
    """Reaction-level evidence assembled from text, tables, and figures."""

    reaction_event_id: str
    doc_id: str
    reaction_smiles: str | None = None
    reactants: list[ReactionParticipant] = Field(default_factory=list)
    products: list[ReactionParticipant] = Field(default_factory=list)
    reagents: list[str] = Field(default_factory=list)
    catalysts: list[str] = Field(default_factory=list)
    solvents: list[str] = Field(default_factory=list)
    temperature: str | None = None
    time: str | None = None
    yield_value: YieldValue | None = None
    procedure_text: str | None = None
    source: SourceProvenance | None = None
    supporting_block_ids: list[str] = Field(default_factory=list)
    supporting_table_ids: list[str] = Field(default_factory=list)
    supporting_figure_ids: list[str] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence_completeness: float | None = Field(default=None, ge=0.0, le=1.0)
    raw_payload: dict[str, Any] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)


class FactCard(ChemEvoBaseModel):
    """Claim, property, mechanism, identity, or condition evidence."""

    fact_card_id: str
    doc_id: str
    fact_type: FactType = "unknown"
    claim: str
    entities: list[str] = Field(default_factory=list)
    supporting_blocks: list[str] = Field(default_factory=list)
    supporting_figures: list[str] = Field(default_factory=list)
    source: SourceProvenance | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    raw_payload: dict[str, Any] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)


class CandidateEvidence(ChemEvoBaseModel):
    """Evidence item selected for answer generation."""

    evidence_id: str
    evidence_type: Literal["molecule", "reaction_event", "fact", "document_block"]
    summary: str | None = None
    structured_slots: dict[str, Any] = Field(default_factory=dict)
    source: SourceProvenance | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class EvidencePackage(ChemEvoBaseModel):
    """Evidence bundle passed from retrieval to solver."""

    query: str
    intent: IntentType = "unknown"
    resolved_entities: list[MoleculeCard] = Field(default_factory=list)
    candidate_evidence: list[CandidateEvidence] = Field(default_factory=list)
    provenance: list[SourceProvenance] = Field(default_factory=list)
    missing_slots: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    retrieval_path: list[str] = Field(default_factory=list)


class SupportingEvidence(ChemEvoBaseModel):
    """Evidence cited in the final answer."""

    evidence_id: str
    evidence_type: str
    evidence: str
    source: SourceProvenance | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class GroundedAnswer(ChemEvoBaseModel):
    """Final structured answer returned to users."""

    answer: str
    supporting_evidence: list[SupportingEvidence] = Field(default_factory=list)
    provenance: list[SourceProvenance] = Field(default_factory=list)
    involved_entities: list[str] = Field(default_factory=list)
    retrieval_path: list[str] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    uncertainty: str | None = None
    raw_payload: dict[str, Any] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
