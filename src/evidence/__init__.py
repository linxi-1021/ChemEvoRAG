"""Evidence data contracts for ChemEvoRAG Phase 1."""

from .builder import EvidenceBundle, EvidenceBuilder
from .schemas import (
    CandidateEvidence,
    DocumentBlock,
    EvidencePackage,
    FactCard,
    GroundedAnswer,
    MoleculeCard,
    MoleculeIdentityCandidate,
    MoleculeImageSource,
    MoleculeMention,
    RawExtraction,
    RawExtractionItem,
    ReactionEventCard,
    ReactionParticipant,
    SourceProvenance,
    SupportingEvidence,
    YieldValue,
)

__all__ = [
    "CandidateEvidence",
    "DocumentBlock",
    "EvidenceBuilder",
    "EvidenceBundle",
    "EvidencePackage",
    "FactCard",
    "GroundedAnswer",
    "MoleculeCard",
    "MoleculeIdentityCandidate",
    "MoleculeImageSource",
    "MoleculeMention",
    "RawExtraction",
    "RawExtractionItem",
    "ReactionEventCard",
    "ReactionParticipant",
    "SourceProvenance",
    "SupportingEvidence",
    "YieldValue",
]
