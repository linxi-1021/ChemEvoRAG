"""Local retrieval utilities for ChemEvoRAG Phase 1."""

from .entity import entity_search
from .lexical import lexical_search
from .lightrag_adapter import LightRAGAdapter, LightRAGEvidenceDocument
from .provenance import provenance_backtrack
from .reaction import reaction_event_search
from .router import RetrievalRouter
from .structure import structure_search

# Dense imports are lazy — sentence_transformers may not be installed
try:
    from .dense import DenseRetriever, dense_search
except ImportError:
    DenseRetriever = None  # type: ignore[assignment,misc]
    dense_search = None  # type: ignore[assignment]

__all__ = [
    "RetrievalRouter",
    "DenseRetriever",
    "dense_search",
    "LightRAGAdapter",
    "LightRAGEvidenceDocument",
    "entity_search",
    "lexical_search",
    "provenance_backtrack",
    "reaction_event_search",
    "structure_search",
]
