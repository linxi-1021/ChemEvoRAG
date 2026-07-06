"""Shared type definitions for the ChemEvoRAG Skill Evolution module.

Extracted from attribution.py and kept separate so that the retained
modules (patch, apply, validation) do not depend on the archived
attribution pipeline.

These are the types that patch.py / apply.py / validation.py import.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

try:
    from pydantic import BaseModel, ConfigDict, Field
except ImportError:
    raise ImportError("pydantic is required for skill_evolution.types")


# ── Enums ──────────────────────────────────────────────────────────────────


class FailureType(str, Enum):
    ENTITY_MISS = "entity_miss"
    ALIAS_MISS = "alias_miss"
    LOCAL_ID_MISS = "local_id_miss"
    TOOL_ERROR = "tool_error"
    ROUTING_ERROR = "routing_error"
    PLANNER_ERROR = "planner_error"
    GRAPH_EXPANSION_ERROR = "graph_expansion_error"
    ASSESSMENT_FALSE_NEG = "assessment_false_negative"
    TABLE_EXTRACTION_ERROR = "table_extraction_error"
    TABLE_REASONING_ERROR = "table_reasoning_error"
    GENERATION_ERROR = "generation_error"
    UNKNOWN_FAILURE = "unknown_failure"

    @property
    def is_patchable(self) -> bool:
        return self not in (
            FailureType.TOOL_ERROR,
            FailureType.UNKNOWN_FAILURE,
            FailureType.TABLE_EXTRACTION_ERROR,
        )

    @property
    def mutation_dimension(self) -> str:
        _map = {
            FailureType.ENTITY_MISS: "planning",
            FailureType.ALIAS_MISS: "planning",
            FailureType.LOCAL_ID_MISS: "planning",
            FailureType.ROUTING_ERROR: "routing",
            FailureType.PLANNER_ERROR: "planning",
            FailureType.GRAPH_EXPANSION_ERROR: "graph_expansion",
            FailureType.ASSESSMENT_FALSE_NEG: "prompt",
            FailureType.TABLE_REASONING_ERROR: "prompt",
            FailureType.GENERATION_ERROR: "prompt",
            FailureType.TOOL_ERROR: "",
            FailureType.UNKNOWN_FAILURE: "",
        }
        return _map.get(self, "")


class AttributionSource(str, Enum):
    RULE = "rule"
    LLM = "llm"
    HYBRID = "hybrid"


# ── Pydantic models ────────────────────────────────────────────────────────


class _BaseModel(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True, validate_assignment=True, extra="forbid"
    )


class FailureRecord(_BaseModel):
    question_id: str
    intent: str
    skill_name: str = ""
    skill_version: str = "1.0.0"
    score: float
    primary_failure_type: FailureType = FailureType.UNKNOWN_FAILURE
    contributing_failure_types: list[FailureType] = Field(default_factory=list)
    attribution_source: AttributionSource = AttributionSource.RULE
    attribution_confidence: float = 0.0
    attribution_explanation: str = ""
    evidence_contained_answer: bool = False
    corpus_contained_answer: bool | None = None
    gold_answer: str = ""
    predicted_answer: str = ""
    retrieved_evidence_ids: list[str] = Field(default_factory=list)
    gold_evidence_ids: list[str] = Field(default_factory=list)
    cited_evidence_ids: list[str] = Field(default_factory=list)
    react_rounds_used: int = 1
    stop_reason: str = ""
    proposed_patch_targets: list[str] = Field(default_factory=list)
    evolution_action: str = "generate_patch_candidate"
    retrieval_rounds: list[dict[str, Any]] = Field(default_factory=list)
    channel_stats: dict[str, int] = Field(default_factory=dict)


class SuccessPattern(_BaseModel):
    intent: str
    pattern: str
    frequency: int = 0
    avg_score: float = 0.0
    supporting_question_ids: list[str] = Field(default_factory=list)
    candidate_template_action: str = "distill_or_refine_template"
    query_structure: str = ""
    evidence_type: str = ""
    answer_structure: str = ""
    trigger_conditions: dict[str, Any] = Field(default_factory=dict)
    reusable_template: str = ""


class CoverageGapRecord(_BaseModel):
    question_id: str
    intent: str
    score: float
    question: str = ""
    gold_answer: str = ""
    missing_entities: list[str] = Field(default_factory=list)
    corpus_search_hint: str = ""
    source_paper: str = ""
    detected_at: str = ""
    status: str = "open"


class PartialSuccessRecord(_BaseModel):
    question_id: str
    intent: str
    score: float
    partial_success_reason: str = ""
    primary_failure_type: FailureType | None = None
    evolution_action: str = "uncertain_queue"
