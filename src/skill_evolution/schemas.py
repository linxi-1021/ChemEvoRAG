"""Pydantic models for Skill YAML schema and Prompt Registry entries."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pydantic
from pydantic import BaseModel, ConfigDict, Field


class ChemEvoBaseModel(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        validate_assignment=True,
        extra="forbid",
    )


# --- Skill YAML sub-models ---

class Trigger(ChemEvoBaseModel):
    intent: str
    confidence_threshold: float = 0.7
    aliases: list[str] = Field(default_factory=list)
    required_signals: list[str] = Field(default_factory=list)
    negative_intents: list[str] = Field(default_factory=list)


class Scope(ChemEvoBaseModel):
    supported_properties: list[str] = Field(default_factory=list)
    supported_units: list[str] = Field(default_factory=list)
    unsupported_properties: list[str] = Field(default_factory=list)
    domain_constraints: list[str] = Field(default_factory=list)


class Dependencies(ChemEvoBaseModel):
    prompts: list[str] = Field(default_factory=list)
    indexes: list[str] = Field(default_factory=list)
    validators: list[str] = Field(default_factory=list)
    core_modules: list[str] = Field(default_factory=list)


class Interface(ChemEvoBaseModel):
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)
    normalized_output: dict[str, Any] = Field(default_factory=dict)


class TopKConfig(ChemEvoBaseModel):
    entity_search: int = 8
    lexical_search: int = 10
    reaction_event_search: int = 12
    dense_search: int = 10
    structure_search: int = 8
    final_rerank: int = 15


class BoostWeight(ChemEvoBaseModel):
    weight: float
    rationale: str = ""


class RetrievalRouting(ChemEvoBaseModel):
    primary_channels: list[str] = Field(default_factory=list)
    top_k: TopKConfig = Field(default_factory=TopKConfig)
    boost_weights: dict[str, BoostWeight | float] = Field(default_factory=dict)


class QueryRewrite(ChemEvoBaseModel):
    enabled: bool = True
    max_variants: int = 3
    diversity_threshold: float = 0.8
    preserve_original_query: bool = True
    fallback_terms: list[str] = Field(default_factory=list)


class EvidenceExpansion(ChemEvoBaseModel):
    enabled: bool = True
    expand_from_block_neighbors: bool = True
    expand_from_molecule_mentions: bool = True
    max_expanded_blocks: int = 5


class Assessment(ChemEvoBaseModel):
    system_prompt_ref: str = "EVIDENCE_ASSESSMENT_SYSTEM"
    require_numeric_extraction: bool = True
    require_citation_check: bool = True
    failure_labels: list[str] = Field(default_factory=list)


class AnswerGeneration(ChemEvoBaseModel):
    system_prompt_ref: str = "ANSWER_GENERATION_SYSTEM"
    max_evidence_items: int = 15
    require_grounded_answer: bool = True
    uncertainty_policy: str = "state_uncertainty_when_evidence_is_partial"


class Strategy(ChemEvoBaseModel):
    retrieval_routing: RetrievalRouting = Field(default_factory=RetrievalRouting)
    query_rewrite: QueryRewrite = Field(default_factory=QueryRewrite)
    evidence_expansion: EvidenceExpansion = Field(default_factory=EvidenceExpansion)
    assessment: Assessment = Field(default_factory=Assessment)
    answer_generation: AnswerGeneration = Field(default_factory=AnswerGeneration)


class Template(ChemEvoBaseModel):
    name: str
    description: str = ""
    applicable_when: dict[str, Any] = Field(default_factory=dict)
    not_applicable_when: list[str] = Field(default_factory=list)
    slots: list[str] = Field(default_factory=list)
    examples: list[dict[str, Any]] = Field(default_factory=list)


class Evaluation(ChemEvoBaseModel):
    metrics: list[str] = Field(default_factory=list)
    pass_criteria: dict[str, float] = Field(default_factory=dict)
    regression_cases_ref: str = ""


class FailureToPatchEntry(ChemEvoBaseModel):
    target_paths: list[str] = Field(default_factory=list)
    allowed_operations: list[str] = Field(default_factory=list)
    allowed_fields: list[str] = Field(default_factory=list)


class MutationPolicy(ChemEvoBaseModel):
    allow_prompt_mutation: bool = True
    allow_policy_mutation: bool = True
    allow_template_addition: bool = True
    allow_template_merge: bool = True
    require_regression_test: bool = True
    require_human_approval_for: list[str] = Field(default_factory=list)


class Evolution(ChemEvoBaseModel):
    enabled: bool = True
    mutable_paths: list[str] = Field(default_factory=list)
    frozen_paths: list[str] = Field(default_factory=list)
    mutation_policy: MutationPolicy = Field(default_factory=MutationPolicy)
    failure_to_patch_mapping: dict[str, FailureToPatchEntry] = Field(default_factory=dict)

    @pydantic.model_validator(mode="after")
    def _check_no_overlap(self) -> Evolution:
        mutable_set = set(self.mutable_paths)
        frozen_set = set(self.frozen_paths)
        overlap = mutable_set & frozen_set
        if overlap:
            raise ValueError(
                f"mutable_paths and frozen_paths overlap: {overlap}"
            )
        return self


class SchemaValidation(ChemEvoBaseModel):
    before_patch: bool = True
    after_patch: bool = True
    reject_on_schema_error: bool = True


class RiskLevel(ChemEvoBaseModel):
    allow_auto_promote_after_regression: bool = False
    require_regression_pass: bool = True
    require_composition_validation: bool = False
    require_human_approval: bool = False


class RiskPolicy(ChemEvoBaseModel):
    low: RiskLevel = Field(default_factory=lambda: RiskLevel(allow_auto_promote_after_regression=True))
    medium: RiskLevel = Field(default_factory=lambda: RiskLevel(require_composition_validation=True))
    high: RiskLevel = Field(default_factory=lambda: RiskLevel(require_human_approval=True))


class Rollback(ChemEvoBaseModel):
    snapshot_before_apply: bool = True
    rollback_on_score_drop: bool = True
    rollback_scope: str = "last_promoted_patch_batch"


class Safety(ChemEvoBaseModel):
    elementkg_modification: str = "forbidden"
    max_patches_per_round: int = 2
    consecutive_fail_limit: int = 3
    patch_confidence_threshold: float = 0.75
    uncertain_patch_queue: bool = True
    schema_validation: SchemaValidation = Field(default_factory=SchemaValidation)
    semantic_invariants: list[str] = Field(default_factory=list)
    risk_policy: RiskPolicy = Field(default_factory=RiskPolicy)
    rollback: Rollback = Field(default_factory=Rollback)


class ChangelogEntry(ChemEvoBaseModel):
    version: str
    round: str
    change_type: str
    reason: str
    affected_paths: list[str] = Field(default_factory=list)


class Provenance(ChemEvoBaseModel):
    created_from: list[str] = Field(default_factory=lambda: ["manual_seed"])
    last_modified_by: str = ""
    last_modified_reason: str = ""


class ExternalCaseRefs(ChemEvoBaseModel):
    success_cases: str = ""
    failure_cases: str = ""
    regression_cases: str = ""


class Cases(ChemEvoBaseModel):
    representative: list[dict[str, Any]] = Field(default_factory=list)
    external_case_refs: ExternalCaseRefs = Field(default_factory=ExternalCaseRefs)


class Lifecycle(ChemEvoBaseModel):
    created_at: str = ""
    updated_at: str = ""
    owner: str = "EVOChemRAG"


class SkillYAML(ChemEvoBaseModel):
    """Top-level Skill YAML schema."""
    schema_version: str = "1.0"
    name: str
    skill_version: str = "1.0.0"
    status: str = "stable"
    description: str = ""
    lifecycle: Lifecycle = Field(default_factory=Lifecycle)
    trigger: Trigger = Field(default_factory=lambda: Trigger(intent="unknown"))
    scope: Scope = Field(default_factory=Scope)
    dependencies: Dependencies = Field(default_factory=Dependencies)
    interface: Interface = Field(default_factory=Interface)
    strategy: Strategy = Field(default_factory=Strategy)
    templates: list[Template] = Field(default_factory=list)
    evaluation: Evaluation = Field(default_factory=Evaluation)
    evolution: Evolution = Field(default_factory=Evolution)
    safety: Safety = Field(default_factory=Safety)
    cases: Cases = Field(default_factory=Cases)
    provenance: Provenance = Field(default_factory=Provenance)
    changelog: list[ChangelogEntry] = Field(default_factory=list)


class PromptEntry(ChemEvoBaseModel):
    """Schema for a Prompt Registry YAML file."""
    prompt_ref: str
    version: str = "1.0.0"
    status: str = "stable"
    created_from: str = ""
    created_by: str = "manual_seed"
    created_at: str = ""
    compatible_skills: list[str] = Field(default_factory=list)
    content: str
    diff_summary: str = ""
    validation: dict[str, Any] = Field(default_factory=dict)
    hash: str = ""
