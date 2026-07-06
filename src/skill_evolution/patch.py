"""Patch Schema and Patch Generator for ChemEvoRAG Skill Evolution.

Implements §5 of the Skill Evolution design spec:
  - PatchOperation / PatchStatus enums
  - PatchSchema Pydantic model
  - FailureCluster grouping
  - PatchGenerator: creates candidate patches from failure clusters
  - TemplateDistillation: distills templates from success patterns

Usage:
    from skill_evolution.patch import PatchGenerator, FailureCluster
    clusters = PatchGenerator.cluster_failures(report.failures)
    patches = PatchGenerator.generate_patches(clusters[0], skill_config)
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

try:
    from pydantic import BaseModel, ConfigDict, Field
except ImportError:
    raise ImportError("pydantic is required for skill_evolution.patch")

from .types import FailureRecord, FailureType, SuccessPattern, PartialSuccessRecord


# ---------------------------------------------------------------------------
# §5.1 Enums
# ---------------------------------------------------------------------------

class PatchOperation(str, Enum):
    ADD = "add"
    UPDATE = "update"
    DELETE = "delete"
    MERGE = "merge"
    REPLACE = "replace"


class PatchStatus(str, Enum):
    CANDIDATE = "candidate"
    SCHEMA_VALIDATED = "schema_validated"
    REGRESSION_VALIDATED = "regression_validated"
    PROMOTED = "promoted"
    APPLIED = "applied"
    MONITORED = "monitored"
    REJECTED = "rejected"
    MANUAL_REVIEW = "manual_review"
    ROLLED_BACK = "rolled_back"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class _BaseModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True, validate_assignment=True, extra="forbid")


class PromptArtifact(_BaseModel):
    prompt_ref: str = ""
    base_prompt_ref: str = ""
    prompt_role: str = ""
    version: str = "2.0"
    prompt_filename: str = ""
    created_from: str = ""
    diff_summary: str = ""
    registry_path: str = ""
    content: str = ""
    change_summary: list[str] = Field(default_factory=list)
    preserved_constraints: list[str] = Field(default_factory=list)
    targeted_failure_ids: list[str] = Field(default_factory=list)
    validation_requirements: list[str] = Field(default_factory=list)


# Patch types for Skill Evolution
PATCH_TYPE_PROMPT_CONTENT_UPDATE = "prompt_content_update"
PATCH_TYPE_PROMPT_REF_UPDATE = "prompt_ref_update"
PATCH_TYPE_RETRIEVAL_STRATEGY_UPDATE = "retrieval_strategy_update"
PATCH_TYPE_QUERY_REWRITE_UPDATE = "query_rewrite_update"
PATCH_TYPE_EVIDENCE_EXPANSION_UPDATE = "evidence_expansion_update"
PATCH_TYPE_ANSWER_GENERATION_UPDATE = "answer_generation_update"
PATCH_TYPE_TEMPLATE_ADD = "template_add"
PATCH_TYPE_TEMPLATE_REWRITE = "template_rewrite"
PATCH_TYPE_TEMPLATE_MERGE = "template_merge"
PATCH_TYPE_FALLBACK_UPDATE = "fallback_update"
PATCH_TYPE_STOP_CONDITION_UPDATE = "stop_condition_update"

_ALL_PATCH_TYPES = {
    PATCH_TYPE_PROMPT_CONTENT_UPDATE,
    PATCH_TYPE_PROMPT_REF_UPDATE,
    PATCH_TYPE_RETRIEVAL_STRATEGY_UPDATE,
    PATCH_TYPE_QUERY_REWRITE_UPDATE,
    PATCH_TYPE_EVIDENCE_EXPANSION_UPDATE,
    PATCH_TYPE_ANSWER_GENERATION_UPDATE,
    PATCH_TYPE_TEMPLATE_ADD,
    PATCH_TYPE_TEMPLATE_REWRITE,
    PATCH_TYPE_TEMPLATE_MERGE,
    PATCH_TYPE_FALLBACK_UPDATE,
    PATCH_TYPE_STOP_CONDITION_UPDATE,
}


class PatchSchema(_BaseModel):
    """A candidate patch to be applied to a Skill YAML or Prompt Registry."""
    patch_id: str = ""
    patch_type: str = ""  # One of PATCH_TYPE_* constants
    skill_name: str
    skill_version: str = "1.0.0"
    target_file: str = ""  # e.g., "config/skills/reaction_condition_query.yaml"
    source_failure_ids: list[str] = Field(default_factory=list)
    targeted_failure_ids: list[str] = Field(default_factory=list)
    primary_failure_type: FailureType
    target_path: str
    operation: PatchOperation = PatchOperation.UPDATE
    old_value: Any = None
    new_value: Any = None
    current_value: dict[str, Any] = Field(default_factory=dict)
    proposed_value: dict[str, Any] = Field(default_factory=dict)
    prompt_artifacts: list[PromptArtifact] = Field(default_factory=list)
    rationale: str = ""
    expected_improvement: str = ""
    risk_level: str = "medium"
    confidence: float = 0.0
    status: PatchStatus = PatchStatus.CANDIDATE
    dependencies: list[str] = Field(default_factory=list)
    validation_status: str = "pending"


class FailureCluster(_BaseModel):
    """A group of failures sharing the same skill, failure_type, and target_path."""
    skill_name: str
    primary_failure_type: FailureType
    target_paths: list[str]
    failures: list[FailureRecord] = Field(default_factory=list)
    cluster_size: int = 0


# ---------------------------------------------------------------------------
# §5.3 Failure clustering
# ---------------------------------------------------------------------------

def _cluster_key(f: FailureRecord) -> tuple[str, str, tuple[str, ...]]:
    """Generate a cluster key from a failure record."""
    targets = tuple(f.proposed_patch_targets)
    return (f.skill_name, f.primary_failure_type.value, targets)


def cluster_failures(failures: list[FailureRecord]) -> list[FailureCluster]:
    """Group failures by skill_name + primary_failure_type + proposed_patch_targets.

    This prevents generating one patch per failure; instead, each cluster
    gets 1-3 patches that address the systemic issue.
    """
    clusters: dict[tuple, FailureCluster] = {}
    for f in failures:
        if f.attribution_confidence < 0.5:
            continue  # too uncertain to patch
        if f.primary_failure_type == FailureType.UNKNOWN_FAILURE:
            continue  # can't patch unknown failures
        if f.primary_failure_type == FailureType.TABLE_EXTRACTION_ERROR:
            continue  # parser issue, not patchable
        key = _cluster_key(f)
        if key not in clusters:
            clusters[key] = FailureCluster(
                skill_name=f.skill_name,
                primary_failure_type=f.primary_failure_type,
                target_paths=f.proposed_patch_targets,
            )
        clusters[key].failures.append(f)
        clusters[key].cluster_size = len(clusters[key].failures)
    return list(clusters.values())


# ---------------------------------------------------------------------------
# §5.1+§5.5 Patch generation (rule-based for MVP)
# ---------------------------------------------------------------------------

# Mapping from failure_type → allowed_fields for the primary target_path
_FAILURE_FIELD_MAP: dict[FailureType, dict[str, list[str]]] = {
    FailureType.ENTITY_MISS: {
        "strategy.query_rewrite": ["fallback_terms", "diversity_threshold"],
        "strategy.retrieval_routing": ["boost_weights"],
    },
    FailureType.ALIAS_MISS: {
        "strategy.query_rewrite": ["fallback_terms"],
        "strategy.retrieval_routing": ["primary_channels"],
    },
    FailureType.LOCAL_ID_MISS: {
        "strategy.query_rewrite": ["fallback_terms", "max_variants"],
    },
    FailureType.ROUTING_ERROR: {
        "strategy.retrieval_routing": ["primary_channels", "top_k", "boost_weights"],
    },
    FailureType.GRAPH_EXPANSION_ERROR: {
        "strategy.evidence_expansion": ["expand_from_block_neighbors", "max_expanded_blocks"],
        "strategy.retrieval_routing": ["top_k"],
    },
    FailureType.PLANNER_ERROR: {
        "strategy.query_rewrite": ["diversity_threshold"],
    },
    FailureType.ASSESSMENT_FALSE_NEG: {
        "strategy.assessment": ["system_prompt_ref", "require_numeric_extraction"],
    },
    FailureType.TABLE_REASONING_ERROR: {
        "strategy.assessment": ["system_prompt_ref"],
        "strategy.answer_generation": ["system_prompt_ref", "max_evidence_items"],
    },
    FailureType.GENERATION_ERROR: {
        "strategy.answer_generation": ["system_prompt_ref", "max_evidence_items"],
    },
}


def _get_nested_value(d: dict, path: str) -> Any:
    """Get value from nested dict by dot-separated path."""
    parts = path.split(".")
    current = d
    for p in parts:
        if isinstance(current, dict) and p in current:
            current = current[p]
        else:
            return None
    return current


def _set_nested_value(d: dict, path: str, value: Any) -> None:
    """Set value in nested dict by dot-separated path."""
    parts = path.split(".")
    current = d
    for p in parts[:-1]:
        if p not in current:
            current[p] = {}
        current = current[p]
    current[parts[-1]] = value


def generate_patches_for_cluster(
    cluster: FailureCluster,
    skill_config: dict,
) -> list[PatchSchema]:
    """Generate candidate patches for a failure cluster.

    Uses rule-based patch generation for common failure types.
    Returns 1-3 patches depending on failure type and severity.
    """
    failure_type = cluster.primary_failure_type
    skill_name = cluster.skill_name
    skill_version = skill_config.get("skill_version", "1.0.0")
    failure_ids = [f.question_id for f in cluster.failures]
    avg_confidence = (
        sum(f.attribution_confidence for f in cluster.failures) / len(cluster.failures)
        if cluster.failures else 0.0
    )

    patches: list[PatchSchema] = []

    if failure_type == FailureType.ASSESSMENT_FALSE_NEG:
        patches.extend(_generate_assessment_patches(
            cluster, skill_name, skill_version, failure_ids, avg_confidence, skill_config
        ))
    elif failure_type == FailureType.TABLE_REASONING_ERROR:
        patches.extend(_generate_table_reasoning_patches(
            cluster, skill_name, skill_version, failure_ids, avg_confidence, skill_config
        ))
    elif failure_type in (FailureType.ENTITY_MISS, FailureType.ALIAS_MISS, FailureType.LOCAL_ID_MISS):
        patches.extend(_generate_entity_patches(
            cluster, skill_name, skill_version, failure_ids, avg_confidence, skill_config
        ))
    elif failure_type == FailureType.ROUTING_ERROR:
        patches.extend(_generate_routing_patches(
            cluster, skill_name, skill_version, failure_ids, avg_confidence, skill_config
        ))
    elif failure_type == FailureType.GENERATION_ERROR:
        patches.extend(_generate_generation_patches(
            cluster, skill_name, skill_version, failure_ids, avg_confidence, skill_config
        ))

    # Assign patch IDs
    for i, p in enumerate(patches):
        if not p.patch_id:
            p.patch_id = f"{skill_name}_round1_patch{i+1}"

    return patches


# --- Specific patch generators per failure type ---

def _load_prompt_content(prompt_ref: str, prompts_dir: Path | None = None) -> str:
    """Load prompt content by prompt_ref from config/prompts/."""
    if prompts_dir is None:
        prompts_dir = Path(__file__).resolve().parents[2] / "config" / "prompts"
    if not prompts_dir.is_dir():
        return ""
    try:
        import yaml as _yaml
        for f in prompts_dir.glob("*.yaml"):
            data = _yaml.safe_load(f.read_text("utf-8"))
            if data and data.get("prompt_ref") == prompt_ref:
                return data.get("content", "")
    except Exception:
        pass
    return ""


def _generate_improved_assessment_content(
    source_content: str,
    failures: list,
) -> str:
    """Generate improved assessment prompt content based on failure analysis.

    Analyzes the failure patterns and appends targeted rules to the source prompt.
    Returns the full improved prompt content string.
    """
    if not source_content:
        return ""

    # Analyze failure patterns to determine what improvements are needed
    failure_signals: list[str] = []
    for f in failures:
        gold = getattr(f, "gold_answer", "") or ""
        predicted = getattr(f, "predicted_answer", "") or ""
        explanation = getattr(f, "attribution_explanation", "") or ""

        # Detect cross-compound confusion
        if "different compound" in explanation.lower() or "cross-compound" in explanation.lower():
            failure_signals.append("cross_compound")
        # Detect numeric extraction failures
        if any(c.isdigit() for c in gold) and "insufficient" in predicted.lower():
            failure_signals.append("numeric_extraction")
        # Detect table entry confusion
        if "entry" in explanation.lower() or "row" in explanation.lower():
            failure_signals.append("table_entry_confusion")

    # Deduplicate signals
    unique_signals = list(dict.fromkeys(failure_signals))

    if not unique_signals:
        # No specific signals — add general improvement
        unique_signals = ["general"]

    # Build improvement rules based on signals
    improvement_rules: list[str] = []

    if "cross_compound" in unique_signals:
        improvement_rules.append("""
CROSS-COMPOUND VERIFICATION (auto-added by Skill Evolution):
- Before declaring sufficient=true, verify that the evidence mentions the SAME
  compound/entity that the question asks about.
- If the question asks about compound "2c" but evidence only contains data for "2a",
  this is NOT sufficient. Do NOT use data from a different compound.
- Extract the compound identifier from the question and explicitly check each
  evidence item for that identifier.""")

    if "numeric_extraction" in unique_signals:
        improvement_rules.append("""
NUMERIC VALUE DETECTION (auto-added by Skill Evolution):
- Before declaring sufficient=false, carefully check if the evidence contains
  numeric values (yields, temperatures, percentages, ratios) that answer the question.
- A table cell containing "85" under a "Yield" column means 85% yield — this IS
  a valid numeric answer even without the percent sign.
- Extract exact numbers from evidence before deciding insufficiency.""")

    if "table_entry_confusion" in unique_signals:
        improvement_rules.append("""
TABLE ENTRY MATCHING (auto-added by Skill Evolution):
- When the question references a specific entry number (e.g., "entry 5"), carefully
  match that entry number to the correct table row.
- Do NOT confuse data from different entries or rows.
- If the question asks which entry has the highest value, compare ALL visible
  entries before concluding.""")

    if "general" in unique_signals:
        improvement_rules.append("""
GENERAL ASSESSMENT IMPROVEMENT (auto-added by Skill Evolution):
- Before declaring sufficient=false, re-read each evidence item carefully.
- Check if the answer can be derived by combining information from multiple items.
- Only declare insufficient if NO combination of evidence items contains the
  information needed to answer the question.""")

    # Append rules to source content
    rules_text = "\n".join(improvement_rules)
    improved = source_content.rstrip() + "\n" + rules_text + "\n"

    return improved


def _generate_assessment_patches(
    cluster, skill_name, skill_version, failure_ids, confidence, skill_config
) -> list[PatchSchema]:
    """Generate patches for assessment_false_negative.

    These failures mean the assessment LLM had evidence but said insufficient.
    The most effective patch is to enhance the assessment prompt.
    """
    current_prompt_ref = (
        skill_config.get("strategy", {}).get("assessment", {}).get("system_prompt_ref", "")
    )
    new_prompt_ref = f"{current_prompt_ref}_V2" if current_prompt_ref else "EVIDENCE_ASSESSMENT_SYSTEM_V2"

    # Load source prompt content and generate improved version
    source_content = _load_prompt_content(current_prompt_ref)
    improved_content = _generate_improved_assessment_content(source_content, cluster.failures)

    # Determine improvement summary from failure analysis
    diff_parts = []
    for f in cluster.failures:
        explanation = getattr(f, "attribution_explanation", "") or ""
        if "numeric" in explanation.lower() or "value" in explanation.lower():
            diff_parts.append("numeric extraction")
        if "table" in explanation.lower() or "entry" in explanation.lower():
            diff_parts.append("table entry matching")
        if "compound" in explanation.lower() or "entity" in explanation.lower():
            diff_parts.append("cross-compound verification")
    diff_summary = "Enhanced assessment: " + ", ".join(set(diff_parts)) if diff_parts else "General assessment improvement"

    patch = PatchSchema(
        skill_name=skill_name,
        skill_version=skill_version,
        source_failure_ids=failure_ids,
        primary_failure_type=FailureType.ASSESSMENT_FALSE_NEG,
        target_path="strategy.assessment",
        operation=PatchOperation.UPDATE,
        current_value={"system_prompt_ref": current_prompt_ref},
        proposed_value={"system_prompt_ref": new_prompt_ref},
        prompt_artifacts=[PromptArtifact(
            prompt_ref=new_prompt_ref,
            created_from=current_prompt_ref,
            diff_summary=diff_summary,
            registry_path=f"config/prompts/{new_prompt_ref.lower()}.yaml",
            content=improved_content,
        )],
        rationale="Evidence contained target values but assessment judged insufficient.",
        expected_improvement="Reduce false negatives in evidence assessment for numeric extraction.",
        risk_level="medium",
        confidence=confidence,
    )
    return [patch]


def _generate_table_reasoning_patches(
    cluster, skill_name, skill_version, failure_ids, confidence, skill_config
) -> list[PatchSchema]:
    """Generate patches for table_reasoning_error.

    Table data was present but LLM misinterpreted it.
    Two patches: enhanced assessment prompt + increased max_evidence_items.
    """
    current_prompt_ref = (
        skill_config.get("strategy", {}).get("assessment", {}).get("system_prompt_ref", "")
    )
    new_prompt_ref = f"{current_prompt_ref}_TABLE_V2" if current_prompt_ref else "EVIDENCE_ASSESSMENT_SYSTEM_TABLE_V2"

    current_max = (
        skill_config.get("strategy", {}).get("answer_generation", {}).get("max_evidence_items", 10)
    )

    patch1 = PatchSchema(
        skill_name=skill_name,
        skill_version=skill_version,
        source_failure_ids=failure_ids,
        primary_failure_type=FailureType.TABLE_REASONING_ERROR,
        target_path="strategy.assessment",
        operation=PatchOperation.UPDATE,
        current_value={"system_prompt_ref": current_prompt_ref},
        proposed_value={"system_prompt_ref": new_prompt_ref},
        prompt_artifacts=[PromptArtifact(
            prompt_ref=new_prompt_ref,
            created_from=current_prompt_ref,
            diff_summary="Add explicit table row/column parsing instructions to prevent entry confusion.",
            registry_path=f"config/prompts/{new_prompt_ref.lower()}.yaml",
        )],
        rationale="Table entries were misread (wrong row, wrong column, or entry confusion).",
        expected_improvement="Improve table parsing accuracy for comparison and property queries.",
        risk_level="medium",
        confidence=confidence,
    )

    patch2 = PatchSchema(
        skill_name=skill_name,
        skill_version=skill_version,
        source_failure_ids=failure_ids,
        primary_failure_type=FailureType.TABLE_REASONING_ERROR,
        target_path="strategy.answer_generation",
        operation=PatchOperation.UPDATE,
        current_value={"max_evidence_items": current_max},
        proposed_value={"max_evidence_items": min(current_max + 3, 20)},
        rationale="Increase evidence limit to capture more context for table-based questions.",
        expected_improvement="More evidence items for LLM to compare table entries.",
        risk_level="low",
        confidence=confidence * 0.8,
    )

    return [patch1, patch2]


def _generate_entity_patches(
    cluster, skill_name, skill_version, failure_ids, confidence, skill_config
) -> list[PatchSchema]:
    """Generate patches for entity_miss/alias_miss/local_id_miss."""
    current_fallback = (
        skill_config.get("strategy", {}).get("query_rewrite", {}).get("fallback_terms", [])
    )
    # Add more context terms based on failure type
    new_terms = list(current_fallback)
    if cluster.primary_failure_type == FailureType.LOCAL_ID_MISS:
        new_terms.extend(["compound", "label", "number"])
    elif cluster.primary_failure_type == FailureType.ALIAS_MISS:
        new_terms.extend(["alias", "abbreviation", "abbreviation", "full name"])
    else:
        new_terms.extend(["compound", "molecule", "chemical"])
    new_terms = list(dict.fromkeys(new_terms))  # dedupe preserving order

    patch = PatchSchema(
        skill_name=skill_name,
        skill_version=skill_version,
        source_failure_ids=failure_ids,
        primary_failure_type=cluster.primary_failure_type,
        target_path="strategy.query_rewrite",
        operation=PatchOperation.UPDATE,
        current_value={"fallback_terms": current_fallback},
        proposed_value={"fallback_terms": new_terms},
        rationale=f"Expand fallback terms to improve recall for {cluster.primary_failure_type.value}.",
        expected_improvement="Better entity retrieval through expanded query variants.",
        risk_level="low",
        confidence=confidence,
    )
    return [patch]


def _generate_routing_patches(
    cluster, skill_name, skill_version, failure_ids, confidence, skill_config
) -> list[PatchSchema]:
    """Generate patches for routing_error — very few items retrieved."""
    current_top_k = (
        skill_config.get("strategy", {}).get("retrieval_routing", {}).get("top_k", {})
    )
    new_top_k = dict(current_top_k)
    for key in ["lexical_search", "reaction_event_search", "dense_search"]:
        if key in new_top_k:
            new_top_k[key] = min(new_top_k[key] + 5, 25)

    patch = PatchSchema(
        skill_name=skill_name,
        skill_version=skill_version,
        source_failure_ids=failure_ids,
        primary_failure_type=FailureType.ROUTING_ERROR,
        target_path="strategy.retrieval_routing",
        operation=PatchOperation.UPDATE,
        current_value={"top_k": current_top_k},
        proposed_value={"top_k": new_top_k},
        rationale="Increase retrieval limits to improve recall.",
        expected_improvement="More candidate evidence items for assessment.",
        risk_level="low",
        confidence=confidence,
    )
    return [patch]


def _generate_generation_patches(
    cluster, skill_name, skill_version, failure_ids, confidence, skill_config
) -> list[PatchSchema]:
    """Generate patches for generation_error — assessment passed but answer wrong."""
    current_max = (
        skill_config.get("strategy", {}).get("answer_generation", {}).get("max_evidence_items", 10)
    )
    patch = PatchSchema(
        skill_name=skill_name,
        skill_version=skill_version,
        source_failure_ids=failure_ids,
        primary_failure_type=FailureType.GENERATION_ERROR,
        target_path="strategy.answer_generation",
        operation=PatchOperation.UPDATE,
        current_value={"max_evidence_items": current_max},
        proposed_value={"max_evidence_items": min(current_max + 3, 20)},
        rationale="Increase evidence items to give LLM more context for answer generation.",
        expected_improvement="Better answers from richer evidence context.",
        risk_level="low",
        confidence=confidence,
    )
    return [patch]


# ---------------------------------------------------------------------------
# LLM-driven Patch Generation (Stage 4: Patch Cluster by skill)
# ---------------------------------------------------------------------------

def generate_patches_from_llm_analysis(
    analysis_report: Any,
    prompt_generation_result: Any,
    skill_configs: dict[str, dict],
) -> list[PatchSchema]:
    """Generate patches from LLM analysis results for ALL components.

    Stage 4: Split the analysis back into per-skill patches.
    Generates patches for: prompt, retrieval_routing, query_rewrite,
    evidence_expansion, answer_generation.
    """
    if not analysis_report.should_generate_patch:
        return []

    patches: list[PatchSchema] = []

    # Get the failure IDs from the analysis
    failure_ids = []
    for rc in analysis_report.common_root_causes:
        failure_ids.extend(rc.get("supporting_failure_ids", []))
    for sc in analysis_report.subclusters:
        failure_ids.extend(sc.get("failure_ids", []))
    failure_ids = list(set(failure_ids))

    primary_failure_type = analysis_report.primary_failure_type
    try:
        ft = FailureType(primary_failure_type)
    except ValueError:
        ft = FailureType.ASSESSMENT_FALSE_NEG

    recommended = analysis_report.recommended_changes
    affected_skills = list(skill_configs.keys())

    # --- 1. Prompt patches ---
    prompt_changes = recommended.get("prompt", {})
    if prompt_changes.get("should_change") and prompt_generation_result and prompt_generation_result.success:
        new_prompt_ref = prompt_generation_result.prompt_ref
        base_prompt_ref = prompt_generation_result.base_prompt_ref

        prompt_role = prompt_changes.get("prompt_role", "evidence_assessment")
        if prompt_role == "answer_generation":
            strategy_path = "strategy.answer_generation"
        else:
            strategy_path = "strategy.assessment"

        # 1a. prompt_content_update patch
        content_patch_id = f"{new_prompt_ref.lower()}_prompt_content"
        content_patch = PatchSchema(
            patch_id=content_patch_id,
            patch_type=PATCH_TYPE_PROMPT_CONTENT_UPDATE,
            skill_name=affected_skills[0] if affected_skills else "unknown",
            target_file=f"config/prompts/{prompt_generation_result.prompt_filename}",
            source_failure_ids=failure_ids,
            targeted_failure_ids=prompt_generation_result.targeted_failure_ids or failure_ids,
            primary_failure_type=ft,
            target_path=f"config/prompts/{prompt_generation_result.prompt_filename}",
            operation=PatchOperation.ADD,
            old_value=None,
            new_value=prompt_generation_result.content,
            prompt_artifacts=[PromptArtifact(
                prompt_ref=new_prompt_ref,
                base_prompt_ref=base_prompt_ref,
                prompt_role=prompt_role,
                version=prompt_generation_result.version,
                prompt_filename=prompt_generation_result.prompt_filename,
                created_from=base_prompt_ref,
                diff_summary="; ".join(prompt_generation_result.change_summary[:3]),
                registry_path=f"config/prompts/{prompt_generation_result.prompt_filename}",
                content=prompt_generation_result.content,
                change_summary=prompt_generation_result.change_summary,
                preserved_constraints=prompt_generation_result.preserved_constraints,
                targeted_failure_ids=prompt_generation_result.targeted_failure_ids or failure_ids,
                validation_requirements=prompt_generation_result.validation_requirements,
            )],
            rationale=f"LLM-generated V2 prompt: {'; '.join(prompt_generation_result.change_summary[:3])}",
            expected_improvement=f"Address {primary_failure_type} failures via improved prompt",
            risk_level=prompt_changes.get("risk_level", "medium"),
            confidence=prompt_changes.get("confidence", 0.7),
        )
        patches.append(content_patch)

        # 1b. prompt_ref_update patches (one per affected skill)
        for skill_name, config in skill_configs.items():
            section = "assessment" if prompt_role != "answer_generation" else "answer_generation"
            current_ref = config.get("strategy", {}).get(section, {}).get("system_prompt_ref", "")
            if current_ref == base_prompt_ref:
                ref_patch = PatchSchema(
                    patch_id=f"{skill_name}_{new_prompt_ref.lower()}_ref",
                    patch_type=PATCH_TYPE_PROMPT_REF_UPDATE,
                    skill_name=skill_name,
                    skill_version=config.get("skill_version", "1.0.0"),
                    target_file=f"config/skills/{skill_name}.yaml",
                    source_failure_ids=failure_ids,
                    targeted_failure_ids=prompt_generation_result.targeted_failure_ids or failure_ids,
                    primary_failure_type=ft,
                    target_path=strategy_path,
                    operation=PatchOperation.UPDATE,
                    old_value=current_ref,
                    new_value=new_prompt_ref,
                    current_value={"system_prompt_ref": current_ref},
                    proposed_value={"system_prompt_ref": new_prompt_ref},
                    rationale=f"Point to V2 prompt: {'; '.join(prompt_generation_result.change_summary[:2])}",
                    expected_improvement=f"Use improved V2 prompt for {skill_name}",
                    risk_level=prompt_changes.get("risk_level", "medium"),
                    confidence=prompt_changes.get("confidence", 0.7),
                    dependencies=[content_patch_id],
                )
                patches.append(ref_patch)

    # --- 2. Retrieval routing patches ---
    rr_changes = recommended.get("retrieval_routing", {})
    if rr_changes.get("should_change"):
        for change in rr_changes.get("changes", []):
            for skill_name, config in skill_configs.items():
                patch = _make_param_patch(
                    skill_name, config, change, ft,
                    PATCH_TYPE_RETRIEVAL_STRATEGY_UPDATE, failure_ids,
                )
                if patch:
                    patches.append(patch)

    # --- 3. Query rewrite patches ---
    qr_changes = recommended.get("query_rewrite", {})
    if qr_changes.get("should_change"):
        for change in qr_changes.get("changes", []):
            for skill_name, config in skill_configs.items():
                patch = _make_param_patch(
                    skill_name, config, change, ft,
                    PATCH_TYPE_QUERY_REWRITE_UPDATE, failure_ids,
                )
                if patch:
                    patches.append(patch)

    # --- 4. Evidence expansion patches ---
    ee_changes = recommended.get("evidence_expansion", {})
    if ee_changes.get("should_change"):
        for change in ee_changes.get("changes", []):
            for skill_name, config in skill_configs.items():
                patch = _make_param_patch(
                    skill_name, config, change, ft,
                    PATCH_TYPE_EVIDENCE_EXPANSION_UPDATE, failure_ids,
                )
                if patch:
                    patches.append(patch)

    # --- 5. Answer generation patches (non-prompt) ---
    ag_changes = recommended.get("answer_generation", {})
    if ag_changes.get("should_change"):
        for change in ag_changes.get("changes", []):
            for skill_name, config in skill_configs.items():
                patch = _make_param_patch(
                    skill_name, config, change, ft,
                    PATCH_TYPE_ANSWER_GENERATION_UPDATE, failure_ids,
                )
                if patch:
                    patches.append(patch)

    return patches


def _make_param_patch(
    skill_name: str,
    skill_config: dict,
    change: dict[str, Any],
    failure_type: FailureType,
    patch_type: str,
    failure_ids: list[str],
) -> PatchSchema | None:
    """Create a parameter update patch from an LLM-recommended change."""
    target_path = change.get("target_path", "")
    if not target_path:
        return None

    # Verify the target path exists in the skill config
    parts = target_path.split(".")
    current = skill_config
    for p in parts:
        if isinstance(current, dict) and p in current:
            current = current[p]
        else:
            return None

    return PatchSchema(
        patch_id=f"{skill_name}_{patch_type}_{target_path.replace('.', '_')}",
        patch_type=patch_type,
        skill_name=skill_name,
        skill_version=skill_config.get("skill_version", "1.0.0"),
        target_file=f"config/skills/{skill_name}.yaml",
        source_failure_ids=failure_ids,
        targeted_failure_ids=change.get("source_failure_ids", failure_ids),
        primary_failure_type=failure_type,
        target_path=target_path,
        operation=PatchOperation.UPDATE,
        old_value=change.get("current_value"),
        new_value=change.get("proposed_value"),
        current_value={parts[-1]: change.get("current_value")},
        proposed_value={parts[-1]: change.get("proposed_value")},
        rationale=change.get("rationale", ""),
        expected_improvement=change.get("expected_improvement", ""),
        risk_level=change.get("risk_level", "low"),
        confidence=change.get("confidence", 0.7),
    )


# ---------------------------------------------------------------------------
# §5.4 Template Distillation
# ---------------------------------------------------------------------------

def distill_templates(
    success_patterns: list[SuccessPattern],
    partial_successes: list[PartialSuccessRecord],
) -> list[PatchSchema]:
    """Identify template distillation candidates from success patterns.

    Triggers: same intent + similar pattern + frequency >= 3.
    Returns template_add patches.

    Requirements for a valid template:
    1. pattern must NOT be a bare score label (e.g. "score_1.0")
    2. must have at least 3 UNIQUE question_ids (deduplicated)
    3. must have meaningful structure (query_structure, evidence_type, etc.)
    4. must have reusable_template content (not empty)
    """
    # Filter out score-label patterns — these carry no structural information
    _SCORE_LABEL_RE = re.compile(r"^score_\d+\.\d+$")
    # Also reject patterns that are just score labels with no structural info
    _SCORE_ONLY_RE = re.compile(r"^score_\d")

    # Group success patterns by intent + pattern, deduplicating question_ids
    pattern_groups: dict[tuple[str, str], set[str]] = {}
    pattern_scores: dict[tuple[str, str], list[float]] = {}
    pattern_details: dict[tuple[str, str], SuccessPattern] = {}
    for sp in success_patterns:
        # Skip bare score labels
        if _SCORE_LABEL_RE.match(sp.pattern) or _SCORE_ONLY_RE.match(sp.pattern):
            continue
        # Skip patterns with no reusable template
        if not sp.reusable_template and not sp.trigger_conditions:
            continue
        key = (sp.intent, sp.pattern)
        pattern_groups.setdefault(key, set()).update(sp.supporting_question_ids)
        pattern_scores.setdefault(key, []).append(sp.avg_score)
        # Keep the first occurrence for details
        if key not in pattern_details:
            pattern_details[key] = sp

    patches: list[PatchSchema] = []
    for idx, ((intent, pattern), qids) in enumerate(pattern_groups.items()):
        # Require at least 3 UNIQUE question_ids
        if len(qids) < 3:
            continue
        group_key = (intent, pattern)
        scores = pattern_scores.get(group_key, [0.0])
        avg_score = sum(scores) / len(scores) if scores else 0.0
        if avg_score < 0.85:
            continue

        # Get structural details from the stored pattern
        details = pattern_details.get(group_key)
        trigger_conditions = {}
        reusable_template = ""
        query_structure = ""
        evidence_type = ""
        answer_structure = ""
        if details:
            trigger_conditions = details.trigger_conditions or {"intent": intent}
            reusable_template = details.reusable_template or ""
            query_structure = details.query_structure or ""
            evidence_type = details.evidence_type or ""
            answer_structure = details.answer_structure or ""

        # Build a meaningful template value
        template_value: dict[str, Any] = {
            "name": f"distilled_{pattern}_{intent}",
            "description": (
                f"Auto-distilled from {len(qids)} successful queries "
                f"with structure '{pattern}' in {intent}."
            ),
            "applicable_when": trigger_conditions or {"intent": intent},
            "not_applicable_when": [],
            "slots": [],
            "examples": [],
        }
        if query_structure:
            template_value["applicable_when"]["query_structure"] = query_structure
        if evidence_type:
            template_value["applicable_when"]["evidence_type"] = evidence_type

        patches.append(PatchSchema(
            patch_id=f"{intent}_template_distill_{idx + 1}",
            skill_name=intent,
            primary_failure_type=FailureType.UNKNOWN_FAILURE,  # not a failure, but uses same schema
            target_path="templates",
            operation=PatchOperation.ADD,
            proposed_value=template_value,
            rationale=(
                f"Structural pattern '{pattern}' succeeded {len(qids)} times in {intent} "
                f"(avg {avg_score:.2f}). Query structure: {query_structure}, "
                f"evidence type: {evidence_type}, answer structure: {answer_structure}."
            ),
            expected_improvement=(
                f"Reusable template for {query_structure} queries using {evidence_type}."
            ) if query_structure else "Reusable template for similar future queries.",
            risk_level="low",
            confidence=min(avg_score, 0.95),
        ))
    return patches


def generate_all_patches(
    report,
    skill_configs: dict[str, dict],
) -> list[PatchSchema]:
    """Generate all candidate patches from a TraceReport.

    Main entry point for §5.
    """
    # Cluster failures
    clusters = cluster_failures(report.failures)
    patches: list[PatchSchema] = []
    for cluster in clusters:
        config = skill_configs.get(cluster.skill_name, {})
        patches.extend(generate_patches_for_cluster(cluster, config))

    # Template distillation
    patches.extend(distill_templates(report.success_patterns, report.partial_successes))

    # Deduplicate: if multiple patches have identical target_path + proposed_value,
    # merge their source_failure_ids and keep only one
    patches = _deduplicate_patches(patches)

    return patches


def _deduplicate_patches(patches: list[PatchSchema]) -> list[PatchSchema]:
    """Deduplicate patches with identical target_path + proposed_value.

    When multiple failure clusters generate the same patch (e.g., all
    assessment_false_negative → swap to V2), merge their source_failure_ids
    into a single patch.
    """
    seen: dict[str, PatchSchema] = {}
    for p in patches:
        # Build a dedup key from skill_name + target_path + proposed_value
        key_parts = [
            p.skill_name,
            p.target_path,
            json.dumps(p.proposed_value, sort_keys=True, default=str),
        ]
        dedup_key = "|".join(key_parts)

        if dedup_key in seen:
            # Merge source_failure_ids
            existing = seen[dedup_key]
            existing.source_failure_ids = list(set(
                existing.source_failure_ids + p.source_failure_ids
            ))
            # Keep higher confidence
            if p.confidence > existing.confidence:
                existing.confidence = p.confidence
        else:
            seen[dedup_key] = p

    return list(seen.values())
