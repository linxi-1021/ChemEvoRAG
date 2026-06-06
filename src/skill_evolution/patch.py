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

from .attribution import FailureRecord, FailureType, SuccessPattern, PartialSuccessRecord


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
    created_from: str = ""
    diff_summary: str = ""
    registry_path: str = ""


class PatchSchema(_BaseModel):
    """A candidate patch to be applied to a Skill YAML or Prompt Registry."""
    patch_id: str = ""
    skill_name: str
    skill_version: str = "1.0.0"
    source_failure_ids: list[str] = Field(default_factory=list)
    primary_failure_type: FailureType
    target_path: str
    operation: PatchOperation = PatchOperation.UPDATE
    current_value: dict[str, Any] = Field(default_factory=dict)
    proposed_value: dict[str, Any] = Field(default_factory=dict)
    prompt_artifacts: list[PromptArtifact] = Field(default_factory=list)
    rationale: str = ""
    expected_improvement: str = ""
    risk_level: str = "medium"
    confidence: float = 0.0
    status: PatchStatus = PatchStatus.CANDIDATE


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
    FailureType.EVIDENCE_EXPANSION_ERROR: {
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

    # Patch 1: Update prompt ref (will trigger prompt generation later)
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
            diff_summary="Strengthen assessment to detect when numeric values are present in evidence but not recognized.",
            registry_path=f"config/prompts/{new_prompt_ref.lower()}.yaml",
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
# §5.4 Template Distillation
# ---------------------------------------------------------------------------

def distill_templates(
    success_patterns: list[SuccessPattern],
    partial_successes: list[PartialSuccessRecord],
) -> list[PatchSchema]:
    """Identify template distillation candidates from success patterns.

    Triggers: same intent + similar pattern + frequency >= 3.
    Returns template_add patches.
    """
    # Group success patterns by intent + pattern
    pattern_groups: dict[tuple[str, str], list[str]] = {}
    for sp in success_patterns:
        key = (sp.intent, sp.pattern)
        pattern_groups.setdefault(key, []).extend(sp.supporting_question_ids)

    patches: list[PatchSchema] = []
    for (intent, pattern), qids in pattern_groups.items():
        if len(qids) < 3:
            continue
        avg_score = sum(sp.avg_score for sp in success_patterns
                        if sp.intent == intent and sp.pattern == pattern) / max(1, len([
            sp for sp in success_patterns if sp.intent == intent and sp.pattern == pattern
        ]))
        if avg_score < 0.85:
            continue
        patches.append(PatchSchema(
            skill_name=intent,
            primary_failure_type=FailureType.UNKNOWN_FAILURE,  # not a failure, but uses same schema
            target_path="templates",
            operation=PatchOperation.ADD,
            proposed_value={
                "name": f"distilled_{pattern}_{intent}",
                "description": f"Auto-distilled from {len(qids)} successful queries with pattern '{pattern}'.",
                "applicable_when": {"intent": intent, "pattern": pattern},
                "frequency": len(qids),
            },
            rationale=f"Pattern '{pattern}' succeeded {len(qids)} times in {intent} (avg {avg_score:.2f}).",
            expected_improvement="Reusable template for similar future queries.",
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

    return patches
