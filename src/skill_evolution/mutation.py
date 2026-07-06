"""LLM-directed mutation for failure-driven evolution. Stage 4A.

6 mutation dimensions aligned with architecture diagram B layer:
  prompt          — evidence assessment / answer generation prompt
  routing         — retrieval_routing (channels, top_k, boost_weights)
  planning        — query_rewrite (fallback_terms, diversity_threshold, max_variants)
  fallback        — failure_labels, retry strategy
  graph_expansion — evidence_expansion (block_neighbors, graph_neighbors, cross_section)
  stop_condition  — assessment thresholds, max_rounds

Each dimension:
  1. Stage 2 LLM analysis: decides if dimension needs mutation
  2. Stage 3 LLM generation (prompt dimensions only): generates improved V2 prompt
  3. rule-based (parameter dimensions): generates parameter patches from channel stats

All patches go through mutable_paths / frozen_paths whitelist validation.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from pydantic import BaseModel, ConfigDict, Field
except ImportError:
    raise ImportError("pydantic is required for skill_evolution.mutation")

from .attribution import FailureType, FailureRecord
from .patch import (
    PatchSchema, PatchOperation, PromptArtifact,
    PATCH_TYPE_PROMPT_CONTENT_UPDATE, PATCH_TYPE_PROMPT_REF_UPDATE,
    PATCH_TYPE_RETRIEVAL_STRATEGY_UPDATE, PATCH_TYPE_QUERY_REWRITE_UPDATE,
    PATCH_TYPE_EVIDENCE_EXPANSION_UPDATE, PATCH_TYPE_ANSWER_GENERATION_UPDATE,
    PATCH_TYPE_FALLBACK_UPDATE, PATCH_TYPE_STOP_CONDITION_UPDATE,
)
from .config import get_config


# ══════════════════════════════════════════════════════════════════════════
# Dimension definitions
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class MutationDimension:
    name: str
    description: str
    target_yaml_path: str
    mutable_fields: list[str]
    uses_stage3_llm: bool = False
    prompt_role: str = ""  # "evidence_assessment" or "answer_generation"

MUTATION_DIMENSIONS: dict[str, MutationDimension] = {
    "prompt": MutationDimension(
        name="prompt",
        description="Evidence assessment / answer generation prompt improvement",
        target_yaml_path="strategy.assessment",
        mutable_fields=["system_prompt_ref", "require_numeric_extraction", "failure_labels"],
        uses_stage3_llm=True,
        prompt_role="evidence_assessment",
    ),
    "routing": MutationDimension(
        name="routing",
        description="Retrieval channel selection, top_k, boost_weights",
        target_yaml_path="strategy.retrieval_routing",
        mutable_fields=["primary_channels", "top_k.*", "boost_weights.*"],
    ),
    "planning": MutationDimension(
        name="planning",
        description="Query rewrite rules: fallback_terms, diversity_threshold, max_variants",
        target_yaml_path="strategy.query_rewrite",
        mutable_fields=["fallback_terms", "diversity_threshold", "max_variants"],
    ),
    "fallback": MutationDimension(
        name="fallback",
        description="Failure retry / fallback behavior",
        target_yaml_path="strategy.assessment",
        mutable_fields=["failure_labels"],
    ),
    "graph_expansion": MutationDimension(
        name="graph_expansion",
        description="Evidence expansion: block neighbors, reaction graph, cross-section linking",
        target_yaml_path="strategy.evidence_expansion",
        mutable_fields=[
            "max_expanded_blocks", "expand_from_block_neighbors",
            "expand_from_molecule_mentions", "expand_from_reaction_graph",
            "cross_section_linking", "max_graph_neighbors",
        ],
    ),
    "stop_condition": MutationDimension(
        name="stop_condition",
        description="ReAct loop stop criteria: thresholds, max_rounds",
        target_yaml_path="strategy.assessment",
        mutable_fields=["assessment thresholds", "max_rounds"],
    ),
    "answer_generation": MutationDimension(
        name="answer_generation",
        description="Answer generation prompt / max_evidence_items",
        target_yaml_path="strategy.answer_generation",
        mutable_fields=["system_prompt_ref", "max_evidence_items"],
        uses_stage3_llm=True,
        prompt_role="answer_generation",
    ),
}


# ══════════════════════════════════════════════════════════════════════════
# mutable_paths / frozen_paths whitelist validation
# ══════════════════════════════════════════════════════════════════════════

def validate_dimension_mutable(dimension_name: str, skill_config: dict) -> bool:
    """Check if a mutation dimension's target path is in skill's mutable_paths
    and NOT in frozen_paths."""
    dim = MUTATION_DIMENSIONS.get(dimension_name)
    if not dim:
        return False

    evolution = skill_config.get("evolution", {})
    mutable = evolution.get("mutable_paths", [])
    frozen = evolution.get("frozen_paths", [])

    target = dim.target_yaml_path

    # Check frozen — if target or any parent is frozen, reject
    for fp in frozen:
        if target == fp or target.startswith(fp + ".") or fp.startswith(target + "."):
            return False

    # Check mutable — target or a parent must be in mutable
    for mp in mutable:
        if target == mp or target.startswith(mp + ".") or mp.startswith(target + "."):
            return True

    return False


def get_mutable_fields_for_dimension(dimension_name: str, skill_config: dict) -> list[str]:
    """Get the allowed mutable fields for a dimension, filtered by skill's constraints."""
    dim = MUTATION_DIMENSIONS.get(dimension_name)
    if not dim:
        return []

    # Cross-reference with failure_to_patch_mapping constraints
    evolution = skill_config.get("evolution", {})
    patch_map = evolution.get("failure_to_patch_mapping", {})
    allowed: set[str] = set()

    for ft_entry in patch_map.values():
        for target in ft_entry.get("target_paths", []):
            if target == dim.target_yaml_path:
                allowed.update(ft_entry.get("allowed_fields", []))

    if allowed:
        return [f for f in dim.mutable_fields if f in allowed]
    return list(dim.mutable_fields)


# ══════════════════════════════════════════════════════════════════════════
# Stage 2: LLM analysis (calls into llm_analysis module)
# ══════════════════════════════════════════════════════════════════════════

def analyze_cluster_for_mutation(
    cluster: dict[str, Any],
    failure_records: list[dict[str, Any]],
    skill_configs: dict[str, dict],
    current_prompts: dict[str, dict[str, str]],
    previous_feedback: dict | None = None,
) -> dict[str, Any]:
    """Stage 2: LLM analyzes a failure cluster and returns recommended changes.

    Returns FailureAnalysisReport-like dict with per-dimension recommendations.
    Delegates to llm_analysis.analyze_failure_cluster().
    """
    from .llm_analysis import analyze_failure_cluster as llm_analyze

    # If this is a refine round, add previous feedback to the cluster context
    enriched_cluster = dict(cluster)
    if previous_feedback:
        enriched_cluster["previous_version_feedback"] = previous_feedback

    report = llm_analyze(
        cluster=enriched_cluster,
        failure_records=failure_records,
        current_prompts=current_prompts,
        current_skill_configs=skill_configs,
        max_retries=get_config().max_retries,
    )

    if not report.should_generate_patch:
        return {"should_generate_patch": False, "reasons": [report.raw_llm_response]}

    return {
        "should_generate_patch": True,
        "primary_failure_type": report.primary_failure_type,
        "contributing_failure_types": report.contributing_failure_types,
        "common_root_causes": report.common_root_causes,
        "recommended_changes": report.recommended_changes,
        "risk_level": report.risk_level,
    }


# ══════════════════════════════════════════════════════════════════════════
# Stage 3: Prompt generation (for prompt dimensions)
# ══════════════════════════════════════════════════════════════════════════

def generate_prompt_mutation(
    analysis: dict[str, Any],
    failure_records: list[dict[str, Any]],
    base_prompt_ref: str,
    v1_content: str,
    prompt_role: str,
    version: int = 2,
) -> dict[str, Any] | None:
    """Stage 3: Generate next version prompt based on V1 + failure analysis.

    Returns PromptArtifact-like dict on success, None on failure.
    """
    from .llm_analysis import generate_v2_prompt as llm_gen

    new_ref = f"{base_prompt_ref}_V{version}"
    targeted_ids = list(set(
        fid for rc in analysis.get("common_root_causes", [])
        for fid in rc.get("supporting_failure_ids", [])
    ))

    result = llm_gen(
        analysis_report=analysis,  # pass the analysis dict directly
        v1_content=v1_content,
        base_prompt_ref=base_prompt_ref,
        new_prompt_ref=new_ref,
        targeted_failure_ids=targeted_ids,
        failure_records=failure_records,
        max_retries=get_config().max_retries,
    )

    if not result.success:
        return None

    return {
        "prompt_ref": result.prompt_ref,
        "base_prompt_ref": result.base_prompt_ref,
        "prompt_role": result.prompt_role,
        "version": result.version,
        "prompt_filename": result.prompt_filename,
        "content": result.content,
        "change_summary": result.change_summary,
        "preserved_constraints": result.preserved_constraints,
        "targeted_failure_ids": result.targeted_failure_ids,
        "validation_requirements": result.validation_requirements,
    }


def refine_after_regression(
    v1_content: str,
    v2_content: str,
    regression_feedback: dict,
    base_prompt_ref: str,
    new_prompt_ref: str,
    version: int = 3,
) -> dict[str, Any] | None:
    """Refine V2 prompt based on regression feedback (iterative loop).

    Called when regression shows V2 improves targets but causes regressions.
    Feeds pass+regressed samples back to LLM for targeted correction.

    Returns refined prompt dict with 'content', 'change_summary', etc., or None.
    """
    from .llm_analysis import refine_prompt_with_feedback as llm_refine

    # Ensure feedback has required structure
    feedback = dict(regression_feedback)
    feedback.setdefault("pass_count", 0)
    feedback.setdefault("fail_count", 0)
    feedback.setdefault("improved", [])
    feedback.setdefault("regressed", [])
    feedback.setdefault("unchanged_failures", [])

    new_ref = f"{base_prompt_ref}_V{version}"
    print(f"  [Refine] Feedback: {feedback.get('pass_count',0)} pass, "
          f"{len(feedback.get('improved',[]))} improved, "
          f"{len(feedback.get('regressed',[]))} regressed")
    # Only 1 LLM iteration here — runner.py handles the regression-gated loop
    refined = llm_refine(
        v1_content=v1_content,
        v2_content=v2_content,
        regression_feedback=feedback,
        base_prompt_ref=base_prompt_ref,
        new_prompt_ref=new_ref,
        max_refine_iterations=1,  # Single refine, regression happens between calls
    )
    if not refined:
        print(f"  [Refine] LLM unable to refine without regressions")
        return None

    return {
        "prompt_ref": new_ref,
        "base_prompt_ref": base_prompt_ref,
        "version": "3.0",
        "content": refined.get("content", ""),
        "change_summary": refined.get("change_summary", ["Refined after regression feedback"]),
        "preserved_constraints": refined.get("preserved_constraints", []),
        "refined_rules": refined.get("refined_rules", []),
        "regression_analysis": refined.get("regression_analysis", ""),
    }


def _extract_regression_feedback(
    v1_score_map: dict[str, float],
    v2_score_map: dict[str, float],
    v1_answer_map: dict[str, str],
    v2_answer_map: dict[str, str],
    question_map: dict[str, str],
) -> dict:
    """Build detailed regression feedback from score comparisons.

    Compares per-question V1 vs V2 scores and answers to identify
    improved, regressed, and unchanged samples.
    """
    improved = []
    regressed = []
    unchanged_failures = []
    pass_count = 0
    fail_count = 0

    for qid in set(list(v1_score_map.keys()) + list(v2_score_map.keys())):
        v1 = v1_score_map.get(qid, 0.0)
        v2 = v2_score_map.get(qid, 0.0)
        delta = v2 - v1

        if delta >= 0.2:
            improved.append({
                "id": qid,
                "before": v1,
                "after": v2,
                "question": question_map.get(qid, "")[:300],
            })
            pass_count += 1
        elif delta <= -0.2:
            regressed.append({
                "id": qid,
                "before": v1,
                "after": v2,
                "v1_answer": v1_answer_map.get(qid, "")[:300],
                "v2_answer": v2_answer_map.get(qid, "")[:300],
                "question": question_map.get(qid, "")[:300],
            })
            fail_count += 1
        elif v2 < 0.5 and delta < 0.2:
            unchanged_failures.append({
                "id": qid,
                "before": v1,
                "after": v2,
            })
            fail_count += 1
        elif delta >= 0 and delta < 0.2 and v2 >= 0.5:
            pass_count += 1

    return {
        "pass_count": pass_count,
        "fail_count": fail_count,
        "improved": improved,
        "regressed": regressed,
        "unchanged_failures": unchanged_failures,
    }


# ══════════════════════════════════════════════════════════════════════════
# Rule-based parameter patch generation
# ══════════════════════════════════════════════════════════════════════════

def _generate_routing_patches(
    skill_name: str, skill_config: dict, analysis: dict, failure_ids: list[str], ft: FailureType,
    failure_traces: list[dict] | None = None,
    success_analysis_data: dict | None = None,
) -> list[PatchSchema]:
    """Generate retrieval_routing patches from failure type heuristics + success data."""
    patches: list[PatchSchema] = []
    rr = skill_config.get("strategy", {}).get("retrieval_routing", {})

    # Success-data-driven if available
    if success_analysis_data:
        intent_data = success_analysis_data.get("global", {}).get("by_intent", {}).get(skill_name, {})
        if intent_data.get("best_channels"):
            current_channels = rr.get("primary_channels", [])
            best = intent_data["best_channels"]
            if set(best) != set(current_channels):
                patches.append(PatchSchema(
                    skill_name=skill_name, skill_version=skill_config.get("skill_version", "1.0.0"),
                    source_failure_ids=failure_ids, primary_failure_type=ft,
                    target_path="strategy.retrieval_routing", operation=PatchOperation.UPDATE,
                    current_value={"primary_channels": current_channels},
                    proposed_value={"primary_channels": best},
                    rationale=f"Success data: {intent_data.get('trace_count', 0)} traces used {best}.",
                    expected_improvement="Align channels with proven successful patterns.",
                    risk_level="low", confidence=0.75,
                    patch_type=PATCH_TYPE_RETRIEVAL_STRATEGY_UPDATE,
                ))

    # Failure-driven fallback
    if ft in (FailureType.ROUTING_ERROR, FailureType.ASSESSMENT_FALSE_NEG, FailureType.TABLE_REASONING_ERROR):
        current_weights = rr.get("boost_weights", {})
        tb = current_weights.get("table_block", {})
        table_weight = tb.get("weight", 4.0) if isinstance(tb, dict) else (tb if isinstance(tb, (int, float)) else 4.0)
        if float(table_weight) < 8.0:
            patches.append(PatchSchema(
                skill_name=skill_name, skill_version=skill_config.get("skill_version", "1.0.0"),
                source_failure_ids=failure_ids, primary_failure_type=ft,
                target_path="strategy.retrieval_routing", operation=PatchOperation.UPDATE,
                current_value={"boost_weights": current_weights},
                proposed_value={"boost_weights": {"table_block": {"weight": float(table_weight) + 2.0}}},
                rationale="Table evidence often contains the answer but is underweighted.",
                expected_improvement="Improve evidence recall for table-supported questions.",
                risk_level="low", confidence=0.65,
                patch_type=PATCH_TYPE_RETRIEVAL_STRATEGY_UPDATE,
            ))

    return patches


def _generate_planning_patches(
    skill_name: str, skill_config: dict, analysis: dict, failure_ids: list[str], ft: FailureType,
    **kwargs,
) -> list[PatchSchema]:
    """Generate query_rewrite parameter patches."""
    patches: list[PatchSchema] = []
    qr = skill_config.get("strategy", {}).get("query_rewrite", {})

    if ft in (FailureType.ENTITY_MISS, FailureType.ALIAS_MISS, FailureType.LOCAL_ID_MISS):
        current_fallback = qr.get("fallback_terms", [])
        new_terms = list(current_fallback)
        if ft == FailureType.ALIAS_MISS:
            new_terms.extend(["alias", "abbreviation", "alternative name"])
        else:
            new_terms.extend(["compound", "chemical", "molecule"])
        new_terms = list(dict.fromkeys(new_terms))

        patches.append(PatchSchema(
            skill_name=skill_name,
            skill_version=skill_config.get("skill_version", "1.0.0"),
            source_failure_ids=failure_ids,
            primary_failure_type=ft,
            target_path="strategy.query_rewrite",
            operation=PatchOperation.UPDATE,
            current_value={"fallback_terms": current_fallback},
            proposed_value={"fallback_terms": new_terms},
            rationale=f"Expand fallback terms to improve entity/alias recall for {ft.value}.",
            expected_improvement="Better entity retrieval through expanded query variants.",
            risk_level="low",
            confidence=0.65,
            patch_type=PATCH_TYPE_QUERY_REWRITE_UPDATE,
        ))

    return patches


def _generate_graph_expansion_patches(
    skill_name: str, skill_config: dict, analysis: dict, failure_ids: list[str], ft: FailureType,
    **kwargs,
) -> list[PatchSchema]:
    """Generate evidence_expansion parameter patches."""
    patches: list[PatchSchema] = []
    ee = skill_config.get("strategy", {}).get("evidence_expansion", {})

    if ft in (FailureType.GRAPH_EXPANSION_ERROR, FailureType.ENTITY_MISS):
        current_max = ee.get("max_expanded_blocks", 5)
        if current_max < 10:
            patches.append(PatchSchema(
                skill_name=skill_name,
                skill_version=skill_config.get("skill_version", "1.0.0"),
                source_failure_ids=failure_ids,
                primary_failure_type=ft,
                target_path="strategy.evidence_expansion",
                operation=PatchOperation.UPDATE,
                current_value={"max_expanded_blocks": current_max},
                proposed_value={"max_expanded_blocks": min(current_max + 3, 12)},
                rationale="Nearby evidence blocks may contain the missing answer.",
                expected_improvement="Improve evidence completeness through broader expansion.",
                risk_level="low",
                confidence=0.60,
                patch_type=PATCH_TYPE_EVIDENCE_EXPANSION_UPDATE,
            ))

    return patches


def _generate_fallback_patches(
    skill_name: str, skill_config: dict, analysis: dict, failure_ids: list[str], ft: FailureType,
    **kwargs,
) -> list[PatchSchema]:
    """Generate fallback strategy patches."""
    patches: list[PatchSchema] = []
    assessment = skill_config.get("strategy", {}).get("assessment", {})

    if ft == FailureType.ASSESSMENT_FALSE_NEG:
        current_labels = assessment.get("failure_labels", [])
        new_labels = list(current_labels)
        if "premature_stop" not in new_labels:
            new_labels.append("premature_stop")
        patches.append(PatchSchema(
            skill_name=skill_name,
            skill_version=skill_config.get("skill_version", "1.0.0"),
            source_failure_ids=failure_ids,
            primary_failure_type=ft,
            target_path="strategy.assessment",
            operation=PatchOperation.UPDATE,
            current_value={"failure_labels": current_labels},
            proposed_value={"failure_labels": new_labels},
            rationale="Add premature_stop label to better detect false negative assessments.",
            expected_improvement="More precise failure categorization.",
            risk_level="low",
            confidence=0.55,
            patch_type=PATCH_TYPE_FALLBACK_UPDATE,
        ))

    return patches


def _generate_stop_condition_patches(
    skill_name: str, skill_config: dict, analysis: dict, failure_ids: list[str], ft: FailureType,
    failure_traces: list[dict] | None = None,
    **kwargs,
) -> list[PatchSchema]:
    """Analyze early-stop patterns and evidence exhaustion."""
    patches: list[PatchSchema] = []
    assessment = skill_config.get("strategy", {}).get("assessment", {})

    if not failure_traces:
        return patches

    # Count early stops: stopped before max rounds, assessment said insufficient
    early_stops = []
    exhausted = []
    for trace in failure_traces:
        rounds_used = trace.get("react_rounds_used", len(trace.get("retrieval_rounds", [])))
        assessment_result = trace.get("assessment_result", "")
        if rounds_used < 3 and "sufficient=false" in str(assessment_result).lower():
            early_stops.append(trace)
        if all(rr.get("new_evidence_count", 0) == 0 for rr in trace.get("retrieval_rounds", [])):
            exhausted.append(trace)

    if len(early_stops) >= 2:
        current_labels = assessment.get("failure_labels", [])
        new_labels = list(current_labels)
        if "early_stop_warning" not in new_labels:
            new_labels.append("early_stop_warning")
        patches.append(PatchSchema(
            skill_name=skill_name,
            skill_version=skill_config.get("skill_version", "1.0.0"),
            source_failure_ids=failure_ids,
            primary_failure_type=ft,
            target_path="strategy.assessment",
            operation=PatchOperation.UPDATE,
            current_value={"failure_labels": current_labels},
            proposed_value={"failure_labels": new_labels},
            rationale=f"{len(early_stops)}/{len(failure_traces)} failures stopped early (round {min(t.get('react_rounds_used', 0) for t in early_stops)}).",
            expected_improvement="Flag early-stop patterns for re-evaluation.",
            risk_level="low",
            confidence=0.60,
            patch_type=PATCH_TYPE_STOP_CONDITION_UPDATE,
        ))

    if len(exhausted) >= len(failure_traces) * 0.5:
        # Evidence exhaustion across > 50% of failures → likely coverage gap, not strategy issue
        pass

    return patches


def _generate_answer_generation_patches(
    skill_name: str, skill_config: dict, analysis: dict, failure_ids: list[str], ft: FailureType,
) -> list[PatchSchema]:
    """Generate answer_generation parameter patches."""
    patches: list[PatchSchema] = []
    ag = skill_config.get("strategy", {}).get("answer_generation", {})

    if ft == FailureType.GENERATION_ERROR:
        current_max = ag.get("max_evidence_items", 15)
        if current_max < 20:
            patches.append(PatchSchema(
                skill_name=skill_name,
                skill_version=skill_config.get("skill_version", "1.0.0"),
                source_failure_ids=failure_ids,
                primary_failure_type=ft,
                target_path="strategy.answer_generation",
                operation=PatchOperation.UPDATE,
                current_value={"max_evidence_items": current_max},
                proposed_value={"max_evidence_items": min(current_max + 3, 20)},
                rationale="Increase evidence limit for answer generation.",
                expected_improvement="More evidence items for LLM to generate better answers.",
                risk_level="low",
                confidence=0.50,
                patch_type=PATCH_TYPE_ANSWER_GENERATION_UPDATE,
            ))

    return patches


# ══════════════════════════════════════════════════════════════════════════
# Unified mutation entry point
# ══════════════════════════════════════════════════════════════════════════

_RULE_GENERATORS = {
    "routing": _generate_routing_patches,
    "planning": _generate_planning_patches,
    "graph_expansion": _generate_graph_expansion_patches,
    "fallback": _generate_fallback_patches,
    "stop_condition": _generate_stop_condition_patches,
    "answer_generation": _generate_answer_generation_patches,
}


def generate_all_mutations(
    cluster: dict[str, Any],
    failure_records: list[dict[str, Any]],
    skill_configs: dict[str, dict],
    current_prompts: dict[str, dict[str, str]],
    analysis: dict[str, Any] | None = None,
    failure_traces: list[dict] | None = None,
    success_analysis_data: dict | None = None,
    previous_feedback: dict | None = None,
) -> list[PatchSchema]:
    """Generate all candidate patches from a failure cluster (Stage 4A).

    1. Run Stage 2 LLM analysis if not provided
    2. For prompt dimensions: Stage 3 LLM generation
    3. For parameter dimensions: rule-based generation
    4. All patches validated against mutable_paths / frozen_paths
    """
    all_patches: list[PatchSchema] = []

    if not analysis:
        analysis = analyze_cluster_for_mutation(
            cluster, failure_records, skill_configs, current_prompts,
            previous_feedback=previous_feedback,
        )

    if not analysis.get("should_generate_patch"):
        # LLM analysis failed or declined — still try rule-based generators as fallback
        recommended = {}
    else:
        recommended = analysis.get("recommended_changes", {})
    failure_ids = list(set(
        fid for rc in analysis.get("common_root_causes", [])
        for fid in rc.get("supporting_failure_ids", [])
    ))
    ft_str = analysis.get("primary_failure_type", "unknown_failure")
    ft = FailureType(ft_str) if ft_str in [e.value for e in FailureType] else FailureType.UNKNOWN_FAILURE

    config = get_config()
    enabled_dims = config.mutation_enabled_dimensions

    # ── 1. Prompt mutations (Stage 3 LLM) ────────────────────────────────
    for dim_name in ("prompt", "answer_generation"):
        if dim_name not in enabled_dims:
            continue
        dim = MUTATION_DIMENSIONS.get(dim_name)
        if not dim or not dim.uses_stage3_llm:
            continue

        prompt_changes = recommended.get("prompt" if dim_name == "prompt" else "answer_generation", {})
        if not prompt_changes.get("should_change"):
            continue

        base_ref = prompt_changes.get("base_prompt_ref", "")
        if not base_ref:
            continue

        v1_content = current_prompts.get(dim.prompt_role, {}).get("content", "")
        if not v1_content:
            continue

        prompt_result = generate_prompt_mutation(
            analysis, failure_records, base_ref, v1_content, dim.prompt_role,
        )
        if not prompt_result:
            continue

        new_ref = prompt_result["prompt_ref"]
        content_patch_id = f"{new_ref.lower()}_prompt_content"

        # 1a. prompt_content_update
        content_patch = PatchSchema(
            patch_id=content_patch_id,
            patch_type=PATCH_TYPE_PROMPT_CONTENT_UPDATE,
            skill_name=list(skill_configs.keys())[0] if skill_configs else "unknown",
            target_file=f"config/prompts/{prompt_result['prompt_filename']}",
            source_failure_ids=failure_ids,
            targeted_failure_ids=prompt_result.get("targeted_failure_ids", failure_ids),
            primary_failure_type=ft,
            target_path=f"config/prompts/{prompt_result['prompt_filename']}",
            operation=PatchOperation.ADD,
            proposed_value={"content": prompt_result["content"]},
            prompt_artifacts=[PromptArtifact(
                prompt_ref=new_ref,
                created_from=base_ref,
                diff_summary="; ".join(prompt_result.get("change_summary", [])[:3]),
                registry_path=f"config/prompts/{prompt_result['prompt_filename']}",
                content=prompt_result["content"],
                change_summary=prompt_result.get("change_summary", []),
                preserved_constraints=prompt_result.get("preserved_constraints", []),
                targeted_failure_ids=prompt_result.get("targeted_failure_ids", failure_ids),
                validation_requirements=prompt_result.get("validation_requirements", []),
            )],
            rationale=f"LLM-generated V2 prompt: {'; '.join(prompt_result.get('change_summary', [])[:2])}",
            expected_improvement=f"Address {ft.value} failures via improved {dim.prompt_role} prompt.",
            risk_level=prompt_changes.get("risk_level", "medium"),
            confidence=prompt_changes.get("confidence", 0.7),
        )
        all_patches.append(content_patch)

        # 1b. prompt_ref_update (one per skill referencing the base prompt)
        section = "assessment" if dim.prompt_role == "evidence_assessment" else "answer_generation"
        for skill_name, skill_cfg in skill_configs.items():
            current_ref = skill_cfg.get("strategy", {}).get(section, {}).get("system_prompt_ref", "")
            if current_ref == base_ref:
                all_patches.append(PatchSchema(
                    patch_id=f"{skill_name}_{new_ref.lower()}_ref",
                    patch_type=PATCH_TYPE_PROMPT_REF_UPDATE,
                    skill_name=skill_name,
                    skill_version=skill_cfg.get("skill_version", "1.0.0"),
                    target_file=f"config/skills/{skill_name}.yaml",
                    source_failure_ids=failure_ids,
                    targeted_failure_ids=prompt_result.get("targeted_failure_ids", failure_ids),
                    primary_failure_type=ft,
                    target_path=f"strategy.{section}",
                    operation=PatchOperation.UPDATE,
                    current_value={"system_prompt_ref": current_ref},
                    proposed_value={"system_prompt_ref": new_ref},
                    rationale=f"Point to V2 prompt: {'; '.join(prompt_result.get('change_summary', [])[:2])}",
                    expected_improvement=f"Use improved V2 prompt for {skill_name}.",
                    risk_level=prompt_changes.get("risk_level", "medium"),
                    confidence=prompt_changes.get("confidence", 0.7),
                    dependencies=[content_patch_id],
                ))

    # ── 2. Parameter mutations (rule-based) ──────────────────────────────
    for dim_name, generator in _RULE_GENERATORS.items():
        if dim_name not in enabled_dims:
            continue
        dim = MUTATION_DIMENSIONS.get(dim_name)
        if not dim:
            continue

        for skill_name, skill_cfg in skill_configs.items():
            if not validate_dimension_mutable(dim_name, skill_cfg):
                continue
            patches = generator(skill_name, skill_cfg, analysis, failure_ids, ft, failure_traces=failure_traces, success_analysis_data=success_analysis_data)
            all_patches.extend(patches)

    # Ensure all patches have patch_ids (rule generators may not set them)
    for i, p in enumerate(all_patches):
        if not p.patch_id:
            p.patch_id = f"{p.skill_name}_{p.patch_type}_{i}"

    return all_patches
