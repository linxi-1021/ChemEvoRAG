"""Success-driven evolution. Stage 4B + C layer.

Refactored: uses success_analysis.py for LLM-driven analysis of successful traces
instead of generating empty template shells.

Three outputs:
  1. EvidenceUsageReport → feeds mutation.py routing/planning generators
  2. Cross-skill promotion → successful patterns shared across skills
  3. Template rewrite/merge → adjusts existing templates (when non-empty)
"""

from __future__ import annotations

import re
from typing import Any

from .attribution import SuccessPattern, PartialSuccessRecord
from .patch import (
    PatchSchema, PatchOperation,
    PATCH_TYPE_TEMPLATE_ADD, PATCH_TYPE_TEMPLATE_REWRITE, PATCH_TYPE_TEMPLATE_MERGE,
)
from .config import get_config


# ══════════════════════════════════════════════════════════════════════════
# Template ADD — new templates from success patterns
# ══════════════════════════════════════════════════════════════════════════

_SCORE_LABEL_RE = re.compile(r"^score_\d+\.\d+$")

def distill_new_templates(
    success_patterns: list[SuccessPattern],
    success_data: dict | None = None,
) -> list[PatchSchema]:
    """Generate evidence-usage-based recommendations from successful traces.

    Uses success_data (from success_analysis) to generate routing strategy patches
    based on proven successful patterns.
    """
    patches: list[PatchSchema] = []

    if not success_data:
        return patches

    global_data = success_data.get("global", {}).get("by_intent", {})

    patche_idx = 0
    for intent, stats in global_data.items():
        best_channels = stats.get("best_channels", [])
        best_evidence_types = stats.get("best_evidence_types", [])

        if not best_channels or not best_evidence_types:
            continue

        # Generate routing recommendation patch based on actual success data
        recommended_channels = stats.get("recommended_primary_channels", best_channels)
        recommended_boost = stats.get("recommended_boost_weights", {})

        if recommended_channels and len(recommended_channels) >= 2:
            patche_idx += 1
            patches.append(PatchSchema(
                patch_id=f"{intent}_success_routing_{patche_idx}",
                skill_name=intent,
                primary_failure_type="unknown_failure",
                target_path="strategy.retrieval_routing",
                operation=PatchOperation.UPDATE,
                patch_type=PATCH_TYPE_TEMPLATE_ADD,
                proposed_value={
                    "primary_channels": recommended_channels,
                    **({"boost_weights": recommended_boost} if recommended_boost else {}),
                },
                rationale=(
                    f"Success-driven: {stats.get('trace_count', 0)} successful {intent} traces "
                    f"used channels {best_channels} and evidence types {best_evidence_types}."
                ),
                expected_improvement=f"Align retrieval strategy with proven successful patterns in {intent}.",
                risk_level="low",
                confidence=0.80,
            ))

    return patches


# ══════════════════════════════════════════════════════════════════════════
# Template REWRITE — adjust existing templates
# ══════════════════════════════════════════════════════════════════════════

def rewrite_existing_templates(
    existing_templates: list[dict[str, Any]],
    new_patterns: list[SuccessPattern],
    skill_name: str,
) -> list[PatchSchema]:
    """Rewrite existing templates whose trigger_conditions are too narrow or too wide.

    Example: A template has applicable_when.intent="property_query" but new success
    patterns show it also works for "reaction_condition_query" → expand trigger.
    """
    patches: list[PatchSchema] = []

    for i, tmpl in enumerate(existing_templates):
        tmpl_name = tmpl.get("name", f"template_{i}")
        applicable = tmpl.get("applicable_when", {})
        tmpl_intent = applicable.get("intent", "")

        # Check if new patterns suggest expanding trigger
        matching_patterns = [
            sp for sp in new_patterns
            if sp.intent != tmpl_intent  # different intent
            and sp.query_structure == applicable.get("query_structure", "")
            and sp.evidence_type == applicable.get("evidence_type", "")
        ]

        if len(matching_patterns) >= 3:
            new_intents = list(set([tmpl_intent] + [sp.intent for sp in matching_patterns]))
            rewritten = dict(tmpl)
            rewritten["applicable_when"]["intent"] = new_intents
            rewritten["description"] = (
                f"{tmpl.get('description', '')} (expanded to {new_intents})"
            )

            patches.append(PatchSchema(
                patch_id=f"{skill_name}_template_rewrite_{i + 1}",
                skill_name=skill_name,
                primary_failure_type="unknown_failure",
                target_path="templates",
                operation=PatchOperation.UPDATE,
                patch_type=PATCH_TYPE_TEMPLATE_REWRITE,
                current_value=tmpl,
                proposed_value=rewritten,
                rationale=f"Expand template '{tmpl_name}' to cover {matching_patterns[0].intent} "
                          f"based on {len(matching_patterns)} cross-intent successes.",
                expected_improvement="Wider template applicability across intents.",
                risk_level="low",
                confidence=0.70,
            ))

    return patches


# ══════════════════════════════════════════════════════════════════════════
# Template MERGE — merge overlapping templates
# ══════════════════════════════════════════════════════════════════════════

def merge_overlapping_templates(
    existing_templates: list[dict[str, Any]],
    skill_name: str,
) -> list[PatchSchema]:
    """Merge templates whose applicable_when conditions overlap by > threshold."""
    config = get_config()
    patches: list[PatchSchema] = []

    if len(existing_templates) < 2:
        return patches

    for i in range(len(existing_templates)):
        for j in range(i + 1, len(existing_templates)):
            t1 = existing_templates[i]
            t2 = existing_templates[j]
            aw1 = t1.get("applicable_when", {})
            aw2 = t2.get("applicable_when", {})

            similarity = _compute_template_similarity(aw1, aw2)
            if similarity >= config.template_merge_similarity:
                merged_name = f"{t1.get('name', '')}_{t2.get('name', '')}_merged"
                merged_applicable = dict(aw1)
                # Union of conditions
                for k, v in aw2.items():
                    if k in merged_applicable:
                        existing_val = merged_applicable[k]
                        if isinstance(existing_val, list) and isinstance(v, list):
                            merged_applicable[k] = list(set(existing_val + v))
                    else:
                        merged_applicable[k] = v

                merged = {
                    "name": merged_name,
                    "description": f"Merged from {t1.get('name', '')} and {t2.get('name', '')}",
                    "applicable_when": merged_applicable,
                    "not_applicable_when": [],
                    "slots": list(set(t1.get("slots", []) + t2.get("slots", []))),
                    "examples": t1.get("examples", []) + t2.get("examples", []),
                }

                patches.append(PatchSchema(
                    patch_id=f"{skill_name}_template_merge_{i}_{j}",
                    skill_name=skill_name,
                    primary_failure_type="unknown_failure",
                    target_path="templates",
                    operation=PatchOperation.ADD,
                    patch_type=PATCH_TYPE_TEMPLATE_MERGE,
                    proposed_value=merged,
                    rationale=f"Merge overlapping templates (similarity={similarity:.2f}).",
                    expected_improvement="Reduce template redundancy.",
                    risk_level="low",
                    confidence=0.65,
                ))

    return patches


# ══════════════════════════════════════════════════════════════════════════
# Cross-skill promotion
# ══════════════════════════════════════════════════════════════════════════

def promote_to_cross_skill(
    success_patterns: list[SuccessPattern],
    skill_configs: dict[str, dict],
) -> list[PatchSchema]:
    """Promote successful patterns from one skill to compatible skills.

    Architecture diagram 4B: "把一次成功经验升级为通用规则"
    """
    config = get_config()
    patches: list[PatchSchema] = []

    # Group by intent for frequency counting
    intent_patterns: dict[str, dict[str, list[SuccessPattern]]] = {}
    for sp in success_patterns:
        intent_patterns.setdefault(sp.intent, {}).setdefault(sp.pattern, []).append(sp)

    patche_idx = 0
    for source_intent, patterns in intent_patterns.items():
        for pattern_name, sp_list in patterns.items():
            if len(sp_list) < config.cross_skill_promotion_min_frequency:
                continue

            candidate_skills = [
                name for name in skill_configs
                if name != source_intent
                and _compute_skill_similarity(source_intent, name, sp_list[0]) >= config.cross_skill_similarity_threshold
            ]

            for target_skill in candidate_skills:
                patche_idx += 1
                sp = sp_list[0]
                template_value: dict[str, Any] = {
                    "name": f"cross_skill_{pattern_name}_{source_intent}_to_{target_skill}",
                    "description": (
                        f"Cross-skill promotion: pattern '{pattern_name}' "
                        f"from {source_intent} → {target_skill} "
                        f"({len(sp_list)} successes, similarity={_compute_skill_similarity(source_intent, target_skill, sp):.2f})"
                    ),
                    "applicable_when": {
                        "intent": target_skill,
                        "query_structure": sp.query_structure,
                        "evidence_type": sp.evidence_type,
                    },
                    "not_applicable_when": [],
                    "slots": [],
                    "examples": [],
                    "promoted_from": source_intent,
                }

                patches.append(PatchSchema(
                    patch_id=f"{target_skill}_cross_skill_promotion_{patche_idx}",
                    skill_name=target_skill,
                    primary_failure_type="unknown_failure",
                    target_path="templates",
                    operation=PatchOperation.ADD,
                    patch_type=PATCH_TYPE_TEMPLATE_ADD,
                    proposed_value=template_value,
                    rationale=f"Cross-skill promotion from {source_intent}: pattern '{pattern_name}'.",
                    expected_improvement=f"Apply successful {source_intent} pattern to {target_skill}.",
                    risk_level="medium",
                    confidence=sp.avg_score * 0.70,
                ))

    return patches


# ══════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════

def _compute_template_similarity(aw1: dict, aw2: dict) -> float:
    """Compute overlap between two applicable_when conditions."""
    if not aw1 or not aw2:
        return 0.0

    keys = set(aw1.keys()) | set(aw2.keys())
    if not keys:
        return 0.0

    matches = 0
    total = len(keys)
    for k in keys:
        v1 = aw1.get(k)
        v2 = aw2.get(k)
        if k == "intent":
            if v1 == v2:
                matches += 1
        elif isinstance(v1, list) and isinstance(v2, list):
            overlap = len(set(v1) & set(v2))
            union = len(set(v1) | set(v2))
            matches += overlap / union if union else 0
        elif v1 == v2:
            matches += 1

    return matches / total


def _compute_skill_similarity(skill_a: str, skill_b: str, pattern: SuccessPattern | None = None) -> float:
    """Estimate structural similarity between two skills."""
    # Simple heuristic based on shared query structures and evidence types
    similarity = 0.0

    # Both are query types
    query_like = {"property_query", "reaction_condition_query", "mechanism_query",
                  "structure_query", "synthesis_route_query"}
    entity_like = {"entity_lookup", "alias_resolution", "entity_resolution"}
    comparison_like = {"reaction_comparison", "property_comparison"}

    for group in (query_like, entity_like, comparison_like):
        if skill_a in group and skill_b in group:
            similarity += 0.5
            break

    # Shared evidence handling
    if pattern:
        if pattern.evidence_type in ("table_evidence", "entity_evidence"):
            similarity += 0.2

    # Shared query structure
    if pattern and pattern.query_structure:
        similarity += 0.1

    return min(similarity, 1.0)
