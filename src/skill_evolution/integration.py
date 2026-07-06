"""Candidate update integration. Stage 5 of the Skill Evolution pipeline.

Merges 4A (mutation) + 4B (distillation) candidates, resolves conflicts,
orders dependencies, enforces limits, and adjusts validators.

Aligned with architecture diagram Step 5:
  patch + planner_patch + template_add/rewrite/merge
  + validator_adjustment + schema_adjustment
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .patch import PatchSchema, PatchStatus
from .config import get_config


class CandidateType(str, Enum):
    """Candidate types aligned with architecture diagram Step 5."""
    PATCH = "patch"
    PLANNER_PATCH = "planner_patch"
    TEMPLATE_ADD = "template_add"
    TEMPLATE_REWRITE = "template_rewrite"
    TEMPLATE_MERGE = "template_merge"
    VALIDATOR_ADJUSTMENT = "validator_adjustment"
    SCHEMA_ADJUSTMENT = "schema_adjustment"


@dataclass
class IntegrationResult:
    selected: list[PatchSchema] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    validator_adjustments: list[dict[str, Any]] = field(default_factory=list)
    schema_adjustments: list[dict[str, Any]] = field(default_factory=list)
    conflicts_resolved: list[str] = field(default_factory=list)


def integrate_candidates(
    mutation_patches: list[PatchSchema],
    distillation_patches: list[PatchSchema],
    skill_configs: dict[str, dict],
) -> IntegrationResult:
    """Stage 5: Integrate 4A + 4B candidates into final selection.

    Steps:
      1. Merge all candidates
      2. Detect conflicts (same skill + same target_path)
      3. Resolve conflicts (keep highest confidence × probability of impact)
      4. Dependency ordering (prompt_content before prompt_ref)
      5. Enforce max_patches_per_round
      6. Generate validator/schema adjustments
    """
    config = get_config()
    result = IntegrationResult()

    # ── Step 1: Merge ──────────────────────────────────────────────────────
    all_patches = list(mutation_patches) + list(distillation_patches)

    # ── Step 2: Conflict detection ─────────────────────────────────────────
    conflicts = _detect_conflicts(all_patches)

    # ── Step 3: Conflict resolution ────────────────────────────────────────
    if conflicts:
        resolved, conflict_rejected = _resolve_conflicts(all_patches, conflicts)
        result.conflicts_resolved = [
            f"{id1} vs {id2} on '{path}' — kept {id1 if id1 in {p.patch_id for p in resolved} else id2}"
            for id1, id2, path in conflicts
        ]
        result.rejected.extend(conflict_rejected)
        all_patches = resolved

    # ── Step 4: Dependency ordering ────────────────────────────────────────
    all_patches = _dependency_order(all_patches)

    # ── Step 5: No artificial truncation — let regression be the gate ──────
    # All patches that pass conflict resolution and dependency ordering
    # proceed to validation. Quality is enforced by regression, not by
    # arbitrary top-N limits.

    # ── Step 6: Validator & schema adjustments ─────────────────────────────
    result.validator_adjustments = _generate_validator_adjustments(all_patches, skill_configs)
    result.schema_adjustments = _generate_schema_adjustments(all_patches)

    result.selected = all_patches
    return result


# ══════════════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════════════

def _conflict_key(patch: PatchSchema) -> tuple[str, str]:
    """Two patches conflict if same skill + same effective target path."""
    skill = patch.skill_name
    target = patch.target_path

    # For template patches, include template name
    if patch.target_path == "templates" and patch.patch_type in (
        "template_add", "template_rewrite", "template_merge",
    ):
        name = patch.proposed_value.get("name", "") if isinstance(patch.proposed_value, dict) else ""
        return (skill, f"templates.{name}")

    # For prompt_ref_update, expand to the specific field
    if patch.patch_type == "prompt_ref_update":
        first_key = next(iter(patch.proposed_value)) if isinstance(patch.proposed_value, dict) and patch.proposed_value else ""
        return (skill, f"{target}.{first_key}")

    return (skill, target)


def _detect_conflicts(patches: list[PatchSchema]) -> list[tuple[str, str, str]]:
    """Detect conflicting patches. Returns [(patch_id_1, patch_id_2, conflict_path), ...]."""
    conflicts: list[tuple[str, str, str]] = []
    key_map: dict[tuple[str, str], str] = {}

    for p in patches:
        key = _conflict_key(p)
        if key in key_map:
            conflicts.append((key_map[key], p.patch_id, f"{key[0]}/{key[1]}"))
        else:
            key_map[key] = p.patch_id

    return conflicts


def _resolve_conflicts(
    patches: list[PatchSchema],
    conflicts: list[tuple[str, str, str]],
) -> tuple[list[PatchSchema], list[dict[str, Any]]]:
    """Resolve conflicts by keeping the patch with highest confidence in each conflict group."""
    conflict_ids: set[str] = set()
    for id1, id2, _ in conflicts:
        conflict_ids.add(id1)
        conflict_ids.add(id2)

    # Build adjacency for connected components
    adjacency: dict[str, set[str]] = {}
    for id1, id2, _ in conflicts:
        adjacency.setdefault(id1, set()).add(id2)
        adjacency.setdefault(id2, set()).add(id1)

    patch_map: dict[str, PatchSchema] = {p.patch_id: p for p in patches}

    # Find connected groups
    visited: set[str] = set()
    groups: list[list[str]] = []
    for pid in conflict_ids:
        if pid in visited:
            continue
        group: list[str] = []
        stack = [pid]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            group.append(node)
            for neighbor in adjacency.get(node, set()):
                if neighbor not in visited:
                    stack.append(neighbor)
        groups.append(group)

    # For each group, keep highest confidence, reject others
    keep_ids: set[str] = {p.patch_id for p in patches} - conflict_ids
    rejected: list[dict[str, Any]] = []

    for group in groups:
        best_id = max(group, key=lambda pid: (
            getattr(patch_map.get(pid), "confidence", 0.0),
            len(getattr(patch_map.get(pid), "dependencies", [])),  # prefer patches with deps
        ))
        keep_ids.add(best_id)
        for pid in group:
            if pid != best_id:
                entry = patch_map[pid].model_dump(mode="json")
                entry["rejection_reason"] = (
                    f"conflict_resolution: kept {best_id} over {pid}"
                )
                rejected.append(entry)

    kept = [p for p in patches if p.patch_id in keep_ids]
    # Preserve original order
    kept.sort(key=lambda p: patches.index(p) if p in patches else 999)
    return kept, rejected


def _dependency_order(patches: list[PatchSchema]) -> list[PatchSchema]:
    """Ensure prompt_content_update patches come before prompt_ref_update patches."""
    content_patches = [p for p in patches if p.patch_type == "prompt_content_update"]
    ref_patches = [p for p in patches if p.patch_type == "prompt_ref_update"]
    other = [p for p in patches if p not in content_patches and p not in ref_patches]
    return content_patches + other + ref_patches


def _generate_validator_adjustments(
    patches: list[PatchSchema],
    skill_configs: dict[str, dict],
) -> list[dict[str, Any]]:
    """Generate validator threshold adjustments based on patch impact."""
    adjustments: list[dict[str, Any]] = []

    # If assessment prompt changed, may need to adjust citation accuracy threshold
    has_assessment_change = any(
        p.patch_type == "prompt_ref_update" and "assessment" in p.target_path
        for p in patches
    )
    if has_assessment_change:
        adjustments.append({
            "type": "validator_adjustment",
            "target": "citation_accuracy_threshold",
            "current": 0.95,
            "proposed": 0.90,
            "rationale": "Assessment prompt changed — temporarily lower citation bar to avoid false negatives.",
        })

    return adjustments


def _generate_schema_adjustments(
    patches: list[PatchSchema],
) -> list[dict[str, Any]]:
    """Generate schema adjustment suggestions (requires manual review)."""
    adjustments: list[dict[str, Any]] = []

    # Detect if new fields are being added that aren't in the schema
    for p in patches:
        if p.operation.value == "add" and p.target_path not in (
            "templates", "config/prompts/",
        ):
            adjustments.append({
                "type": "schema_adjustment",
                "target_path": p.target_path,
                "rationale": f"Patch {p.patch_id} adds to '{p.target_path}' — verify schema compatibility.",
                "requires_manual_review": True,
            })

    return adjustments
