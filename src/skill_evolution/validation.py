"""Validation Pipeline for ChemEvoRAG Skill Evolution patches.

Implements §5.5 of the Skill Evolution design spec:
  - Schema validation
  - Semantic invariant checks
  - Individual regression validation (placeholder)
  - Composition validation (placeholder)

Usage:
    from skill_evolution.validation import validate_patch, ValidationResult
    result = validate_patch(patch, skill_config)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

try:
    from pydantic import BaseModel, ConfigDict, Field
except ImportError:
    raise ImportError("pydantic is required for skill_evolution.validation")

from .attribution import FailureType
from .patch import PatchOperation, PatchSchema, PatchStatus


# ---------------------------------------------------------------------------
# Validation result model
# ---------------------------------------------------------------------------

class _BaseModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True, validate_assignment=True, extra="forbid")


class ValidationResult(_BaseModel):
    patch_id: str = ""
    valid: bool = True
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# §3.2 Pydantic validation rules
# ---------------------------------------------------------------------------

def _get_nested(d: dict, path: str) -> Any:
    """Get nested value by dot-separated path."""
    parts = path.split(".")
    current = d
    for p in parts:
        if isinstance(current, dict) and p in current:
            current = current[p]
        else:
            return None
    return current


def _is_parent_of(child: str, parent: str) -> bool:
    """Check if child path is a descendant of parent path."""
    return child.startswith(parent + ".") or child == parent


def validate_schema(
    patch: PatchSchema,
    skill_config: dict,
    mutable_paths: list[str],
    frozen_paths: list[str],
) -> ValidationResult:
    """Validate patch against skill schema constraints.

    Checks:
    1. target_path or target_path + allowed_fields must match some mutable_path
    2. frozen_paths must not be modified
    3. Required fields not deleted
    """
    result = ValidationResult(patch_id=patch.patch_id)

    target = patch.target_path

    # Check frozen paths
    for frozen in frozen_paths:
        if _is_parent_of(frozen, target) or target == frozen:
            result.valid = False
            result.errors.append(f"Frozen path '{frozen}' cannot be modified by patch targeting '{target}'.")
            return result

    # Check target_path is under a mutable_path or matches
    is_mutable = False
    for mp in mutable_paths:
        if mp == target or _is_parent_of(mp, target) or _is_parent_of(target, mp):
            is_mutable = True
            break
    if not is_mutable:
        result.valid = False
        result.errors.append(f"Target path '{target}' is not under any mutable_path.")
        return result

    # Check proposed field paths are mutable
    for field_key in patch.proposed_value:
        full_path = f"{target}.{field_key}"
        field_mutable = False
        for mp in mutable_paths:
            if mp == full_path or _is_parent_of(full_path, mp):
                field_mutable = True
                break
        if not field_mutable:
            result.warnings.append(f"Field '{full_path}' is not explicitly in mutable_paths.")

    # Check operation semantics
    if patch.operation == PatchOperation.DELETE:
        for key in patch.proposed_value:
            full_path = f"{target}.{key}"
            if "system_prompt_ref" in key or "primary_channels" in key:
                result.valid = False
                result.errors.append(f"Cannot delete required field '{full_path}'.")
                return result

    return result


def validate_semantic_invariants(
    patch: PatchSchema,
    skill_config: dict,
    skill_filename: str,
) -> ValidationResult:
    """Check semantic invariants that go beyond YAML schema."""
    result = ValidationResult(patch_id=patch.patch_id)

    # skill_name must match filename
    expected_name = skill_filename.replace(".yaml", "").replace(".yml", "")
    if skill_config.get("name") != expected_name:
        result.valid = False
        result.errors.append(f"Skill name '{skill_config.get('name')}' doesn't match filename '{expected_name}'.")

    # trigger.intent must match skill name
    intent = skill_config.get("trigger", {}).get("intent", "")
    if intent != skill_config.get("name"):
        result.valid = False
        result.errors.append(f"trigger.intent '{intent}' doesn't match skill name '{skill_config.get('name')}'.")

    # Check prompt refs exist (if patch modifies system_prompt_ref)
    if "system_prompt_ref" in patch.proposed_value:
        new_ref = patch.proposed_value["system_prompt_ref"]
        # We can't check registry existence here (it's loaded separately),
        # but we can check the ref format is valid
        if not re.match(r"^[A-Z][A-Z0-9_]+$", new_ref):
            result.warnings.append(f"Prompt ref '{new_ref}' doesn't follow UPPER_SNAKE_CASE convention.")

    return result


def validate_patch(
    patch: PatchSchema,
    skill_config: dict,
    mutable_paths: list[str] | None = None,
    frozen_paths: list[str] | None = None,
    skill_filename: str = "",
) -> ValidationResult:
    """Full validation: schema + semantic invariants.

    Returns a ValidationResult indicating whether the patch is safe to apply.
    """
    if mutable_paths is None:
        mutable_paths = skill_config.get("evolution", {}).get("mutable_paths", [])
    if frozen_paths is None:
        frozen_paths = skill_config.get("evolution", {}).get("frozen_paths", [])

    result = validate_schema(patch, skill_config, mutable_paths, frozen_paths)
    if not result.valid:
        return result

    semantic = validate_semantic_invariants(patch, skill_config, skill_filename)
    result.errors.extend(semantic.errors)
    result.warnings.extend(semantic.warnings)
    if not semantic.valid:
        result.valid = False

    # Mark as schema validated if passed
    if result.valid:
        patch.status = PatchStatus.SCHEMA_VALIDATED

    return result
