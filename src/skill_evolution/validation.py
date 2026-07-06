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
    rejection_category: str = ""  # rejected_schema_error, rejected_semantic_error, rejected_risk, rejected_low_quality


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

    Note: prompt_content_update patches skip skill-level schema checks
    since they modify prompt files, not skill YAML.
    """
    result = ValidationResult(patch_id=patch.patch_id)

    # Skip skill-level schema checks for prompt file patches
    if patch.patch_type in ("prompt_content_update",):
        result.valid = True
        return result

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
    prompts_dir: Path | None = None,
    all_patches: list | None = None,
) -> ValidationResult:
    """Check semantic invariants that go beyond YAML schema."""
    result = ValidationResult(patch_id=patch.patch_id)

    # For prompt_content_update patches (add operation on prompt files),
    # skip skill-level checks since they don't modify skill YAML
    is_prompt_file_patch = patch.patch_type in (
        "prompt_content_update",
    )

    if not is_prompt_file_patch:
        # skill_name must match filename (only for skill YAML patches)
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
        # Check the ref format is valid
        if not re.match(r"^[A-Z][A-Z0-9_]+$", new_ref):
            result.warnings.append(f"Prompt ref '{new_ref}' doesn't follow UPPER_SNAKE_CASE convention.")

        # Check if the referenced prompt file actually exists on disk,
        # or if a prompt_artifact in this patch (or its dependencies) will create it.
        if prompts_dir is not None and prompts_dir.is_dir():
            # Build set of prompt_refs that will be created by this patch's artifacts
            # AND by dependency patches' artifacts
            will_be_created: set[str] = set()
            for artifact in patch.prompt_artifacts:
                will_be_created.add(artifact.prompt_ref)
                if artifact.registry_path:
                    will_be_created.add(Path(artifact.registry_path).stem.upper())

            # Also check dependency patches' prompt_artifacts
            if all_patches and patch.dependencies:
                dep_map = {p.patch_id: p for p in all_patches}
                for dep_id in patch.dependencies:
                    dep_patch = dep_map.get(dep_id)
                    if dep_patch:
                        for artifact in dep_patch.prompt_artifacts:
                            will_be_created.add(artifact.prompt_ref)
                            if artifact.registry_path:
                                will_be_created.add(Path(artifact.registry_path).stem.upper())

            # Check if the prompt exists on disk
            prompt_exists = False
            for yaml_file in prompts_dir.glob("*.yaml"):
                try:
                    import yaml as _yaml
                    data = _yaml.safe_load(yaml_file.read_text("utf-8"))
                    if data and data.get("prompt_ref") == new_ref:
                        prompt_exists = True
                        break
                except Exception:
                    pass

            if not prompt_exists and new_ref not in will_be_created:
                result.valid = False
                result.rejection_category = "rejected_semantic_error"
                result.errors.append(
                    f"Prompt ref '{new_ref}' does not exist in {prompts_dir} "
                    f"and no prompt_artifact in this patch will create it. "
                    f"Cannot validate a patch that references a non-existent prompt."
                )

    # For add operations on prompt files, check parent directory exists
    if patch.operation == PatchOperation.ADD and patch.patch_type == "prompt_content_update":
        target_file = patch.target_file or patch.target_path
        if target_file:
            parent_dir = Path(target_file).parent
            if not parent_dir.exists():
                # Check if it's a relative path
                parent_dir_alt = Path("config/prompts")
                if not parent_dir_alt.exists():
                    result.valid = False
                    result.rejection_category = "rejected_semantic_error"
                    result.errors.append(f"Parent directory for new prompt file does not exist: {parent_dir}")

            # Check prompt_ref doesn't conflict with existing prompts
            if prompts_dir and prompts_dir.is_dir():
                for artifact in patch.prompt_artifacts:
                    new_ref = artifact.prompt_ref
                    for yaml_file in prompts_dir.glob("*.yaml"):
                        try:
                            import yaml as _yaml
                            data = _yaml.safe_load(yaml_file.read_text("utf-8"))
                            if data and data.get("prompt_ref") == new_ref:
                                result.valid = False
                                result.rejection_category = "rejected_semantic_error"
                                result.errors.append(
                                    f"Prompt ref '{new_ref}' already exists in {yaml_file.name}. "
                                    f"Cannot add duplicate prompt_ref."
                                )
                                break
                        except Exception:
                            pass

    # Validate prompt content is non-empty for prompt patches
    if patch.patch_type == "prompt_content_update":
        for artifact in patch.prompt_artifacts:
            if not artifact.content or len(artifact.content.strip()) < 50:
                result.valid = False
                result.rejection_category = "rejected_semantic_error"
                result.errors.append("Prompt content is empty or too short (< 50 chars).")

    # Validate template patches have trigger conditions
    if patch.target_path == "templates" and patch.operation == PatchOperation.ADD:
        pv = patch.proposed_value
        if isinstance(pv, dict):
            # Check for empty template name
            if not pv.get("name"):
                result.valid = False
                result.rejection_category = "rejected_semantic_error"
                result.errors.append("Template patch must have a non-empty 'name' field.")
            # Check for trigger conditions
            applicable_when = pv.get("applicable_when", {})
            if not applicable_when or (len(applicable_when) == 1 and "intent" in applicable_when and not any(
                k for k in applicable_when if k != "intent"
            )):
                # Only intent and no other trigger conditions — this is too weak
                if not pv.get("slots") and not pv.get("examples"):
                    result.warnings.append(
                        "Template patch has minimal trigger conditions (intent only) "
                        "and no slots or examples. Consider adding more specificity."
                    )

    # Validate that rationale is not empty
    if not patch.rationale or len(patch.rationale.strip()) < 10:
        result.warnings.append("Patch rationale is too short or empty. Provide a meaningful explanation.")

    # Validate that source_failure_ids are not empty for failure-driven patches
    if patch.target_path != "templates" and not patch.source_failure_ids and not is_prompt_file_patch:
        result.warnings.append("Patch has no source_failure_ids — it's not linked to any specific failures.")

    # Validate dependencies: prompt_ref_update must depend on prompt_content_update
    if patch.patch_type == "prompt_ref_update":
        if not patch.dependencies:
            result.warnings.append(
                "prompt_ref_update patch has no dependencies. "
                "It should depend on the corresponding prompt_content_update patch."
            )

    # Validate parameter value ranges for numeric patches
    if patch.patch_type in ("retrieval_strategy_update", "query_rewrite_update",
                            "evidence_expansion_update", "answer_generation_update"):
        new_val = patch.new_value
        if isinstance(new_val, (int, float)):
            if new_val < 0:
                result.valid = False
                result.rejection_category = "rejected_semantic_error"
                result.errors.append(f"Numeric value cannot be negative: {new_val}")
            if isinstance(new_val, float) and new_val > 100:
                result.warnings.append(f"Unusually large numeric value: {new_val}")

    return result


def validate_patch(
    patch: PatchSchema,
    skill_config: dict,
    mutable_paths: list[str] | None = None,
    frozen_paths: list[str] | None = None,
    skill_filename: str = "",
    prompts_dir: Path | None = None,
    all_patches: list | None = None,
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
        if not result.rejection_category:
            result.rejection_category = "rejected_schema_error"
        return result

    semantic = validate_semantic_invariants(patch, skill_config, skill_filename, prompts_dir=prompts_dir, all_patches=all_patches)
    result.errors.extend(semantic.errors)
    result.warnings.extend(semantic.warnings)
    if not semantic.valid:
        result.valid = False
        if semantic.rejection_category:
            result.rejection_category = semantic.rejection_category
        elif not result.rejection_category:
            result.rejection_category = "rejected_semantic_error"

    # Mark as schema validated if passed
    if result.valid:
        patch.status = PatchStatus.SCHEMA_VALIDATED

    return result
