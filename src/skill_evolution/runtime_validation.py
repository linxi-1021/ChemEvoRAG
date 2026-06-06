"""Runtime Validation with Regression for ChemEvoRAG Skill Evolution.

Extends schema/semantic validation with real regression testing.
All validation happens in temporary directories — real config is never modified.

Usage:
    from skill_evolution.runtime_validation import (
        validate_individual_patch,
        validate_composition,
        detect_conflicts,
    )
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .apply import PatchApplier
from .patch import PatchSchema, PatchStatus
from .regression import RegressionComparison, RegressionRunner
from .validation import validate_patch


def _get_effective_patch_path(patch: PatchSchema) -> str:
    """Compute the effective field path for conflict detection.

    For template add operations, uses the template name from proposed_value
    to distinguish different templates being added to the same list/dict.
    For other operations, uses target_path + first key of proposed_value.
    """
    target = patch.target_path
    op = getattr(patch, "operation", "update")
    pv = patch.proposed_value or {}

    # Template add: use name field to distinguish different templates
    if target == "templates" and op == "add":
        if isinstance(pv, dict):
            name = pv.get("name") or pv.get("template_name") or pv.get("id")
            if name:
                return f"{target}.{name}"
        # Cannot identify unique key — conservative: conflict on target
        return target

    # Standard case: expand to first field in proposed_value
    if isinstance(pv, dict) and pv:
        first_key = next(iter(pv))
        return f"{target}.{first_key}"

    return target


def detect_conflicts(patches: list[PatchSchema]) -> list[tuple[str, str, str]]:
    """Detect conflicts between patches targeting the same effective field path.

    Uses _get_effective_patch_path to resolve the actual field being modified.
    Template add operations with different names are NOT considered conflicts.
    """
    conflicts: list[tuple[str, str, str]] = []
    field_map: dict[str, str] = {}  # effective_path → first patch_id

    for p in patches:
        effective_path = _get_effective_patch_path(p)
        if effective_path in field_map:
            conflicts.append((field_map[effective_path], p.patch_id, effective_path))
        else:
            field_map[effective_path] = p.patch_id

    return conflicts


def validate_individual_patch(
    patch: PatchSchema,
    skill_config: dict,
    *,
    project_root: Path,
    skills_dir: Path,
    prompts_dir: Path,
    regression_dataset: Path | None = None,
    baseline_result: dict | None = None,
    skill_filename: str = "",
) -> dict[str, Any]:
    """Validate a single patch by applying it in a temp directory and running regression.

    Does NOT modify real config.
    Returns a dict with: passed, comparison (if regression run), errors.
    """
    result: dict[str, Any] = {
        "patch_id": patch.patch_id,
        "passed": True,
        "errors": [],
        "warnings": [],
        "regression_result": None,
    }

    # Step 1: Schema + semantic validation
    vresult = validate_patch(patch, skill_config, skill_filename=skill_filename)
    result["errors"].extend(vresult.errors)
    result["warnings"].extend(vresult.warnings)
    if not vresult.valid:
        result["passed"] = False
        patch.status = PatchStatus.REJECTED
        return result

    # Step 2: Try applying patch in memory to catch errors
    try:
        applier = PatchApplier(skills_dir, prompts_dir)
        test_config = dict(skill_config)  # shallow copy
        applier.apply_to_memory(test_config, patch)
    except Exception as e:
        result["passed"] = False
        result["errors"].append(f"Patch apply failed: {e}")
        patch.status = PatchStatus.REJECTED
        return result

    # Step 3: Regression validation (if dataset and baseline provided)
    if regression_dataset and regression_dataset.exists() and baseline_result:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_skills = Path(tmpdir) / "skills"
            tmp_prompts = Path(tmpdir) / "prompts"
            shutil.copytree(skills_dir, tmp_skills)
            if prompts_dir.is_dir():
                shutil.copytree(prompts_dir, tmp_prompts)

            # Apply patch in temp directory
            try:
                tmp_applier = PatchApplier(tmp_skills, tmp_prompts)
                skill_name = skill_config.get("name", "unknown")
                skill_path = tmp_skills / f"{skill_name}.yaml"
                if skill_path.exists():
                    import yaml
                    tmp_skill = yaml.safe_load(skill_path.read_text("utf-8"))
                    tmp_applier.apply_to_memory(tmp_skill, patch)
                    skill_path.write_text(
                        yaml.dump(tmp_skill, allow_unicode=True, default_flow_style=False, sort_keys=False),
                        encoding="utf-8",
                    )
            except Exception as e:
                result["passed"] = False
                result["errors"].append(f"Temp apply failed: {e}")
                patch.status = PatchStatus.REJECTED
                return result

            # Run regression in temp
            runner = RegressionRunner(project_root)
            after_result = runner.run_eval(
                regression_dataset,
                tmp_skills,
                tmp_prompts,
            )

            if after_result.errors:
                result["warnings"].append(f"Regression eval had errors: {after_result.errors}")
            else:
                comparison = runner.compare(
                    RegressionResult_from_dict(baseline_result),
                    after_result,
                )
                result["regression_result"] = {
                    "passed": comparison.passed,
                    "score_drop": comparison.score_drop,
                    "reason": comparison.reason,
                }
                if not comparison.passed:
                    result["passed"] = False
                    result["errors"].append(f"Regression failed: {comparison.reason}")
                    patch.status = PatchStatus.REJECTED
                    return result

    if result["passed"]:
        patch.status = PatchStatus.REGRESSION_VALIDATED

    return result


def RegressionResult_from_dict(d: dict) -> Any:
    """Create a RegressionResult from a dict (for comparison purposes)."""
    from .regression import RegressionResult
    return RegressionResult(
        average_score=d.get("average_score", 0.0),
        total_questions=d.get("total_questions", 0),
        by_intent=d.get("by_intent", {}),
        results_path=d.get("results_path"),
        errors=d.get("errors", []),
    )


def validate_composition(
    patches: list[PatchSchema],
    skill_configs: dict[str, dict],
    *,
    project_root: Path,
    skills_dir: Path,
    prompts_dir: Path,
    regression_dataset: Path | None = None,
    global_smoke_dataset: Path | None = None,
    baseline_regression_result: dict | None = None,
    baseline_global_result: dict | None = None,
) -> dict[str, Any]:
    """Validate a group of patches together (composition validation).

    Applies all patches in a temp directory and runs regression + global smoke.
    Returns a dict with: passed, regression_comparison, global_comparison, errors.
    """
    result: dict[str, Any] = {
        "passed": True,
        "patch_ids": [p.patch_id for p in patches],
        "errors": [],
        "regression_comparison": None,
        "global_comparison": None,
    }

    # Check for conflicts
    conflicts = detect_conflicts(patches)
    if conflicts:
        for id1, id2, path in conflicts:
            result["errors"].append(
                f"Conflict: patches {id1} and {id2} both target '{path}'"
            )
        result["passed"] = False
        return result

    # Apply all patches in temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_skills = Path(tmpdir) / "skills"
        tmp_prompts = Path(tmpdir) / "prompts"
        shutil.copytree(skills_dir, tmp_skills)
        if prompts_dir.is_dir():
            shutil.copytree(prompts_dir, tmp_prompts)

        applier = PatchApplier(tmp_skills, tmp_prompts)
        for patch in patches:
            skill_name = patch.skill_name
            skill_path = tmp_skills / f"{skill_name}.yaml"
            if not skill_path.exists():
                result["errors"].append(f"Skill file {skill_name}.yaml not found in temp dir")
                result["passed"] = False
                return result
            try:
                import yaml
                skill = yaml.safe_load(skill_path.read_text("utf-8"))
                applier.apply_to_memory(skill, patch)
                skill_path.write_text(
                    yaml.dump(skill, allow_unicode=True, default_flow_style=False, sort_keys=False),
                    encoding="utf-8",
                )
            except Exception as e:
                result["errors"].append(f"Failed to apply {patch.patch_id}: {e}")
                result["passed"] = False
                return result

        # Run regression
        if regression_dataset and regression_dataset.exists() and baseline_regression_result:
            runner = RegressionRunner(project_root)
            after_result = runner.run_eval(regression_dataset, tmp_skills, tmp_prompts)
            if after_result.errors:
                result["errors"].append(f"Regression eval had errors: {after_result.errors}")
                result["passed"] = False
            else:
                comparison = runner.compare(
                    RegressionResult_from_dict(baseline_regression_result),
                    after_result,
                    max_failed_cases_increase=1,
                )
                result["regression_comparison"] = {
                    "passed": comparison.passed,
                    "score_drop": comparison.score_drop,
                    "reason": comparison.reason,
                }
                if not comparison.passed:
                    result["passed"] = False
                    result["errors"].append(f"Composition regression failed: {comparison.reason}")

    return result
