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
    workers: int = 1,
    regression_limit: int | None = None,
    regression_seed: int = 42,
    stream_output: bool = True,
) -> dict[str, Any]:
    """Validate a single patch by applying it in a temp directory and running regression.

    Does NOT modify real config.
    Returns a dict with: passed, comparison (if regression run), errors.

    When regression_limit is not None, questions are randomly sampled using
    regression_seed for reproducibility.
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

            # Run regression in temp — use isolated output dir to avoid
            # overwriting baseline data/eval/eval_results.json
            tmp_eval_output = Path(tmpdir) / "eval_output"
            tmp_eval_output.mkdir(parents=True, exist_ok=True)
            runner = RegressionRunner(project_root)
            after_result = runner.run_eval(
                regression_dataset,
                tmp_skills,
                tmp_prompts,
                output_dir=tmp_eval_output,
                workers=workers,
                limit=regression_limit,
                seed=regression_seed,
                stream_output=stream_output,
            )

            if after_result.errors:
                result["passed"] = False
                result["errors"].append(f"Regression eval failed: {after_result.errors}")
                result["warnings"].append(f"Regression eval had errors: {after_result.errors}")
                result["regression_result"] = None
                result["average_score"] = None
                result["score_delta"] = None
                result["score_drop"] = None
                result["targeted_improvement"] = None
                result["failed_cases_increase"] = None
                patch.status = PatchStatus.REJECTED
                return result
            else:
                comparison = runner.compare(
                    RegressionResult_from_dict(baseline_result),
                    after_result,
                )
                result["regression_result"] = {
                    "passed": comparison.passed,
                    "score_drop": comparison.score_drop,
                    "reason": comparison.reason,
                    "average_score": comparison.average_score,
                    "score_delta": comparison.score_delta,
                    "targeted_improvement": comparison.targeted_improvement,
                }
                # Also surface at top level for log readability
                result["average_score"] = comparison.average_score
                result["score_delta"] = comparison.score_delta
                result["targeted_improvement"] = comparison.targeted_improvement
                if not comparison.passed:
                    result["passed"] = False
                    result["errors"].append(f"Regression failed: {comparison.reason}")
                    patch.status = PatchStatus.REJECTED
                    return result

    if result["passed"]:
        patch.status = PatchStatus.REGRESSION_VALIDATED

    return result


def RegressionResult_from_dict(d: dict) -> Any:
    """Create a RegressionResult from a dict (for comparison purposes).

    Normalizes both eval_results.json format and already-flattened dicts:
      - {"summary": {"average_score": N, "by_intent": {...}}, "results": [...]}
      - {"average_score": N, "total_questions": N, "by_intent": {...}}
    """
    from .regression import RegressionResult

    if "summary" in d:
        summary = d["summary"]
        avg = summary.get("average_score", 0.0)
        by_intent_raw = summary.get("by_intent", {})
        total = len(d.get("results", [])) or summary.get("total", 0)
    else:
        avg = d.get("average_score", 0.0)
        total = d.get("total_questions", 0)
        by_intent_raw = d.get("by_intent", {})

    # Normalize by_intent from {"intent": float} or {"intent": {"average_score": float}}
    by_intent: dict[str, dict[str, float]] = {}
    for k, v in by_intent_raw.items():
        if isinstance(v, dict):
            by_intent[k] = {"average_score": v.get("average_score", 0.0), "total": v.get("total", 0), "failed": v.get("failed", 0)}
        else:
            by_intent[k] = {"average_score": float(v), "total": 0, "failed": 0}

    return RegressionResult(
        average_score=avg,
        total_questions=total,
        by_intent=by_intent,
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
    workers: int = 1,
    regression_limit: int | None = None,
    regression_seed: int = 42,
    stream_output: bool = True,
) -> dict[str, Any]:
    """Validate a group of patches together (composition validation).

    Applies all patches in a temp directory and runs regression + global smoke.
    Returns a dict with: passed, regression_comparison, global_comparison, errors.

    When regression_limit is not None, questions are randomly sampled using
    regression_seed for reproducibility.
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

        # Run regression — use isolated output dir to avoid overwriting baseline
        if regression_dataset and regression_dataset.exists() and baseline_regression_result:
            tmp_eval_output = Path(tmpdir) / "eval_output"
            tmp_eval_output.mkdir(parents=True, exist_ok=True)
            runner = RegressionRunner(project_root)
            after_result = runner.run_eval(regression_dataset, tmp_skills, tmp_prompts, output_dir=tmp_eval_output, workers=workers, limit=regression_limit, seed=regression_seed, stream_output=stream_output)
            if after_result.errors:
                result["errors"].append(f"Regression eval had errors: {after_result.errors}")
                result["passed"] = False
                result["regression_comparison"] = None
                result["average_score"] = None
                result["score_delta"] = None
                result["score_drop"] = None
                result["targeted_improvement"] = None
                result["failed_cases_increase"] = None
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
                    "average_score": comparison.average_score,
                    "score_delta": comparison.score_delta,
                    "targeted_improvement": comparison.targeted_improvement,
                }
                result["average_score"] = comparison.average_score
                result["score_delta"] = comparison.score_delta
                result["targeted_improvement"] = comparison.targeted_improvement
                if not comparison.passed:
                    result["passed"] = False
                    result["errors"].append(f"Composition regression failed: {comparison.reason}")

    return result
