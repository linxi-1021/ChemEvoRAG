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


def _conflict_key(patch: PatchSchema) -> tuple[str, str]:
    """Build a conflict key that distinguishes patches by skill + effective path.

    Two patches only conflict if they modify the same effective field path
    **in the same skill file**.  Different skills can safely modify the same
    field path (e.g. two skills both updating strategy.assessment).
    """
    skill = getattr(patch, "skill_name", "") or ""
    return (skill, _get_effective_patch_path(patch))


def detect_conflicts(patches: list[PatchSchema]) -> list[tuple[str, str, str]]:
    """Detect conflicts between patches targeting the same effective field path
    in the same skill.

    Uses _conflict_key which combines skill_name + effective_path so that
    two patches modifying different skill files are never flagged as conflicting.
    Template add operations with different names are also NOT considered conflicts.
    """
    conflicts: list[tuple[str, str, str]] = []
    field_map: dict[tuple[str, str], str] = {}  # (skill, effective_path) → first patch_id

    for p in patches:
        key = _conflict_key(p)
        if key in field_map:
            conflicts.append((field_map[key], p.patch_id, f"{key[0]}/{key[1]}"))
        else:
            field_map[key] = p.patch_id

    return conflicts


def resolve_conflicts(
    patches: list[PatchSchema],
    *,
    individual_results: list[dict] | None = None,
) -> tuple[list[PatchSchema], list[dict]]:
    """Resolve conflicts among patches by keeping the best non-conflicting subset.

    Strategy:
    1. If no conflicts, return all patches unchanged.
    2. If conflicts exist, for each conflict group, keep the patch with the
       highest average_score (from individual_results) and reject the rest.
    3. Patches that don't conflict with anything are kept.

    Returns (kept_patches, rejected_entries) where rejected_entries is a list
    of dicts suitable for appending to rejected_patches.
    """
    conflicts = detect_conflicts(patches)
    if not conflicts:
        return list(patches), []

    # Build a lookup of patch_id → individual_result average_score
    score_map: dict[str, float] = {}
    if individual_results:
        for r in individual_results:
            pid = r.get("patch_id", "")
            score_map[pid] = r.get("average_score") or 0.0

    # Collect conflicting patch_ids per group
    conflict_ids: set[str] = set()
    for id1, id2, _ in conflicts:
        conflict_ids.add(id1)
        conflict_ids.add(id2)

    # Group conflicted patches: find connected components
    patch_map: dict[str, PatchSchema] = {p.patch_id: p for p in patches}
    adjacency: dict[str, set[str]] = {}
    for id1, id2, _ in conflicts:
        adjacency.setdefault(id1, set()).add(id2)
        adjacency.setdefault(id2, set()).add(id1)

    visited: set[str] = set()
    groups: list[list[str]] = []
    for pid in conflict_ids:
        if pid in visited:
            continue
        # BFS to find connected component
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

    # For each group, keep the highest-score patch, reject the rest
    keep_ids: set[str] = {p.patch_id for p in patches} - conflict_ids  # non-conflicting always kept
    rejected: list[dict] = []
    for group in groups:
        # Sort by score desc, then by confidence desc
        best_id = max(group, key=lambda pid: (score_map.get(pid, 0.0),
                                               getattr(patch_map.get(pid), "confidence", 0.0)))
        keep_ids.add(best_id)
        for pid in group:
            if pid != best_id:
                p = patch_map[pid]
                entry = p.model_dump(mode="json")
                entry["rejection_reason"] = (
                    f"composition_conflict_resolution: "
                    f"conflicted with {[x for x in group if x != pid]} "
                    f"on path '{_get_effective_patch_path(p)}' in skill '{p.skill_name}'; "
                    f"kept {best_id} (score={score_map.get(best_id, 0):.4f})"
                )
                rejected.append(entry)

    kept = [patch_map[pid] for pid in keep_ids]
    # Preserve original order
    kept.sort(key=lambda p: patches.index(p))
    return kept, rejected


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
    persist_dir: Path | None = None,
) -> dict[str, Any]:
    """Validate a single patch by applying it in a temp directory and running regression.

    Does NOT modify real config.
    Returns a dict with: passed, comparison (if regression run), errors.

    When regression_limit is not None, questions are randomly sampled using
    regression_seed for reproducibility.

    When persist_dir is provided, eval output (eval_results.json, react_log.txt,
    eval_stdout.log) is saved there instead of a temp directory that gets deleted.
    """
    result: dict[str, Any] = {
        "patch_id": patch.patch_id,
        "passed": True,
        "errors": [],
        "warnings": [],
        "regression_result": None,
    }

    # Step 1: Schema + semantic validation
    vresult = validate_patch(patch, skill_config, skill_filename=skill_filename, prompts_dir=prompts_dir)
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
        # Use persist_dir if provided, otherwise a temp dir that auto-cleans
        _tmpdir: str
        _should_cleanup: bool
        if persist_dir is not None:
            # Clean up from any previous run (always overwrite, never append)
            if persist_dir.exists():
                shutil.rmtree(persist_dir, ignore_errors=True)
            persist_dir.mkdir(parents=True, exist_ok=True)
            _tmpdir = str(persist_dir)
            _should_cleanup = False
        else:
            _tmpdir = tempfile.mkdtemp(prefix="patch_eval_")
            _should_cleanup = True
        try:
            tmpdir = Path(_tmpdir)
            tmp_skills = tmpdir / "skills"
            tmp_prompts = tmpdir / "prompts"
            shutil.copytree(skills_dir, tmp_skills)
            if prompts_dir.is_dir():
                shutil.copytree(prompts_dir, tmp_prompts)

            runner = RegressionRunner(project_root)

            # If regression_limit is set, run a "before" eval on the SAME
            # sampled questions using the ORIGINAL (unpatched) config FIRST,
            # BEFORE applying the patch.  This ensures apples-to-apples.
            if regression_limit is not None and regression_limit > 0:
                tmp_before_output = Path(tmpdir) / "baseline_output"
                tmp_before_output.mkdir(parents=True, exist_ok=True)
                before_result = runner.run_eval(
                    regression_dataset,
                    tmp_skills,   # ORIGINAL — patch not yet applied
                    tmp_prompts,
                    output_dir=tmp_before_output,
                    workers=workers,
                    limit=regression_limit,
                    seed=regression_seed,
                    stream_output=stream_output,
                )
                if before_result.errors:
                    result["warnings"].append(
                        f"Baseline subset eval had errors: {before_result.errors}"
                    )
                    baseline_for_comparison = (
                        RegressionResult_from_dict(baseline_result)
                        if baseline_result else before_result
                    )
                else:
                    baseline_for_comparison = before_result
            else:
                baseline_for_comparison = (
                    RegressionResult_from_dict(baseline_result)
                    if baseline_result else None
                )

            # Apply patch in temp directory (AFTER baseline eval)
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

            tmp_eval_output = Path(tmpdir) / "eval_output"
            tmp_eval_output.mkdir(parents=True, exist_ok=True)
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
                if baseline_for_comparison is None:
                    baseline_for_comparison = RegressionResult_from_dict(baseline_result) if baseline_result else None
                # Pass source_failure_ids from patch so targeted_improvement
                # is calculated on the actual failing questions, not just global metrics
                targeted_ids = list(patch.source_failure_ids) if patch.source_failure_ids else None
                comparison = runner.compare(
                    baseline_for_comparison,
                    after_result,
                    targeted_failure_ids=targeted_ids,
                )
                result["regression_result"] = {
                    "passed": comparison.passed,
                    "score_drop": comparison.score_drop,
                    "reason": comparison.reason,
                    "average_score": comparison.average_score,
                    "score_delta": comparison.score_delta,
                    "targeted_improvement": comparison.targeted_improvement,
                    "failed_cases_increase": comparison.failed_cases_increase,
                }
                # Also surface at top level for log readability
                result["average_score"] = comparison.average_score
                result["score_delta"] = comparison.score_delta
                result["targeted_improvement"] = comparison.targeted_improvement
                result["failed_cases_increase"] = comparison.failed_cases_increase
                if not comparison.passed:
                    result["passed"] = False
                    result["errors"].append(f"Regression failed: {comparison.reason}")
                    patch.status = PatchStatus.REJECTED
                    # Don't return early — fall through to finally for cleanup
        finally:
            if _should_cleanup:
                import shutil as _shutil
                _shutil.rmtree(_tmpdir, ignore_errors=True)
            elif persist_dir is not None and result.get("average_score") is not None:
                # Record summary for later review
                summary_path = persist_dir / "validation_summary.json"
                summary_path.write_text(json.dumps({
                    "patch_id": result.get("patch_id"),
                    "passed": result.get("passed"),
                    "average_score": result.get("average_score"),
                    "score_delta": result.get("score_delta"),
                    "targeted_improvement": result.get("targeted_improvement"),
                    "errors": result.get("errors"),
                }, indent=2, ensure_ascii=False), encoding="utf-8")

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

    # Check for conflicts — auto-resolve by keeping best non-conflicting subset
    conflicts = detect_conflicts(patches)
    if conflicts:
        resolved_patches, conflict_rejected = resolve_conflicts(patches)
        conflict_detail = []
        for id1, id2, path in conflicts:
            conflict_detail.append(
                f"Conflict: patches {id1} and {id2} both target '{path}'"
            )
        result.setdefault("warnings", [])
        result["warnings"].extend(conflict_detail)
        result["conflict_rejected"] = conflict_rejected
        # Only fail if no patches remain after resolution
        if not resolved_patches:
            result["errors"].extend(conflict_detail)
            result["passed"] = False
            return result
        # Re-run with the resolved subset
        result["resolved_patch_ids"] = [p.patch_id for p in resolved_patches]
        result["warnings"].append(
            f"Composition: auto-resolved {len(conflicts)} conflict(s) → "
            f"{len(resolved_patches)}/{len(patches)} patches kept, "
            f"{len(conflict_rejected)} rejected"
        )
        patches = resolved_patches

    # Run baseline subset eval FIRST (before patching) if using sampled questions
    baseline_cmp = RegressionResult_from_dict(baseline_regression_result) if baseline_regression_result else None
    if regression_limit is not None and regression_limit > 0 and regression_dataset and regression_dataset.exists():
        runner = RegressionRunner(project_root)
        tmp_before_output = Path(tempfile.mkdtemp(prefix="comp_baseline_"))
        try:
            before_result = runner.run_eval(
                regression_dataset,
                skills_dir,   # ORIGINAL unpatched skills
                prompts_dir,  # ORIGINAL unpatched prompts
                output_dir=tmp_before_output,
                workers=workers,
                limit=regression_limit,
                seed=regression_seed,
                stream_output=stream_output,
            )
            if not before_result.errors:
                baseline_cmp = before_result
        finally:
            shutil.rmtree(tmp_before_output, ignore_errors=True)

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
            runner = RegressionRunner(project_root)

            tmp_eval_output = Path(tmpdir) / "eval_output"
            tmp_eval_output.mkdir(parents=True, exist_ok=True)
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
            elif baseline_cmp is not None:
                comparison = runner.compare(
                    baseline_cmp,
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
