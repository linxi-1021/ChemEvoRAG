#!/usr/bin/env python
"""ChemEvoRAG Skill Evolution Runner.

Implements §6 of the Skill Evolution design spec with full auto-writeback.

Modes:
  --analyze-only   Generate trace_report.json only (no patches)
  --dry-run        Generate patches + schema/semantic validation, no regression, no write-back
  --validate-only  Generate patches + schema/semantic + individual + composition regression, no write-back
  --apply          Full pipeline: validate + write-back + post-apply eval + rollback-on-failure

Usage:
  python scripts/evolve.py --analyze-only
  python scripts/evolve.py --dry-run
  python scripts/evolve.py --validate-only
  python scripts/evolve.py --apply
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

_dotenv = PROJECT_ROOT / ".env"
if _dotenv.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_dotenv)
    except ImportError:
        pass

from skill_evolution.attribution import generate_trace_report, TraceReport
from skill_evolution.patch import generate_all_patches, PatchSchema, PatchStatus
from skill_evolution.validation import validate_patch
from skill_evolution.regression import RegressionRunner, RegressionResult, decide_after_apply
from skill_evolution.rollback import SnapshotManager
from skill_evolution.apply import PatchApplier, PatchApplyError
from skill_evolution.runtime_validation import (
    validate_individual_patch,
    validate_composition,
    detect_conflicts,
    RegressionResult_from_dict,
)

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

EVAL_DIR = PROJECT_ROOT / "data" / "eval"
EVOLUTION_DIR = PROJECT_ROOT / "data" / "evolution"
SKILLS_DIR = PROJECT_ROOT / "config" / "skills"
PROMPTS_DIR = PROJECT_ROOT / "config" / "prompts"

_log_lines: list[str] = []


def _log(msg: str) -> None:
    _log_lines.append(msg)
    print(msg)


def _load_skill_configs() -> dict[str, dict]:
    """Load all Skill YAML files."""
    if yaml is None:
        _log("WARNING: pyyaml not installed, skill configs empty")
        return {}
    configs: dict[str, dict] = {}
    if not SKILLS_DIR.is_dir():
        return configs
    for f in SKILLS_DIR.glob("*.yaml"):
        try:
            data = yaml.safe_load(f.read_text("utf-8"))
            intent = data.get("trigger", {}).get("intent", "")
            if intent:
                configs[intent] = data
        except Exception:
            pass
    return configs


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _create_run_dir() -> Path:
    run_id = f"run_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
    run_dir = EVOLUTION_DIR / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _copy_patches_to_global(patches: list[dict], category: str) -> None:
    """Copy patches to data/evolution/patches/{accepted|rejected|rolled_back}/"""
    dest = EVOLUTION_DIR / "patches" / category
    dest.mkdir(parents=True, exist_ok=True)
    for p in patches:
        pid = p.get("patch_id", "unknown")
        _save_json(dest / f"{pid}.json", p)


# ── Phase functions ──────────────────────────────────────────────────

def phase_analyze(eval_results_path: Path, react_logs_dir: Path, run_dir: Path) -> TraceReport:
    skill_configs = _load_skill_configs()
    report = generate_trace_report(eval_results_path, react_logs_dir, skill_configs)
    _save_json(run_dir / "trace_report.json", report.model_dump(mode="json"))
    _log(f"Trace report: {len(report.failures)} failures, "
         f"{len(report.partial_successes)} partial, "
         f"{len(report.success_patterns)} success")
    return report


def phase_generate_patches(
    report: TraceReport,
    run_dir: Path,
    skill_configs: dict[str, dict],
) -> list[PatchSchema]:
    patches = generate_all_patches(report, skill_configs)
    _save_json(run_dir / "candidate_patches.json",
               [p.model_dump(mode="json") for p in patches])
    _log(f"Generated {len(patches)} candidate patches")
    return patches


def phase_schema_validate(
    patches: list[PatchSchema],
    run_dir: Path,
    skill_configs: dict[str, dict],
) -> tuple[list[PatchSchema], list[dict]]:
    """Schema + semantic invariant validation (no regression)."""
    validation_results: list[dict] = []
    accepted: list[PatchSchema] = []
    rejected: list[PatchSchema] = []

    for patch in patches:
        config = skill_configs.get(patch.skill_name, {})
        skill_filename = f"{patch.skill_name}.yaml"
        vresult = validate_patch(patch, config, skill_filename=skill_filename)
        entry = {
            "patch_id": patch.patch_id,
            "valid": vresult.valid,
            "errors": vresult.errors,
            "warnings": vresult.warnings,
        }
        validation_results.append(entry)
        if vresult.valid:
            accepted.append(patch)
        else:
            rejected.append(patch)
            _log(f"  REJECTED {patch.patch_id}: {vresult.errors}")

    _save_json(run_dir / "validation_results.json", validation_results)
    _save_json(run_dir / "selected_patches.json",
               [p.model_dump(mode="json") for p in accepted])
    _save_json(run_dir / "rejected_patches.json",
               [p.model_dump(mode="json") for p in rejected])
    _copy_patches_to_global(
        [p.model_dump(mode="json") for p in rejected], "rejected")

    _log(f"Schema validation: {len(accepted)} accepted, {len(rejected)} rejected")
    return accepted, validation_results


def phase_regression_validate(
    patches: list[PatchSchema],
    run_dir: Path,
    skill_configs: dict[str, dict],
    *,
    workers: int = 1,
    regression_limit: int | None = None,
    regression_seed: int = 42,
    stream_output: bool = True,
) -> list[PatchSchema]:
    """Individual + composition regression validation.

    When regression_limit is not None, questions are randomly sampled using
    regression_seed for reproducibility.
    """
    eval_results_path = EVAL_DIR / "eval_results.json"
    regression_dataset = EVAL_DIR / "all_questions.json"

    if not regression_dataset.exists():
        _log("WARNING: all_questions.json not found, skipping regression")
        return patches

    # Load baseline
    baseline_result: dict | None = None
    if eval_results_path.exists():
        baseline_data = json.loads(eval_results_path.read_text("utf-8"))
        baseline_result = {
            "average_score": baseline_data.get("summary", {}).get("average_score", 0.0),
            "total_questions": len(baseline_data.get("results", [])),
            "by_intent": {
                k: {"average_score": v}
                for k, v in baseline_data.get("summary", {}).get("by_intent", {}).items()
            },
        }

    # Individual validation
    individual_results: list[dict] = []
    still_valid: list[PatchSchema] = []
    rejected_in_regression: list[dict] = []

    for idx, patch in enumerate(patches, 1):
        _log(f"[{idx}/{len(patches)}] Validating {patch.patch_id} "
             f"({patch.skill_name} / {patch.operation.value} / {patch.target_path})")

        config = skill_configs.get(patch.skill_name, {})
        result = validate_individual_patch(
            patch, config,
            project_root=PROJECT_ROOT,
            skills_dir=SKILLS_DIR,
            prompts_dir=PROMPTS_DIR,
            regression_dataset=regression_dataset,
            baseline_result=baseline_result,
            skill_filename=f"{patch.skill_name}.yaml",
            workers=workers,
            regression_limit=regression_limit,
            regression_seed=regression_seed,
            stream_output=stream_output,
        )
        individual_results.append(result)
        if result["passed"]:
            still_valid.append(patch)
            _log(f"  [OK] {patch.patch_id}: passed "
                 f"(score={result.get('average_score', 0):.4f}, "
                 f"delta={result.get('score_delta', 0):+.4f}, "
                 f"targeted_improvement={result.get('targeted_improvement', 0):+.4f})")
        else:
            rejected_entry = patch.model_dump(mode="json")
            rejected_entry["rejection_reason"] = f"individual_regression_failed: {result['errors']}"
            rejected_in_regression.append(rejected_entry)
            _log(f"  REGRESSION FAIL {patch.patch_id}: {result['errors']}")

    _log(f"Individual regression: {len(still_valid)} passed, "
         f"{len(patches) - len(still_valid)} failed")

    # Composition validation (if multiple patches)
    composition_rejected: list[dict] = []
    if len(still_valid) > 1:
        _log(f"Running composition validation for {len(still_valid)} patches...")
        comp_result = validate_composition(
            still_valid, skill_configs,
            project_root=PROJECT_ROOT,
            skills_dir=SKILLS_DIR,
            prompts_dir=PROMPTS_DIR,
            regression_dataset=regression_dataset,
            baseline_regression_result=baseline_result,
            workers=workers,
            regression_limit=regression_limit,
            regression_seed=regression_seed,
            stream_output=stream_output,
        )
        _save_json(run_dir / "composition_validation.json", comp_result)
        if comp_result["passed"]:
            _log(f"  [OK] Composition validation PASSED")
        if not comp_result["passed"]:
            _log(f"  COMPOSITION FAIL: {comp_result['errors']}")
            # Record each rejected patch with composition failure reason
            for p in still_valid:
                rejected_entry = p.model_dump(mode="json")
                rejected_entry["rejection_reason"] = f"composition_validation_failed: {comp_result['errors'][:2]}"
                composition_rejected.append(rejected_entry)
            still_valid = []  # reject all if composition fails
    elif len(still_valid) == 1:
        _log("  Single patch — skipping composition validation")

    # Save all validation results
    all_rejected = rejected_in_regression + composition_rejected
    _save_json(run_dir / "regression_validation.json", individual_results)
    _save_json(run_dir / "selected_patches.json",
               [p.model_dump(mode="json") for p in still_valid])
    _save_json(run_dir / "rejected_patches.json", all_rejected)

    return still_valid


def phase_write_back(
    patches: list[PatchSchema],
    run_dir: Path,
    skill_configs: dict[str, dict],
    snapshot_mgr: SnapshotManager,
    round_id: str,
) -> dict[str, Any]:
    """Write validated patches to disk with snapshot protection."""
    _log(f"\nWriting back {len(patches)} patches...")

    # Snapshot before
    snapshot_mgr.snapshot_before()
    _log("  Snapshot saved (before)")

    applier = PatchApplier(SKILLS_DIR, PROMPTS_DIR)
    applied: list[dict] = []

    for patch in patches:
        config = skill_configs.get(patch.skill_name, {})
        try:
            result_config = applier.apply_to_disk(config, [patch], round_id=round_id)
            applied.append(patch.model_dump(mode="json"))
            _log(f"  APPLIED {patch.patch_id}")
        except (PatchApplyError, Exception) as e:
            _log(f"  FAILED {patch.patch_id}: {e}")
            patch.status = PatchStatus.REJECTED

    # Snapshot after
    snapshot_mgr.snapshot_after()
    _log("  Snapshot saved (after)")

    # Save applied patches
    _save_json(run_dir / "applied_patches.json", applied)
    _copy_patches_to_global(applied, "accepted")

    return {"applied_count": len(applied), "total": len(patches)}


def phase_post_eval(
    run_dir: Path,
    snapshot_mgr: SnapshotManager,
    baseline_eval_path: Path,
    max_score_drop: float,
) -> dict[str, Any]:
    """Run post-apply eval and decide keep/rollback."""
    _log("\nRunning post-apply eval...")
    post_eval_output = run_dir / "post_apply_eval_output"
    post_eval_output.mkdir(parents=True, exist_ok=True)
    runner = RegressionRunner(PROJECT_ROOT)
    after_result = runner.run_eval(
        EVAL_DIR / "all_questions.json",
        SKILLS_DIR,
        PROMPTS_DIR,
        output_dir=post_eval_output,
    )

    if after_result.errors:
        _log(f"  Post-apply eval failed: {after_result.errors}")
        return {"decision": "rollback", "reason": f"Post-apply eval failed: {after_result.errors}"}

    # Compare with baseline
    before_eval = json.loads(baseline_eval_path.read_text("utf-8"))
    after_eval = {
        "summary": {
            "average_score": after_result.average_score,
            "by_intent": {k: v["average_score"] for k, v in after_result.by_intent.items()},
        },
        "results": [],  # not needed for comparison
    }

    decision_result = decide_after_apply(
        before_eval, after_eval,
        max_score_drop=max_score_drop,
    )

    _log(f"  Post-apply score: {after_result.average_score:.4f}")
    _log(f"  Decision: {decision_result['decision']} — {decision_result['reason']}")

    _save_json(run_dir / "post_apply_decision.json", decision_result)

    if decision_result["decision"] == "rollback":
        _log("\nROLLBACK triggered. Restoring before snapshot...")
        snapshot_mgr.rollback()
        _log("  Rollback complete.")
        # Copy patches to rolled_back
        applied_path = run_dir / "applied_patches.json"
        if applied_path.exists():
            rolled = json.loads(applied_path.read_text("utf-8"))
            _save_json(run_dir / "rolled_back_patches.json", rolled)
            _copy_patches_to_global(rolled, "rolled_back")

    return decision_result


# ── Main ─────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rounds", type=int, default=1, help="Number of evolution rounds.")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Only generate trace_report.json (no patches).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate patches + schema validation, no regression, no write-back.")
    parser.add_argument("--validate-only", action="store_true",
                        help="Generate patches + full regression validation, no write-back.")
    parser.add_argument("--apply", action="store_true",
                        help="Full pipeline: validate + write-back + post-apply eval.")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of concurrent workers passed to eval_questions.py during regression validation.")
    parser.add_argument("--quiet-regression", action="store_true",
                        help="Capture regression eval output instead of streaming question-level progress.")
    parser.add_argument("--regression-limit", type=int, default=None,
                        help="Number of eval questions randomly sampled per regression run. "
                             "Useful for quick validate-only smoke tests. "
                             "NOT allowed with --apply.")
    parser.add_argument("--regression-seed", type=int, default=42,
                        help="Random seed used when --regression-limit samples questions (default: 42).")
    parser.add_argument("--skip-regression", action="store_true",
                        help="Skip regression (DEBUG only, incompatible with --apply).")
    parser.add_argument("--eval-results", type=str, default=str(EVAL_DIR / "eval_results.json"))
    parser.add_argument("--react-logs", type=str, default=str(EVAL_DIR / "react_logs"))
    args = parser.parse_args()

    # Safety: --skip-regression + --apply is forbidden
    if args.skip_regression and args.apply:
        _log("ERROR: --skip-regression cannot be combined with --apply.")
        return 1

    # Safety: --regression-limit + --apply is forbidden
    if args.apply and args.regression_limit is not None:
        _log("ERROR: --apply requires full regression; "
             "--regression-limit is only allowed for --validate-only / debug.")
        return 1

    # Validate regression_limit
    if args.regression_limit is not None and args.regression_limit <= 0:
        _log("ERROR: --regression-limit must be > 0.")
        return 1

    eval_results_path = Path(args.eval_results)
    react_logs_dir = Path(args.react_logs)
    baseline_eval_path = Path(args.eval_results)

    if not eval_results_path.exists():
        _log(f"ERROR: eval_results.json not found at {eval_results_path}")
        return 1

    run_dir = _create_run_dir()
    _log(f"Run directory: {run_dir}")

    snapshot_mgr = SnapshotManager(run_dir, SKILLS_DIR, PROMPTS_DIR)
    skill_configs = _load_skill_configs()
    round_id = f"round{args.rounds}"

    try:
        # Phase 1: Analysis
        report = phase_analyze(eval_results_path, react_logs_dir, run_dir)

        if args.analyze_only:
            _log("\n--analyze-only: stopping after analysis.")
            _write_log(run_dir)
            return 0

        # Phase 2: Generate patches
        patches = phase_generate_patches(report, run_dir, skill_configs)

        if not patches:
            _log("No patches generated. Evolution round complete.")
            _write_log(run_dir)
            return 0

        # Phase 3: Schema + semantic validation
        accepted, validation_results = phase_schema_validate(patches, run_dir, skill_configs)

        if args.dry_run:
            _log(f"\n--dry-run: {len(accepted)} patches validated, not applied.")
            _write_log(run_dir)
            return 0

        if not accepted:
            _log("No patches passed schema validation. Stopping.")
            _write_log(run_dir)
            return 0

        # Phase 4: Regression validation
        if args.skip_regression:
            _log("\nWARNING: --skip-regression active, skipping regression.")
            final_patches = accepted
        else:
            stream_output_regression = not args.quiet_regression
            if args.regression_limit is not None:
                _log(f"Regression limit: {args.regression_limit} randomly sampled questions per regression eval")
                _log(f"Regression seed: {args.regression_seed}")
            final_patches = phase_regression_validate(
                accepted, run_dir, skill_configs,
                workers=args.workers,
                regression_limit=args.regression_limit,
                regression_seed=args.regression_seed,
                stream_output=stream_output_regression,
            )

        if args.validate_only:
            _log(f"\n--validate-only: {len(final_patches)} patches validated, not applied.")
            _write_log(run_dir)
            return 0

        if not final_patches:
            _log("No patches passed regression. Stopping.")
            _write_log(run_dir)
            return 0

        if not args.apply:
            _log(f"\n{len(final_patches)} patches ready for review at:")
            _log(f"  {run_dir / 'selected_patches.json'}")
            _write_log(run_dir)
            return 0

        # ─── --apply mode ────────────────────────────────────────────
        _log(f"\n=== APPLY MODE: {len(final_patches)} patches ===")

        # Conflict detection
        conflicts = detect_conflicts(final_patches)
        if conflicts:
            _log(f"  WARNING: {len(conflicts)} conflict(s) detected")
            for id1, id2, path in conflicts:
                _log(f"    {id1} vs {id2} on '{path}'")
            # Remove lower-confidence patch from each conflict pair
            seen = set()
            filtered: list[PatchSchema] = []
            for p in final_patches:
                conflicting_ids = set()
                for id1, id2, _ in conflicts:
                    if p.patch_id == id1:
                        conflicting_ids.add(id2)
                    elif p.patch_id == id2:
                        conflicting_ids.add(id1)
                if p.patch_id not in seen:
                    # Keep the one with higher confidence
                    conflicting_patches = [x for x in final_patches if x.patch_id in conflicting_ids]
                    if all(p.confidence >= x.confidence for x in conflicting_patches):
                        filtered.append(p)
                    seen.add(p.patch_id)
                    for cp in conflicting_patches:
                        seen.add(cp.patch_id)
            final_patches = filtered
            _log(f"  After conflict resolution: {len(final_patches)} patches")

        if not final_patches:
            _log("No patches left after conflict resolution.")
            _write_log(run_dir)
            return 0

        # Write back
        write_result = phase_write_back(
            final_patches, run_dir, skill_configs, snapshot_mgr, round_id)

        if write_result["applied_count"] == 0:
            _log("No patches were applied. Stopping.")
            _write_log(run_dir)
            return 0

        # Post-apply eval
        decision = phase_post_eval(
            run_dir, snapshot_mgr, baseline_eval_path, max_score_drop=0.01)

        _log(f"\n{'='*50}")
        _log(f"FINAL DECISION: {decision['decision'].upper()}")
        _log(f"Reason: {decision['reason']}")

    except Exception as e:
        _log(f"\nFATAL ERROR: {e}")
        _log(traceback.format_exc())
        # Attempt rollback if we had written back
        if snapshot_mgr.has_before_snapshot() and snapshot_mgr.has_after_snapshot():
            _log("Attempting emergency rollback...")
            try:
                snapshot_mgr.rollback()
                _log("  Emergency rollback complete.")
            except Exception as rb_err:
                _log(f"  Emergency rollback failed: {rb_err}")

    _write_log(run_dir)
    return 0


def _write_log(run_dir: Path) -> None:
    log_path = run_dir / "evolution_log.txt"
    log_path.write_text("\n".join(_log_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
