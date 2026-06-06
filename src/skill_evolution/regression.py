"""Regression Runner for ChemEvoRAG Skill Evolution.

Runs eval_questions.py with current skill/prompt configs, compares before/after
results, and determines whether a patch improves or degrades performance.

Usage:
    from skill_evolution.regression import RegressionRunner, RegressionResult
    runner = RegressionRunner(project_root)
    result_before = runner.run_eval(dataset, skills_dir, prompts_dir, output_dir)
    # ... apply patches ...
    result_after = runner.run_eval(dataset, skills_dir, prompts_dir, output_dir)
    comparison = runner.compare(result_before, result_after, targeted_failure_ids=[...])
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RegressionResult:
    """Result of a single eval run."""
    average_score: float = 0.0
    total_questions: int = 0
    by_intent: dict[str, dict[str, float]] = field(default_factory=dict)
    results_path: str | None = None
    errors: list[str] = field(default_factory=list)


@dataclass
class RegressionComparison:
    """Result of comparing before vs after eval runs."""
    passed: bool = True
    average_score: float = 0.0
    score_drop: float = 0.0
    score_delta: float = 0.0
    failed_cases_increase: int = 0
    targeted_improvement: float = 0.0
    new_high_severity_failures: int = 0
    report_path: str | None = None
    errors: list[str] = field(default_factory=list)
    reason: str = ""


class RegressionRunner:
    """Runs eval_questions.py and compares results for regression validation."""

    def __init__(
        self,
        project_root: Path,
        python_executable: str = sys.executable,
    ) -> None:
        self.project_root = Path(project_root)
        self.python = python_executable
        self.eval_script = self.project_root / "scripts" / "eval_questions.py"

    def run_eval(
        self,
        dataset_path: Path,
        skills_dir: Path | None = None,
        prompts_dir: Path | None = None,
        output_dir: Path | None = None,
        *,
        use_react: bool = True,
        workers: int = 1,
        timeout: int = 1800,
    ) -> RegressionResult:
        """Run eval_questions.py and return the results.

        Passes --skills-dir, --prompts-dir, --dataset, --output-dir to the
        eval script so it uses the correct config and writes output where
        the caller expects it.

        output_dir is REQUIRED for regression validation. Calling with
        output_dir=None will raise ValueError to prevent accidental
        overwriting of the baseline eval_results.json.
        """
        # Safety: require explicit output_dir to prevent overwriting baseline
        baseline_eval = self.project_root / "data" / "eval"
        if output_dir is None:
            raise ValueError(
                "output_dir is required for regression eval. "
                "Pass output_dir explicitly to prevent overwriting "
                f"baseline at {baseline_eval}"
            )
        actual_output_dir = Path(output_dir)
        if actual_output_dir.resolve() == baseline_eval.resolve():
            raise ValueError(
                f"Regression eval cannot write to baseline directory: {baseline_eval}. "
                "Use a temporary or run-specific output directory."
            )

        cmd = [self.python, str(self.eval_script)]
        if use_react:
            cmd.append("--react")
        cmd.extend(["--workers", str(workers)])

        # Always pass --skills when skills_dir is provided
        if skills_dir and skills_dir.is_dir():
            cmd.append("--skills")
            cmd.extend(["--skills-dir", str(skills_dir)])

        if prompts_dir and prompts_dir.is_dir():
            cmd.extend(["--prompts-dir", str(prompts_dir)])

        # Always pass --dataset (required by eval script)
        cmd.extend(["--dataset", str(dataset_path)])

        # Always pass --output-dir so results go to the right place
        actual_output_dir.mkdir(parents=True, exist_ok=True)
        cmd.extend(["--output-dir", str(actual_output_dir)])

        import os
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.project_root / "src")

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return RegressionResult(errors=[f"Eval timed out after {timeout}s"])
        except Exception as e:
            return RegressionResult(errors=[f"Eval failed: {e}"])

        if result.returncode != 0:
            return RegressionResult(errors=[
                f"Eval exited with code {result.returncode}",
                result.stderr[:500] if result.stderr else "",
            ])

        # Load eval_results.json from the output directory
        eval_results_path = actual_output_dir / "eval_results.json"
        if not eval_results_path.exists():
            return RegressionResult(errors=["eval_results.json not found after eval run"])

        try:
            data = json.loads(eval_results_path.read_text("utf-8"))
            summary = data.get("summary", {})
            by_intent_raw = summary.get("by_intent", {})

            # Calculate failed cases (score == 0)
            results = data.get("results", [])
            total = len(results)
            avg = summary.get("average_score", 0.0)

            by_intent: dict[str, dict[str, float]] = {}
            for intent, score in by_intent_raw.items():
                intent_results = [r for r in results if r.get("intent") == intent]
                by_intent[intent] = {
                    "average_score": score,
                    "total": len(intent_results),
                    "failed": sum(1 for r in intent_results if r.get("score", 0) < 0.3),
                }

            return RegressionResult(
                average_score=avg,
                total_questions=total,
                by_intent=by_intent,
                results_path=str(eval_results_path),
            )
        except Exception as e:
            return RegressionResult(errors=[f"Failed to parse eval_results.json: {e}"])

    def compare(
        self,
        before: RegressionResult,
        after: RegressionResult,
        *,
        min_score: float = 0.80,
        max_score_drop: float = 0.01,
        targeted_failure_ids: list[str] | None = None,
        min_targeted_improvement: float = 0.05,
        max_failed_cases_increase: int = 1,
    ) -> RegressionComparison:
        """Compare before vs after eval results and determine pass/fail.

        Checks (all must pass):
        1. after.average_score >= min_score
        2. score_drop <= max_score_drop (where score_drop = before - after)
        3. If targeted_failure_ids provided, targeted improvement >= min_targeted_improvement
        4. failed_cases_increase <= max_failed_cases_increase
        """
        if before.errors:
            return RegressionComparison(
                passed=False,
                errors=before.errors,
                reason=f"Before eval had errors: {before.errors}",
            )
        if after.errors:
            return RegressionComparison(
                passed=False,
                errors=after.errors,
                reason=f"After eval had errors: {after.errors}",
            )

        score_drop = before.average_score - after.average_score
        score_delta = after.average_score - before.average_score

        # Count failed cases (score < 0.3)
        before_failed = sum(
            intent_data.get("failed", 0)
            for intent_data in before.by_intent.values()
        )
        after_failed = sum(
            intent_data.get("failed", 0)
            for intent_data in after.by_intent.values()
        )
        failed_cases_increase = after_failed - before_failed

        # Calculate targeted improvement
        targeted_improvement = 0.0
        if targeted_failure_ids and before.results_path:
            try:
                before_data = json.loads(Path(before.results_path).read_text("utf-8"))
                before_results = {r["id"]: r for r in before_data.get("results", [])}
                after_data = json.loads(Path(after.results_path).read_text("utf-8"))
                after_results = {r["id"]: r for r in after_data.get("results", [])}

                improvements = []
                for fid in targeted_failure_ids:
                    b_score = before_results.get(fid, {}).get("score", 0.0)
                    a_score = after_results.get(fid, {}).get("score", 0.0)
                    improvements.append(a_score - b_score)
                targeted_improvement = sum(improvements) / len(improvements) if improvements else 0.0
            except Exception:
                pass

        # Check new high-severity failures (score dropped from >= 0.5 to < 0.3)
        new_high_severity = 0
        if before.results_path and after.results_path:
            try:
                before_data = json.loads(Path(before.results_path).read_text("utf-8"))
                after_data = json.loads(Path(after.results_path).read_text("utf-8"))
                before_map = {r["id"]: r.get("score", 0) for r in before_data.get("results", [])}
                after_map = {r["id"]: r.get("score", 0) for r in after_data.get("results", [])}
                for qid in after_map:
                    if before_map.get(qid, 0) >= 0.5 and after_map[qid] < 0.3:
                        new_high_severity += 1
            except Exception:
                pass

        # Decision logic
        comparison = RegressionComparison(
            average_score=after.average_score,
            score_drop=score_drop,
            score_delta=score_delta,
            failed_cases_increase=failed_cases_increase,
            targeted_improvement=targeted_improvement,
            new_high_severity_failures=new_high_severity,
        )

        errors = []

        # Check 1: minimum score
        if after.average_score < min_score:
            errors.append(f"Average score {after.average_score:.4f} below minimum {min_score}")

        # Check 2: score drop
        if score_drop > max_score_drop:
            errors.append(f"Score dropped by {score_drop:.4f} (max allowed: {max_score_drop})")

        # Check 3: targeted improvement
        if targeted_failure_ids and len(targeted_failure_ids) > 0:
            if targeted_improvement < min_targeted_improvement:
                errors.append(
                    f"Targeted improvement {targeted_improvement:.4f} below minimum {min_targeted_improvement}"
                )

        # Check 4: failed cases increase
        if failed_cases_increase > max_failed_cases_increase:
            errors.append(
                f"Failed cases increased by {failed_cases_increase} (max allowed: {max_failed_cases_increase})"
            )

        # Check 5: new high-severity failures
        if new_high_severity > 0:
            errors.append(f"{new_high_severity} new high-severity failure(s) detected")

        comparison.errors = errors
        comparison.passed = len(errors) == 0
        comparison.reason = "; ".join(errors) if errors else "All regression checks passed"

        return comparison


def decide_after_apply(
    before_full_eval: dict,
    after_full_eval: dict,
    *,
    max_score_drop: float = 0.01,
    max_failed_cases_increase: int = 1,
) -> dict[str, Any]:
    """Decide whether to keep, rollback, or send to manual review after apply.

    Returns a dict with decision, scores, and reason.
    """
    before_score = before_full_eval.get("summary", {}).get("average_score", 0.0)
    after_score = after_full_eval.get("summary", {}).get("average_score", 0.0)
    score_delta = after_score - before_score

    # Count failed cases
    before_results = before_full_eval.get("results", [])
    after_results = after_full_eval.get("results", [])
    before_failed = sum(1 for r in before_results if r.get("score", 0) < 0.3)
    after_failed = sum(1 for r in after_results if r.get("score", 0) < 0.3)
    failed_delta = after_failed - before_failed

    # Check for new high-severity failures
    before_map = {r["id"]: r.get("score", 0) for r in before_results}
    after_map = {r["id"]: r.get("score", 0) for r in after_results}
    new_high_severity = 0
    for qid in after_map:
        if before_map.get(qid, 0) >= 0.5 and after_map[qid] < 0.3:
            new_high_severity += 1

    # Decision logic
    decision = "keep"
    reason = "Score maintained or improved."

    if score_delta < -max_score_drop:
        decision = "rollback"
        reason = f"Score dropped by {-score_delta:.4f} (threshold: {max_score_drop})."
    elif failed_delta > max_failed_cases_increase:
        decision = "rollback"
        reason = f"Failed cases increased by {failed_delta} (threshold: {max_failed_cases_increase})."
    elif new_high_severity > 0:
        decision = "rollback"
        reason = f"{new_high_severity} new high-severity failure(s) detected."
    elif score_delta < 0 and score_delta >= -max_score_drop:
        decision = "manual_review"
        reason = f"Score slightly dropped ({score_delta:.4f}) — within threshold but worth reviewing."

    return {
        "decision": decision,
        "before_average_score": before_score,
        "after_average_score": after_score,
        "score_delta": score_delta,
        "failed_cases_delta": failed_delta,
        "new_high_severity_failures": new_high_severity,
        "reason": reason,
    }
