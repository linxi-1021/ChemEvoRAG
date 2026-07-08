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
import os
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
    targeted_improvement_origin: float = 0.0  # vs original eval score
    baseline_flip: bool = False  # original was failure, regression baseline became success
    collateral_improvement: float = 0.0  # improvement on non-targeted failures
    new_high_severity_failures: int = 0
    report_path: str | None = None
    errors: list[str] = field(default_factory=list)
    reason: str = ""


class RegressionRunner:
    """Runs eval_questions.py and compares results for regression validation."""

    def __init__(
        self,
        project_root: Path,
    ) -> None:
        self.project_root = Path(project_root)

    def run_eval(
        self,
        dataset_path: Path,
        skills_dir: Path | None = None,
        prompts_dir: Path | None = None,
        output_dir: Path | None = None,
        *,
        use_react: bool = True,
        workers: int = 1,
        limit: int | None = None,
        seed: int = 42,
        stream_output: bool = True,
    ) -> RegressionResult:
        """Run eval_questions.py and return the results.

        Calls eval_questions.run_evaluation() directly (no subprocess) so the
        eval runs in the same Python process with the same installed packages.

        output_dir is REQUIRED for regression validation. Calling with
        output_dir=None will raise ValueError to prevent accidental
        overwriting of the baseline eval_results.json.

        When limit is not None, random sampling is used with the given seed
        (default 42) for reproducibility. When limit is None, the full
        dataset is evaluated.

        When stream_output=True (default), eval progress (tqdm bars, per-question
        results) prints to stdout in real time. Set stream_output=False or pass
        --quiet-regression to capture silently.
        """
        # Validate workers
        if workers < 1:
            raise ValueError(f"workers must be >= 1, got {workers}")

        # Validate limit
        if limit is not None and limit <= 0:
            raise ValueError(f"limit must be > 0, got {limit}")

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

        actual_output_dir.mkdir(parents=True, exist_ok=True)

        # Reload .env so API_KEY is available in the sub-process / thread context.
        # eval_questions.py loads dotenv at import time, but router/judge/solver
        # cache env state at their own import time. Explicit reload here ensures
        # the sandbox eval process can see API_KEY.
        _dotenv = self.project_root / ".env"
        if _dotenv.exists():
            from dotenv import load_dotenv as _load_dotenv
            _load_dotenv(_dotenv)

        # Ensure scripts/ is importable so we can call eval_questions directly
        scripts_dir = str(self.project_root / "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)

        import eval_questions as _eval_mod

        # Thread-safe GBK fix: use env var instead of modifying shared sys.stdout
        _old_encoding = os.environ.get("PYTHONIOENCODING", "")
        if sys.platform == "win32":
            os.environ["PYTHONIOENCODING"] = "utf-8"

        try:
            result = self._run_eval_inner(
                _eval_mod, dataset_path, actual_output_dir,
                limit, seed, use_react, workers,
                skills_dir, prompts_dir, stream_output,
            )
        finally:
            # Restore original encoding
            if _old_encoding:
                os.environ["PYTHONIOENCODING"] = _old_encoding
            elif "PYTHONIOENCODING" in os.environ:
                del os.environ["PYTHONIOENCODING"]

        return result

    def _run_eval_inner(
        self,
        _eval_mod,
        dataset_path: Path,
        actual_output_dir: Path,
        limit, seed, use_react, workers,
        skills_dir, prompts_dir, stream_output,
    ) -> RegressionResult:
        """Inner eval runner (separated for stream cleanup)."""

        if not stream_output:
            # --- Quiet mode: suppress ReAct logs but keep tqdm progress bar ---
            import io
            captured = io.StringIO()
            import contextlib
            try:
                with contextlib.redirect_stdout(captured):
                    _rc = _eval_mod.run_evaluation(
                        dataset=dataset_path,
                        output_dir=actual_output_dir,
                        limit=limit,
                        seed=seed,
                        use_react=use_react,
                        workers=workers,
                        use_skills=bool(skills_dir and skills_dir.is_dir()),
                        skills_dir=skills_dir,
                        prompts_dir=prompts_dir,
                        quiet=True,
                    )
            except Exception as e:
                captured.write(f"\n[ERROR] eval_questions raised: {e}\n")
                log_path = actual_output_dir / "eval_stdout.log"
                log_path.write_text(captured.getvalue(), encoding="utf-8", errors="replace")
                return RegressionResult(errors=[f"Eval failed: {e}"])
            # Save captured stdout for debugging
            log_path = actual_output_dir / "eval_stdout.log"
            log_path.write_text(captured.getvalue(), encoding="utf-8", errors="replace")
        else:
            # --- Streaming mode: eval output goes directly to stdout ---
            try:
                _eval_mod.run_evaluation(
                    dataset=dataset_path,
                    output_dir=actual_output_dir,
                    limit=limit,
                    seed=seed,
                    use_react=use_react,
                    workers=workers,
                    use_skills=bool(skills_dir and skills_dir.is_dir()),
                    skills_dir=skills_dir,
                    prompts_dir=prompts_dir,
                )
            except Exception as e:
                return RegressionResult(errors=[f"Eval failed: {e}"])

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
        original_eval_path: str | Path | None = None,
    ) -> RegressionComparison:
        """Compare before vs after eval results and determine pass/fail.

        Checks (all must pass):
        1. after.average_score >= min_score
        2. score_drop <= max_score_drop (where score_drop = before - after)
        3. If targeted_failure_ids provided, targeted_improvement >= min_targeted_improvement
        4. failed_cases_increase <= max_failed_cases_increase

        When original_eval_path is provided, targeted_improvement is calculated
        against the original eval scores (not regression baseline), so that
        LLM non-determinism doesn't "wash out" the improvement signal.
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
        # Use original eval scores as baseline (not regression baseline)
        # to avoid LLM non-determinism "washing out" the improvement signal
        targeted_improvement = 0.0
        targeted_improvement_origin = 0.0
        baseline_flip = False
        collateral_improvement = 0.0

        # Load original eval results if available
        original_results: dict = {}
        if original_eval_path:
            try:
                orig_path = Path(original_eval_path)
                if orig_path.exists():
                    orig_data = json.loads(orig_path.read_text("utf-8"))
                    original_results = {f"{r['id']}_{r.get('source_paper', '')}": r
                                        for r in orig_data.get("results", [])}
            except Exception:
                pass

        # Load before/after regression results
        before_results: dict = {}
        after_results: dict = {}
        try:
            if before.results_path:
                before_data = json.loads(Path(before.results_path).read_text("utf-8"))
                before_results = {f"{r['id']}_{r.get('source_paper', '')}": r
                                  for r in before_data.get("results", [])}
            if after.results_path:
                after_data = json.loads(Path(after.results_path).read_text("utf-8"))
                after_results = {f"{r['id']}_{r.get('source_paper', '')}": r
                                 for r in after_data.get("results", [])}
        except Exception:
            pass

        if targeted_failure_ids and after_results:
            improvements_origin = []
            improvements_replay = []
            flip_detected = False

            for fid in targeted_failure_ids:
                # Match by composite key or plain ID
                a_scores = [r.get("score", 0.0) for k, r in after_results.items()
                            if k == fid or k.startswith(f"{fid}_") or r.get("id") == fid]
                a_avg = sum(a_scores) / len(a_scores) if a_scores else 0.0

                # Calculate vs original eval (primary metric)
                if original_results:
                    o_scores = [r.get("score", 0.0) for k, r in original_results.items()
                                if k == fid or k.startswith(f"{fid}_") or r.get("id") == fid]
                    o_avg = sum(o_scores) / len(o_scores) if o_scores else 0.0
                    improvements_origin.append(a_avg - o_avg)

                    # Check baseline flip: original was failure, regression baseline became success
                    if before_results:
                        b_scores = [r.get("score", 0.0) for k, r in before_results.items()
                                    if k == fid or k.startswith(f"{fid}_") or r.get("id") == fid]
                        b_avg = sum(b_scores) / len(b_scores) if b_scores else 0.0
                        if o_avg < 0.5 and b_avg >= 0.8:
                            flip_detected = True

                # Calculate vs regression baseline (diagnostic metric)
                if before_results:
                    b_scores = [r.get("score", 0.0) for k, r in before_results.items()
                                if k == fid or k.startswith(f"{fid}_") or r.get("id") == fid]
                    b_avg = sum(b_scores) / len(b_scores) if b_scores else 0.0
                    improvements_replay.append(a_avg - b_avg)

            targeted_improvement_origin = sum(improvements_origin) / len(improvements_origin) if improvements_origin else 0.0
            targeted_improvement = sum(improvements_replay) / len(improvements_replay) if improvements_replay else 0.0
            baseline_flip = flip_detected

        # Calculate collateral improvement (improvements on non-targeted failures)
        if original_results and after_results and targeted_failure_ids:
            targeted_set = set(targeted_failure_ids)
            collateral_deltas = []
            for k, orig_r in original_results.items():
                qid = orig_r.get("id", "")
                if qid in targeted_set:
                    continue
                orig_score = orig_r.get("score", 0.0)
                if orig_score >= 0.5:
                    continue  # Not a failure
                after_r = after_results.get(k)
                if after_r:
                    after_score = after_r.get("score", 0.0)
                    collateral_deltas.append(after_score - orig_score)
            if collateral_deltas:
                collateral_improvement = sum(collateral_deltas) / len(collateral_deltas)

        # Check new high-severity failures (score dropped from >= 0.5 to < 0.3)
        new_high_severity = 0
        if before.results_path and after.results_path:
            try:
                before_data = json.loads(Path(before.results_path).read_text("utf-8"))
                after_data = json.loads(Path(after.results_path).read_text("utf-8"))
                # Use composite key to avoid false matches across papers
                before_map = {f"{r['id']}_{r.get('source_paper', '')}": r.get("score", 0)
                              for r in before_data.get("results", [])}
                after_map = {f"{r['id']}_{r.get('source_paper', '')}": r.get("score", 0)
                             for r in after_data.get("results", [])}
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
            targeted_improvement_origin=targeted_improvement_origin,
            baseline_flip=baseline_flip,
            collateral_improvement=collateral_improvement,
            new_high_severity_failures=new_high_severity,
        )

        errors = []

        # Check 1: minimum score
        if after.average_score < min_score:
            errors.append(f"Average score {after.average_score:.4f} below minimum {min_score}")

        # Check 2: score drop (with targeted improvement adjustment)
        # Use targeted_improvement_origin (vs original eval) for decision
        effective_max_drop = max_score_drop
        if targeted_improvement_origin > 0:
            effective_max_drop = max_score_drop + min(targeted_improvement_origin * 0.5, max_score_drop)
        if score_drop > effective_max_drop:
            errors.append(f"Score dropped by {score_drop:.4f} (max allowed: {effective_max_drop:.4f})")

        # Check 3: targeted improvement (use original eval as baseline)
        if targeted_failure_ids and len(targeted_failure_ids) > 0:
            if original_results and after.results_path:
                if targeted_improvement_origin < min_targeted_improvement:
                    errors.append(
                        f"Targeted improvement {targeted_improvement_origin:.4f} below minimum {min_targeted_improvement}"
                    )
            # If no original_results, fall back to regression baseline comparison
            elif before.results_path and after.results_path:
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

    # Check for new high-severity failures (use composite key for duplicate IDs)
    def _rkey(r):
        return f"{r['id']}_{r.get('source_paper', '')}"
    before_map = {_rkey(r): r.get("score", 0) for r in before_results}
    after_map = {_rkey(r): r.get("score", 0) for r in after_results}
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
