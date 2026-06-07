#!/usr/bin/env python
"""Baseline Stability Test: run eval twice without patches, compare results.

Usage:
  python scripts/baseline_stability_test.py [--limit N] [--seed S]

Outputs:
  data/evolution/stability/baseline_run_1.json
  data/evolution/stability/baseline_run_2.json
  data/evolution/stability/baseline_stability_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from skill_evolution.regression import RegressionRunner, RegressionResult

EVAL_DIR = PROJECT_ROOT / "data" / "eval"
STABILITY_DIR = PROJECT_ROOT / "data" / "evolution" / "stability"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                        help="Number of questions to sample per run.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling.")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    STABILITY_DIR.mkdir(parents=True, exist_ok=True)
    runner = RegressionRunner(PROJECT_ROOT)
    dataset = EVAL_DIR / "all_questions.json"

    if not dataset.exists():
        print("ERROR: all_questions.json not found")
        return 1

    # Run 1
    print("=" * 50)
    print("BASELINE RUN 1")
    print("=" * 50)
    output_1 = STABILITY_DIR / "run_1_output"
    result_1 = runner.run_eval(
        dataset, None, None,
        output_dir=output_1,
        workers=args.workers,
        limit=args.limit,
        seed=args.seed,
    )
    if result_1.errors:
        print(f"Run 1 errors: {result_1.errors}")
        return 1

    # Run 2
    print("=" * 50)
    print("BASELINE RUN 2")
    print("=" * 50)
    output_2 = STABILITY_DIR / "run_2_output"
    result_2 = runner.run_eval(
        dataset, None, None,
        output_dir=output_2,
        workers=args.workers,
        limit=args.limit,
        seed=args.seed,
    )
    if result_2.errors:
        print(f"Run 2 errors: {result_2.errors}")
        return 1

    # Compare
    comparison = runner.compare(result_1, result_2)

    # Load detailed results
    r1_data = json.loads((output_1 / "eval_results.json").read_text("utf-8"))
    r2_data = json.loads((output_2 / "eval_results.json").read_text("utf-8"))

    # Per-question comparison
    r1_map = {r["id"]: r for r in r1_data.get("results", [])}
    r2_map = {r["id"]: r for r in r2_data.get("results", [])}

    per_question = {}
    for qid in set(list(r1_map.keys()) + list(r2_map.keys())):
        s1 = r1_map.get(qid, {}).get("score", None)
        s2 = r2_map.get(qid, {}).get("score", None)
        if s1 is not None and s2 is not None:
            per_question[qid] = {
                "run_1_score": s1,
                "run_2_score": s2,
                "delta": s2 - s1,
                "changed": abs(s2 - s1) > 0.01,
            }

    # Per-intent comparison
    per_intent = {}
    for intent in set(list(result_1.by_intent.keys()) + list(result_2.by_intent.keys())):
        i1 = result_1.by_intent.get(intent, {})
        i2 = result_2.by_intent.get(intent, {})
        per_intent[intent] = {
            "run_1_avg": i1.get("average_score", 0),
            "run_2_avg": i2.get("average_score", 0),
            "delta": i2.get("average_score", 0) - i1.get("average_score", 0),
        }

    report = {
        "generated_at": datetime.now().isoformat(),
        "run_1": {
            "average_score": result_1.average_score,
            "total_questions": result_1.total_questions,
        },
        "run_2": {
            "average_score": result_2.average_score,
            "total_questions": result_2.total_questions,
        },
        "comparison": {
            "score_delta": comparison.score_delta,
            "score_drop": comparison.score_drop,
            "failed_cases_increase": comparison.failed_cases_increase,
            "passed": comparison.passed,
            "reason": comparison.reason,
        },
        "per_intent": per_intent,
        "questions_with_changed_score": {
            k: v for k, v in per_question.items() if v["changed"]
        },
        "total_changed": sum(1 for v in per_question.values() if v["changed"]),
        "judgment": "STABLE" if abs(comparison.score_delta) < 0.01 and comparison.failed_cases_increase <= 1
        else "UNSTABLE",
    }

    # Save
    (STABILITY_DIR / "baseline_run_1.json").write_text(
        json.dumps({"average_score": result_1.average_score, "by_intent": result_1.by_intent}, indent=2),
        encoding="utf-8",
    )
    (STABILITY_DIR / "baseline_run_2.json").write_text(
        json.dumps({"average_score": result_2.average_score, "by_intent": result_2.by_intent}, indent=2),
        encoding="utf-8",
    )
    (STABILITY_DIR / "baseline_stability_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    print(f"\n{'=' * 50}")
    print(f"STABILITY REPORT")
    print(f"{'=' * 50}")
    print(f"Run 1 avg: {result_1.average_score:.4f}")
    print(f"Run 2 avg: {result_2.average_score:.4f}")
    print(f"Delta:     {comparison.score_delta:+.4f}")
    print(f"Changed:   {report['total_changed']} questions")
    print(f"Judgment:  {report['judgment']}")
    print(f"\nReport saved to: {STABILITY_DIR / 'baseline_stability_report.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
