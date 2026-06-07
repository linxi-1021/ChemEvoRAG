#!/usr/bin/env python
"""No-op Patch Test: verify regression runner stability with a zero-effect patch.

Creates a patch that sets a field to its CURRENT value (no actual change),
then runs it through the full individual regression validation.

If this no-op patch shows score_drop > 0 or failed_cases_increase > 0,
the regression runner / LLM judge is inherently unstable.

Usage:
  python scripts/noop_patch_test.py [--limit N] [--seed S]

Outputs:
  data/evolution/stability/noop_patch.json
  data/evolution/stability/noop_regression_result.json
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

import yaml
from skill_evolution.patch import PatchSchema, PatchOperation, FailureType, PatchStatus
from skill_evolution.runtime_validation import validate_individual_patch

EVAL_DIR = PROJECT_ROOT / "data" / "eval"
SKILLS_DIR = PROJECT_ROOT / "config" / "skills"
PROMPTS_DIR = PROJECT_ROOT / "config" / "prompts"
STABILITY_DIR = PROJECT_ROOT / "data" / "evolution" / "stability"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                        help="Number of questions to sample per regression run.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    STABILITY_DIR.mkdir(parents=True, exist_ok=True)

    # Load a skill config to get current values
    skill_name = "entity_lookup"
    skill_path = SKILLS_DIR / f"{skill_name}.yaml"
    if not skill_path.exists():
        print(f"ERROR: {skill_path} not found")
        return 1

    skill_config = yaml.safe_load(skill_path.read_text("utf-8"))
    current_threshold = (
        skill_config.get("strategy", {})
        .get("query_rewrite", {})
        .get("diversity_threshold", 0.8)
    )

    # Create a no-op patch: set diversity_threshold to its CURRENT value
    noop_patch = PatchSchema(
        patch_id="noop_test_patch",
        skill_name=skill_name,
        skill_version=skill_config.get("skill_version", "1.0.0"),
        source_failure_ids=[],
        primary_failure_type=FailureType.UNKNOWN_FAILURE,
        target_path="strategy.query_rewrite",
        operation=PatchOperation.UPDATE,
        current_value={"diversity_threshold": current_threshold},
        proposed_value={"diversity_threshold": current_threshold},  # SAME value = no-op
        rationale="No-op patch for stability testing. Sets field to its current value.",
        expected_improvement="None — this is a stability test.",
        risk_level="low",
        confidence=1.0,
    )

    # Save the no-op patch
    patch_path = STABILITY_DIR / "noop_patch.json"
    patch_path.write_text(
        json.dumps(noop_patch.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"No-op patch saved to: {patch_path}")
    print(f"  Current value: {current_threshold}")
    print(f"  Proposed value: {current_threshold} (SAME)")

    # Load baseline
    eval_results_path = EVAL_DIR / "eval_results.json"
    baseline_result = None
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

    regression_dataset = EVAL_DIR / "all_questions.json"

    # Run regression validation
    print("\nRunning regression validation on no-op patch...")
    persist_dir = STABILITY_DIR / "noop_eval_output"
    result = validate_individual_patch(
        noop_patch, skill_config,
        project_root=PROJECT_ROOT,
        skills_dir=SKILLS_DIR,
        prompts_dir=PROMPTS_DIR,
        regression_dataset=regression_dataset,
        baseline_result=baseline_result,
        skill_filename=f"{skill_name}.yaml",
        workers=args.workers,
        regression_limit=args.limit,
        regression_seed=args.seed,
        stream_output=True,
        persist_dir=persist_dir,
    )

    # Save result
    result_path = STABILITY_DIR / "noop_regression_result.json"
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    # Print summary
    print(f"\n{'=' * 50}")
    print(f"NO-OP PATCH REGRESSION RESULT")
    print(f"{'=' * 50}")
    print(f"Passed:  {result.get('passed')}")
    print(f"Score:   {result.get('average_score', 'N/A')}")
    print(f"Delta:   {result.get('score_delta', 'N/A')}")
    print(f"Failed+: {result.get('failed_cases_increase', 'N/A')}")
    print(f"Errors:  {result.get('errors', [])}")

    if result.get("passed"):
        print("\nJUDGMENT: STABLE — no-op patch passed regression (expected)")
    else:
        print("\nJUDGMENT: UNSTABLE — no-op patch failed regression!")
        print("  This means the regression runner / LLM judge has inherent noise.")
        print("  Current regression gates cannot reliably distinguish real changes from noise.")

    print(f"\nResult saved to: {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
