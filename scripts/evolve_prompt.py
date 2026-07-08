#!/usr/bin/env python
"""ChemEvoRAG Prompt Evolution CLI — Reflexion-style self-improving prompts.

Reads eval_results.json, generates failure reflections (saved to memory),
bootstraps few-shot examples from successes, and produces improved prompt
candidates via LLM.

Modes:
  --analyze-only     Read eval results, print evolvable skills (no mutations)
  --evolve           Full evolution: reflect + bootstrap + generate candidates
  --validate-only    Generate candidates + run targeted regression (no writeback)
  --apply            Full cycle: evolve + regression + writeback to config/

Usage:
  python scripts/evolve_prompt.py --analyze-only
  python scripts/evolve_prompt.py --evolve --target-intent reaction_comparison
  python scripts/evolve_prompt.py --validate-only --target-intent alias_resolution
  python scripts/evolve_prompt.py --apply --target-intent reaction_condition_query
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Load .env
_dotenv = PROJECT_ROOT / ".env"
if _dotenv.exists():
    from dotenv import load_dotenv
    load_dotenv(_dotenv)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="ChemEvoRAG Prompt Evolution — Reflexion-style self-improving prompts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/evolve_prompt.py --analyze-only
  python scripts/evolve_prompt.py --evolve --target-intent reaction_comparison
  python scripts/evolve_prompt.py --validate-only --target-intent alias_resolution
  python scripts/evolve_prompt.py --apply --target-intent reaction_condition_query

DEPRECATED: scripts/evolve.py has been archived to archive/skill_evolution_v1/.
Use evolve_prompt.py for the new Reflexion-style pipeline.
""",
    )
    parser.add_argument(
        "--analyze-only", action="store_true",
        help="Read eval results and print evolvable skills (no mutations).",
    )
    parser.add_argument(
        "--evolve", action="store_true",
        help="Full evolution: reflect + bootstrap + generate prompt candidates.",
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Generate candidates + run targeted regression (no writeback).",
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="Full cycle: evolve + regression + writeback to config/.",
    )
    parser.add_argument(
        "--target-intent", type=str, default=None,
        help="Only evolve prompts for a specific intent (e.g. reaction_comparison).",
    )
    parser.add_argument(
        "--round", type=int, default=0,
        help="Evolution round number (for memory tracking).",
    )
    parser.add_argument(
        "--eval-results", type=Path,
        default=PROJECT_ROOT / "data" / "eval" / "eval_results.json",
        help="Path to eval_results.json.",
    )
    parser.add_argument(
        "--skills-dir", type=Path,
        default=PROJECT_ROOT / "config" / "skills",
    )
    parser.add_argument(
        "--prompts-dir", type=Path,
        default=PROJECT_ROOT / "config" / "prompts",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "data" / "evolution" / "runs",
    )
    parser.add_argument(
        "--regression-limit", type=int, default=10,
        help="Max questions for targeted regression (default: 10).",
    )
    parser.add_argument(
        "--regression-workers", type=int, default=1,
        help="Workers for regression eval (default: 1).",
    )
    args = parser.parse_args()

    # Determine mode
    modes = []
    if args.analyze_only:
        modes.append("analyze")
    if args.evolve:
        modes.append("evolve")
    if args.validate_only:
        modes.append("validate")
    if args.apply:
        modes.append("apply")

    if len(modes) == 0:
        parser.print_help()
        print("\nERROR: Must specify one of --analyze-only, --evolve, --validate-only, --apply")
        return 1
    if len(modes) > 1:
        print(f"ERROR: Only one mode allowed, got: {modes}")
        return 1

    mode = modes[0]

    # Verify eval results exist
    if not args.eval_results.exists():
        print(f"ERROR: eval_results.json not found at {args.eval_results}")
        print("Run: python scripts/eval_questions.py --react --skills")
        return 1

    # Run
    from skill_evolution.simple_runner import SimpleEvolutionRunner

    runner = SimpleEvolutionRunner(
        project_root=PROJECT_ROOT,
        eval_results_path=args.eval_results,
        skills_dir=args.skills_dir,
        prompts_dir=args.prompts_dir,
    )

    try:
        if mode == "analyze":
            report = runner.analyze(
                target_intent=args.target_intent,
                round_id=args.round,
            )
            print(f"\nReport saved to: {runner.run_dir / 'analysis_report.json'}")

        elif mode == "evolve":
            result = runner.evolve(
                target_intent=args.target_intent,
                round_id=args.round,
            )
            patches = result.get("candidate_patches", [])
            print(f"\n{'='*60}")
            print(f"Evolution complete: {len(patches)} candidate patches generated")
            print(f"Reflections: {len(result.get('reflections', []))}")
            print(f"Run directory: {runner.run_dir}")

        elif mode == "validate":
            result = runner.evolve(
                target_intent=args.target_intent,
                round_id=args.round,
            )
            patches = result.get("candidate_patches", [])
            if not patches:
                print("No patches to validate.")
                return 0

            # Schema validate
            from skill_evolution.validation import validate_patch
            skill_configs = _load_skill_configs(args.skills_dir)

            valid_patches = []
            for p in patches:
                config = skill_configs.get(p.skill_name, {})
                vresult = validate_patch(
                    p, config,
                    skill_filename=f"{p.skill_name}.yaml",
                    prompts_dir=args.prompts_dir,
                    all_patches=list(patches),
                )
                if vresult.valid:
                    valid_patches.append(p)
                    print(f"  [SCHEMA OK] {p.patch_id}")
                else:
                    print(f"  [SCHEMA FAIL] {p.patch_id}: {'; '.join(vresult.errors)}")

            if not valid_patches:
                print("No patches passed schema validation.")
                return 0

            # Targeted regression — scope to target skill only
            from skill_evolution.runtime_validation import validate_individual_patch
            from skill_evolution.regression import RegressionRunner

            print(f"\nRunning targeted regression on {len(valid_patches)} patch(es)...")

            # Build targeted dataset: target skill failures + same-skill successes
            all_qs = json.loads(
                (args.eval_results.parent / "all_questions.json").read_text("utf-8")
            )
            dataset_by_intent = json.loads(args.eval_results.read_text("utf-8"))
            target_intent_list = list(set(
                p.skill_name for p in valid_patches
            ))

            # For each target intent: pick failures (score < 0.8) + successes (score >= 0.9)
            targeted_ids: set[str] = set()
            n_failures = 0
            n_successes = 0
            for intent in target_intent_list:
                intent_results = [
                    r for r in dataset_by_intent.get("results", [])
                    if r.get("intent") == intent
                ]
                # Failures
                for r in intent_results:
                    score = r.get("score", r.get("judge_score", 0))
                    if score < 0.8:
                        targeted_ids.add(r.get("id", ""))
                        n_failures += 1
                # Successes (holdout)
                for r in intent_results:
                    score = r.get("score", r.get("judge_score", 0))
                    if score >= 0.9 and r.get("id") not in targeted_ids:
                        targeted_ids.add(r.get("id", ""))
                        n_successes += 1

            # Filter all_questions to targeted IDs
            targeted_qs = [q for q in all_qs if q.get("id") in targeted_ids]

            if not targeted_qs:
                print("  WARNING: No targeted questions found, falling back to random sample")
                targeted_qs = all_qs[:args.regression_limit]

            targeted_dataset = args.eval_results.parent / "_targeted_questions.json"
            targeted_dataset.write_text(
                json.dumps(targeted_qs, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"  Targeted set: {len(targeted_qs)} questions "
                  f"({n_failures} target failures, {n_successes} same-skill successes)")

            regression_passed = []
            for p in valid_patches:
                print(f"  Validating: {p.patch_id}")
                config = skill_configs.get(p.skill_name, {})
                persist_dir = runner.run_dir / "per_patch_evals" / p.patch_id

                result = validate_individual_patch(
                    p, config,
                    project_root=PROJECT_ROOT,
                    skills_dir=args.skills_dir,
                    prompts_dir=args.prompts_dir,
                    regression_dataset=targeted_dataset,
                    baseline_result={
                        "average_score": 0.8661,
                    },
                    skill_filename=f"{p.skill_name}.yaml",
                    workers=args.regression_workers,
                    regression_limit=None,  # None = run all targeted questions (no sampling)
                    regression_seed=42,
                    stream_output=True,
                    persist_dir=persist_dir,
                    original_eval_path=args.eval_results,
                    all_patches=list(valid_patches),
                )
                if result.get("passed"):
                    regression_passed.append(p)
                    print(f"    [REGRESSION OK] score={result.get('average_score', 0):.4f}")
                else:
                    errors = result.get("errors", [])
                    print(f"    [REGRESSION FAIL] {errors[:2]}")

            print(f"\nValidation summary: {len(regression_passed)}/{len(valid_patches)} passed regression")

        elif mode == "apply":
            # Full cycle with regression gate
            result = runner.evolve(
                target_intent=args.target_intent,
                round_id=args.round,
            )
            patches = result.get("candidate_patches", [])
            if not patches:
                print("No patches to apply.")
                return 0

            # Schema validation
            from skill_evolution.validation import validate_patch
            skill_configs = _load_skill_configs(args.skills_dir)

            valid_patches = []
            for p in patches:
                config = skill_configs.get(p.skill_name, {})
                vresult = validate_patch(
                    p, config,
                    skill_filename=f"{p.skill_name}.yaml",
                    prompts_dir=args.prompts_dir,
                    all_patches=list(patches),
                )
                if vresult.valid:
                    valid_patches.append(p)

            if not valid_patches:
                print("No patches passed schema validation. Aborting.")
                return 1

            # Regression gate (minimal — targeted eval on 10 questions)
            from skill_evolution.runtime_validation import validate_individual_patch
            regression_passed = []
            for p in valid_patches:
                config = skill_configs.get(p.skill_name, {})
                persist_dir = runner.run_dir / "per_patch_evals" / p.patch_id
                vr = validate_individual_patch(
                    p, config,
                    project_root=PROJECT_ROOT,
                    skills_dir=args.skills_dir,
                    prompts_dir=args.prompts_dir,
                    regression_dataset=args.eval_results.parent / "all_questions.json",
                    baseline_result={"average_score": 0.8661},
                    skill_filename=f"{p.skill_name}.yaml",
                    workers=1,
                    regression_limit=10,
                    regression_seed=42,
                    stream_output=True,
                    persist_dir=persist_dir,
                    original_eval_path=args.eval_results,
                    all_patches=list(valid_patches),
                )
                if vr.get("passed"):
                    regression_passed.append(p)

            if not regression_passed:
                print("No patches passed regression. Aborting apply.")
                return 1

            # Apply
            apply_result = runner.apply(regression_passed)
            print(f"\nApply complete: {len(apply_result['applied'])} patches applied")
            if apply_result["failed"]:
                print(f"Failures: {apply_result['failed']}")

    except Exception as e:
        print(f"\nFATAL: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


def _load_skill_configs(skills_dir: Path) -> dict[str, dict]:
    configs: dict[str, dict] = {}
    if not skills_dir.is_dir():
        return configs
    import yaml
    for f in skills_dir.glob("*.yaml"):
        try:
            data = yaml.safe_load(f.read_text("utf-8"))
            intent = data.get("trigger", {}).get("intent", "")
            if intent:
                configs[intent] = data
        except Exception:
            pass
    return configs


if __name__ == "__main__":
    raise SystemExit(main())
