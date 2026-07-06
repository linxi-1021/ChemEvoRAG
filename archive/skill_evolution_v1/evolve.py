#!/usr/bin/env python
"""ChemEvoRAG Skill Evolution CLI.

Modes:
  --analyze-only   Generate trace_report.json (Stage 1-3, no patches)
  --dry-run        Generate patches + schema/semantic validation + diff output (Stage 1-5)
  --validate-only  Full pipeline including regression validation (Stage 1-6, no writeback)
  --apply          Full pipeline + writeback + post-eval + auto-rollback (Stage 1-7)

Usage:
  python scripts/evolve.py --analyze-only
  python scripts/evolve.py --dry-run
  python scripts/evolve.py --validate-only [--regression-limit 10] [--workers 10]
  python scripts/evolve.py --apply
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Load .env if available
_dotenv = PROJECT_ROOT / ".env"
if _dotenv.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_dotenv)
    except ImportError:
        pass

from skill_evolution.runner import EvolutionRunner, RunMode
from skill_evolution.config import EvolutionConfig, set_config


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Mode
    parser.add_argument("--analyze-only", action="store_true",
                        help="Stage 1-3 only (trace + attribution, no patches)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Stage 1-5 (generate patches + schema validation, no regression, no writeback)")
    parser.add_argument("--validate-only", action="store_true",
                        help="Stage 1-6 (full pipeline including regression, no writeback)")
    parser.add_argument("--apply", action="store_true",
                        help="Stage 1-7 (full pipeline + writeback + post-eval + rollback)")

    # Regression
    parser.add_argument("--workers", type=int, default=1,
                        help="Concurrent workers for eval_questions.py")
    parser.add_argument("--quiet-regression", action="store_true",
                        help="Suppress eval progress output during regression")
    parser.add_argument("--regression-limit", type=int, default=None,
                        help="Max questions per regression eval (None = full 118q)")
    parser.add_argument("--regression-seed", type=int, default=42,
                        help="Random seed for question sampling")
    parser.add_argument("--regression-runs", type=int, default=3,
                        help="Number of parallel regression runs per patch (default: 3)")
    parser.add_argument("--skip-regression", action="store_true",
                        help="Skip regression entirely (debug only, NOT allowed with --apply)")

    # Paths
    parser.add_argument("--skills-dir", type=str,
                        default=str(PROJECT_ROOT / "config" / "skills"),
                        help="Skills directory")
    parser.add_argument("--prompts-dir", type=str,
                        default=str(PROJECT_ROOT / "config" / "prompts"),
                        help="Prompts directory")
    parser.add_argument("--dataset", type=str,
                        default=str(PROJECT_ROOT / "data" / "eval" / "all_questions.json"),
                        help="Eval dataset JSON")
    parser.add_argument("--eval-results", type=str,
                        default=str(PROJECT_ROOT / "data" / "eval" / "eval_results.json"),
                        help="eval_results.json path")
    parser.add_argument("--react-logs", type=str,
                        default=str(PROJECT_ROOT / "data" / "eval" / "react_logs"),
                        help="react_logs directory")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory for run results")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Override run ID")

    # Filtering
    parser.add_argument("--max-patches", type=int, default=None,
                        help="Max patches to process per round")
    parser.add_argument("--target-intent", type=str, default=None,
                        help="Only process failures for a specific intent")

    args = parser.parse_args()

    # ── Safety checks ────────────────────────────────────────────────────
    if args.skip_regression and args.apply:
        print("ERROR: --skip-regression cannot be combined with --apply.", file=sys.stderr)
        return 1

    if args.apply and args.regression_limit is not None:
        print("ERROR: --apply requires full regression; --regression-limit is for validation only.",
              file=sys.stderr)
        return 1

    if args.regression_limit is not None and args.regression_limit <= 0:
        print("ERROR: --regression-limit must be > 0.", file=sys.stderr)
        return 1

    # ── Determine mode ───────────────────────────────────────────────────
    if args.analyze_only:
        mode = RunMode.ANALYZE_ONLY
    elif args.dry_run:
        mode = RunMode.DRY_RUN
    elif args.validate_only:
        mode = RunMode.VALIDATE_ONLY
    elif args.apply:
        mode = RunMode.APPLY
    else:
        # Default: dry-run
        mode = RunMode.DRY_RUN
        print("No mode specified — defaulting to --dry-run", file=sys.stderr)

    # ── Configure ────────────────────────────────────────────────────────
    config = EvolutionConfig(
        regression_runs=args.regression_runs,
        regression_limit=args.regression_limit,
        regression_seed=args.regression_seed,
        workers=args.workers,
        max_patches_per_round=args.max_patches or 2,
        stream_output=not args.quiet_regression,
    )
    set_config(config)

    # ── Run ──────────────────────────────────────────────────────────────
    runner = EvolutionRunner(PROJECT_ROOT, config)

    # Override paths from CLI
    runner.skills_dir = Path(args.skills_dir)
    runner.prompts_dir = Path(args.prompts_dir)
    runner.dataset_path = Path(args.dataset)
    runner.eval_results_path = Path(args.eval_results)
    runner.react_logs_dir = Path(args.react_logs)

    result = runner.run(
        mode,
        run_id=args.run_id,
        output_dir=args.output_dir,
        workers=args.workers,
        regression_limit=args.regression_limit,
        regression_seed=args.regression_seed,
        skip_regression=args.skip_regression,
        target_intent=args.target_intent,
        quiet_regression=args.quiet_regression,
    )

    print(f"\nStatus: {result.get('status', 'unknown')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
