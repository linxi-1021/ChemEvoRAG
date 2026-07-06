"""Minimal closed-loop Skill Evolution runner.

Reflexion-style evolution loop:
  1. Load eval_results.json
  2. Split into successes (score >= 0.9) and failures (score < 0.8)
  3. Generate reflections → save to memory
  4. Bootstrap few-shot from successes
  5. Generate prompt candidates via LLM
  6. (Optional) Validate candidates via regression
  7. Output selected patches

Usage:
    from skill_evolution.simple_runner import SimpleEvolutionRunner

    runner = SimpleEvolutionRunner(project_root=Path("."))
    report = runner.analyze()         # Stage 1: analyze only
    patches = runner.evolve()         # Stage 1+2: analyze + generate patches
    runner.apply(patches)             # Stage 3: apply patches
"""

from __future__ import annotations

import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from .memory import generate_reflections, append_reflections, load_memory, format_memory_for_prompt
from .bootstrap import bootstrap_examples, bootstrap_summary
from .prompt_evolver import generate_prompt_candidates, build_prompt_update_patches
from .patch import PatchSchema


class SimpleEvolutionRunner:
    """Minimal Reflexion-style evolution runner.

    Focused exclusively on prompt evolution (Phase 1 scope).
    Does NOT modify retrieval_routing, query_rewrite, or other params.
    """

    def __init__(
        self,
        project_root: Path | None = None,
        eval_results_path: Path | None = None,
        skills_dir: Path | None = None,
        prompts_dir: Path | None = None,
    ) -> None:
        self.project_root = project_root or PROJECT_ROOT
        self.eval_results_path = (
            eval_results_path or self.project_root / "data" / "eval" / "eval_results.json"
        )
        self.skills_dir = skills_dir or self.project_root / "config" / "skills"
        self.prompts_dir = prompts_dir or self.project_root / "config" / "prompts"
        self.run_dir: Path | None = None

    # ── Stage 1: Analyze ─────────────────────────────────────────────────

    def analyze(
        self,
        target_intent: str | None = None,
        round_id: int = 0,
    ) -> dict[str, Any]:
        """Analyze evaluation results and produce an evolution report.

        Does NOT modify any files — read-only analysis.
        """
        if not self.eval_results_path.exists():
            raise FileNotFoundError(f"eval_results.json not found: {self.eval_results_path}")

        # Create run directory
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.run_dir = self.project_root / "data" / "evolution" / "runs" / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        data = json.loads(self.eval_results_path.read_text("utf-8"))
        results = data.get("results", [])
        summary = data.get("summary", {})

        # Filter by intent if specified
        if target_intent:
            results = [r for r in results if r.get("intent") == target_intent]

        # Split into successes and failures
        successes = [r for r in results if r.get("score", r.get("judge_score", 0)) >= 0.9]
        failures = [r for r in results if r.get("score", r.get("judge_score", 0)) < 0.8]
        partials = [
            r for r in results
            if 0.8 <= r.get("score", r.get("judge_score", 0)) < 0.9
        ]

        # Group by intent
        by_intent: dict[str, dict[str, list]] = {}
        intents = set(r.get("intent", "unknown") for r in results)
        for intent in sorted(intents):
            intent_results = [r for r in results if r.get("intent") == intent]
            intent_successes = [r for r in successes if r.get("intent") == intent]
            intent_failures = [r for r in failures if r.get("intent") == intent]
            intent_partials = [r for r in partials if r.get("intent") == intent]
            by_intent[intent] = {
                "total": len(intent_results),
                "successes": len(intent_successes),
                "failures": len(intent_failures),
                "partials": len(intent_partials),
                "avg_score": (
                    sum(r.get("score", r.get("judge_score", 0)) for r in intent_results)
                    / len(intent_results)
                ) if intent_results else 0.0,
            }

        report = {
            "run_id": run_id,
            "generated_at": datetime.now().isoformat(),
            "round": round_id,
            "baseline_summary": summary,
            "by_intent": by_intent,
            "total_successes": len(successes),
            "total_failures": len(failures),
            "total_partials": len(partials),
            "evolvable_skills": [],
        }

        # Identify evolvable skills (skills with failures that can be addressed)
        for intent, stats in by_intent.items():
            if stats["failures"] > 0:
                report["evolvable_skills"].append({
                    "intent": intent,
                    "failure_count": stats["failures"],
                    "avg_failure_score": (
                        sum(
                            r.get("score", r.get("judge_score", 0))
                            for r in failures
                            if r.get("intent") == intent
                        ) / stats["failures"]
                    ) if stats["failures"] > 0 else 0.0,
                    "has_bootstrappable_successes": stats["successes"] >= 1,
                })

        # Save report
        _save_json(self.run_dir / "analysis_report.json", report)

        # Print summary
        print(f"\n{'='*60}")
        print(f"Skill Evolution Analysis — Round {round_id}")
        print(f"{'='*60}")
        print(f"Total questions: {len(results)}")
        print(f"Baseline score: {summary.get('average_score', 0):.4f}")
        print(f"Successes (>=0.9): {len(successes)}")
        print(f"Partial (0.8-0.9): {len(partials)}")
        print(f"Failures (<0.8):   {len(failures)}")
        print(f"\nBy intent:")
        for intent, stats in sorted(by_intent.items()):
            marker = " ← EVOLVABLE" if stats["failures"] > 0 else ""
            print(
                f"  {intent}: {stats['total']:3d} Q | "
                f"avg={stats['avg_score']:.3f} | "
                f"success={stats['successes']}, fail={stats['failures']}"
                f"{marker}"
            )
        print(f"\nEvolvable skills: {len(report['evolvable_skills'])}")
        for s in report["evolvable_skills"]:
            print(f"  - {s['intent']}: {s['failure_count']} failures")

        return report

    # ── Stage 2: Evolve ──────────────────────────────────────────────────

    def evolve(
        self,
        target_intent: str | None = None,
        round_id: int = 0,
    ) -> dict[str, Any]:
        """Full evolution cycle: analyze → reflect → bootstrap → mutate → validate.

        Returns a dict with:
          - analysis: dict
          - reflections: list
          - bootstrapped_examples: dict
          - candidate_patches: list[PatchSchema]
          - memory_updated: bool
        """
        # Step 1: Analyze
        analysis = self.analyze(target_intent=target_intent, round_id=round_id)

        data = json.loads(self.eval_results_path.read_text("utf-8"))
        all_results = data.get("results", [])

        if target_intent:
            all_results = [r for r in all_results if r.get("intent") == target_intent]

        failures = [
            r for r in all_results
            if r.get("score", r.get("judge_score", 0)) < 0.8
        ]

        if not failures:
            print("\nNo failures to evolve from. Stopping.")
            return {
                "analysis": analysis,
                "reflections": [],
                "bootstrapped_examples": {},
                "candidate_patches": [],
                "memory_updated": False,
            }

        # Group failures by intent
        by_intent: dict[str, list[dict]] = {}
        for f in failures:
            intent = f.get("intent", "unknown")
            by_intent.setdefault(intent, []).append(f)

        # Step 2: Generate reflections per skill
        all_reflections: list[dict] = []
        for intent, intent_failures in by_intent.items():
            reflections = generate_reflections(intent_failures, skill_name=intent)
            append_reflections(reflections, skill_name=intent, round_id=round_id)
            all_reflections.extend(reflections)
            print(f"  {intent}: {len(reflections)} reflections saved to memory")

        # Step 3: Bootstrap few-shot from successes
        examples = bootstrap_examples(self.eval_results_path, min_score=0.9, max_per_skill=3)
        print(f"\n{bootstrap_summary(examples)}")

        # Step 4: Generate prompt candidates
        all_patches: list[PatchSchema] = []
        skill_configs = _load_skill_configs(self.skills_dir)

        for intent, intent_failures in by_intent.items():
            config = skill_configs.get(intent, {})
            skill_version = config.get("skill_version", "1.0.0")

            # Load current prompts
            current_assessment_ref = (
                config.get("strategy", {})
                .get("assessment", {})
                .get("system_prompt_ref", "EVIDENCE_ASSESSMENT_SYSTEM")
            )
            current_answer_ref = (
                config.get("strategy", {})
                .get("answer_generation", {})
                .get("system_prompt_ref", "ANSWER_GENERATION_SYSTEM")
            )

            current_assessment = _load_prompt_content(current_assessment_ref, self.prompts_dir)
            current_answer = _load_prompt_content(current_answer_ref, self.prompts_dir)

            # Load memory for this skill
            memories = load_memory(skill_name=intent, last_n=10)
            memory_context = format_memory_for_prompt(memories)

            # Bootstrapped examples for this skill
            skill_examples = examples.get(intent, [])

            failure_ids = [f.get("id", f.get("question_id", "")) for f in intent_failures]

            print(f"\n{'─'*50}")
            print(f"Generating candidates for: {intent}")
            print(f"  Failures: {len(intent_failures)}")
            print(f"  Memories: {len(memories)}")
            print(f"  Bootstrapped examples: {len(skill_examples)}")
            print(f"  Current assessment ref: {current_assessment_ref}")
            print(f"  Current answer ref: {current_answer_ref}")

            # --- Assessment prompt candidates ---
            if current_assessment:
                print(f"\n  [Assessment] Generating candidates...")
                candidates = generate_prompt_candidates(
                    prompt_role="evidence_assessment",
                    skill_name=intent,
                    current_prompt=current_assessment,
                    reflections=memories,
                    failures=intent_failures,
                    bootstrapped_examples=[
                        (ex["question"], ex["system_answer"])
                        for ex in skill_examples
                    ],
                    num_candidates=2,
                )
                if candidates:
                    patches = build_prompt_update_patches(
                        candidates=candidates,
                        skill_name=intent,
                        skill_version=skill_version,
                        failure_ids=failure_ids,
                        prompt_role="evidence_assessment",
                        current_prompt_ref=current_assessment_ref,
                    )
                    all_patches.extend(patches)
                    print(f"    → {len(candidates)} candidate(s), {len(patches)} patch(es)")

            # --- Answer generation candidates ---
            if current_answer:
                print(f"\n  [AnswerGen] Generating candidates...")
                candidates = generate_prompt_candidates(
                    prompt_role="answer_generation",
                    skill_name=intent,
                    current_prompt=current_answer,
                    reflections=memories,
                    failures=intent_failures,
                    bootstrapped_examples=[
                        (ex["question"], ex["system_answer"])
                        for ex in skill_examples
                    ],
                    num_candidates=2,
                )
                if candidates:
                    patches = build_prompt_update_patches(
                        candidates=candidates,
                        skill_name=intent,
                        skill_version=skill_version,
                        failure_ids=failure_ids,
                        prompt_role="answer_generation",
                        current_prompt_ref=current_answer_ref,
                    )
                    all_patches.extend(patches)
                    print(f"    → {len(candidates)} candidate(s), {len(patches)} patch(es)")

        # Save patches
        _save_json(
            self.run_dir / "candidate_patches.json",
            [p.model_dump(mode="json") for p in all_patches],
        )

        result = {
            "analysis": analysis,
            "reflections": all_reflections,
            "bootstrapped_examples": examples,
            "candidate_patches": all_patches,
            "memory_updated": len(all_reflections) > 0,
        }

        # Save final report
        _save_json(
            self.run_dir / "evolution_report.json",
            {
                "run_id": analysis["run_id"],
                "round": round_id,
                "generated_at": datetime.now().isoformat(),
                "summary": analysis["baseline_summary"],
                "by_intent": analysis["by_intent"],
                "reflection_count": len(all_reflections),
                "bootstrapped_examples_count": sum(len(v) for v in examples.values()),
                "candidate_patch_count": len(all_patches),
                "candidate_patches": [
                    {
                        "patch_id": p.patch_id,
                        "skill_name": p.skill_name,
                        "patch_type": p.patch_type,
                        "target_path": p.target_path,
                        "confidence": p.confidence,
                    }
                    for p in all_patches
                ],
            },
        )

        return result

    # ── Stage 3: Apply ───────────────────────────────────────────────────

    def apply(self, patches: list[PatchSchema]) -> dict[str, Any]:
        """Apply validated patches to config files."""
        from .apply import PatchApplier
        from .rollback import SnapshotManager

        snap = SnapshotManager(self.run_dir or Path("."), self.skills_dir, self.prompts_dir)
        snap.snapshot_before()

        applier = PatchApplier(self.skills_dir, self.prompts_dir)
        applied = []
        failed = []

        for patch in patches:
            try:
                skill_configs = _load_skill_configs(self.skills_dir)
                config = skill_configs.get(patch.skill_name, {})
                applier.apply_to_disk(config, [patch], round_id="auto")
                applied.append(patch.patch_id)
                print(f"  APPLIED: {patch.patch_id}")
            except Exception as e:
                failed.append({"patch_id": patch.patch_id, "error": str(e)})
                print(f"  FAILED: {patch.patch_id}: {e}")

        snap.snapshot_after()

        result = {
            "applied": applied,
            "failed": failed,
            "snapshot_pre": snap.before_snapshot_dir,
            "snapshot_post": snap.after_snapshot_dir,
        }

        if self.run_dir:
            _save_json(self.run_dir / "apply_result.json", result)

        return result


# ── Helpers ────────────────────────────────────────────────────────────────

def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


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


def _load_prompt_content(prompt_ref: str, prompts_dir: Path) -> str:
    if not prompts_dir.is_dir():
        return ""
    import yaml
    for f in prompts_dir.glob("*.yaml"):
        try:
            data = yaml.safe_load(f.read_text("utf-8"))
            if data and data.get("prompt_ref") == prompt_ref:
                return data.get("content", "")
        except Exception:
            pass
    return ""
