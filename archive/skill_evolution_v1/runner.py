"""Evolution Pipeline Runner. Orchestrates all stages.

Usage:
    from skill_evolution.runner import EvolutionRunner, RunMode
    runner = EvolutionRunner(project_root)
    result = runner.run(RunMode.VALIDATE_ONLY)
"""

from __future__ import annotations

import json
import shutil
import sys
import traceback
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from .config import EvolutionConfig, get_config, set_config
from .attribution import generate_trace_report, TraceReport
from .mutation import generate_all_mutations
from .distillation import distill_new_templates, promote_to_cross_skill
from .integration import integrate_candidates, IntegrationResult
from .validation import validate_patch
from .patch import PatchSchema, PatchStatus
from .metrics import RoundMetrics, write_metrics


class RunMode(str, Enum):
    ANALYZE_ONLY = "analyze_only"      # Stage 1-3
    DRY_RUN = "dry_run"                # Stage 1-5 (no regression, no writeback)
    VALIDATE_ONLY = "validate_only"    # Stage 1-6 (regression, no writeback)
    APPLY = "apply"                    # Stage 1-7 (full pipeline + writeback)


class EvolutionRunner:
    """Main pipeline orchestrator."""

    def __init__(self, project_root: Path, config: EvolutionConfig | None = None):
        self.project_root = Path(project_root)
        if config:
            set_config(config)
        self.cfg = get_config()

        # Default paths
        self.eval_dir = self.project_root / "data" / "eval"
        self.evolution_dir = self.project_root / "data" / "evolution"
        self.skills_dir = self.project_root / "config" / "skills"
        self.prompts_dir = self.project_root / "config" / "prompts"
        self.eval_results_path = self.eval_dir / "eval_results.json"
        self.react_logs_dir = self.eval_dir / "react_logs"
        self.dataset_path = self.eval_dir / "all_questions.json"

        self.run_dir: Path | None = None
        self.log_lines: list[str] = []

    def _log(self, msg: str) -> None:
        self.log_lines.append(msg)
        print(msg)

    def _save_json(self, path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def run(self, mode: RunMode, **overrides: Any) -> dict[str, Any]:
        """Execute the evolution pipeline."""
        # Apply overrides
        for k, v in overrides.items():
            if hasattr(self, k):
                setattr(self, k, Path(v) if k.endswith("_dir") or k.endswith("_path") else v)

        # Create run directory
        run_id = overrides.get("run_id") or f"run_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
        output_dir = overrides.get("output_dir")
        if output_dir:
            self.run_dir = Path(output_dir) / run_id
        else:
            self.run_dir = self.evolution_dir / "runs" / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._log(f"Run directory: {self.run_dir}")

        try:
            # ── Stage 1-3: Trace → Evaluate → Attribute ──────────────────
            report = self._stage_1_to_3()

            if mode == RunMode.ANALYZE_ONLY:
                self._log("\n--analyze-only: stopping after analysis.")
                self._write_log()
                return {"status": "analyzed", "report": report.model_dump(mode="json")}

            # ── Stage 4A: Mutation (failure-driven) ───────────────────────
            success_data = self._run_success_analysis(report)
            mutation_patches = self._stage_4a_mutation(report, success_data)

            # ── Stage 4B: Distillation (success-driven) ───────────────────
            distillation_patches = self._stage_4b_distillation(report, success_data)

            # ── Stage 5: Integration ──────────────────────────────────────
            integration = self._stage_5_integration(mutation_patches, distillation_patches)

            if mode == RunMode.DRY_RUN:
                self._log(f"\n--dry-run: {len(integration.selected)} patches selected.")
                self._print_diff(integration.selected)
                self._save_json(self.run_dir / "final_report.json", self._build_final_report(report, integration))
                self._write_log()
                return {"status": "dry_run", "integration": integration}

            # ── Stage 6: Validation ──────────────────────────────────────
            if not integration.selected:
                self._log("\nNo patches selected — nothing to validate.")
                self._save_json(self.run_dir / "final_report.json", self._build_final_report(report, integration))
                self._write_log()
                return {"status": "no_patches"}

            validated = self._stage_6_validate(integration.selected, overrides)

            # ── Refine Loop: if prompt patches failed, go back to Stage 4A ──
            failed_prompt = [p for p in integration.selected
                           if p.patch_type == "prompt_content_update"
                           and p not in validated]
            if failed_prompt and mode != RunMode.DRY_RUN:
                version_history: list[dict] = []
                for refine_round in range(self.cfg.max_refine_iterations):
                    self._log(f"\n[Refine Round {refine_round + 1}/{self.cfg.max_refine_iterations}] "
                              f"{len(failed_prompt)} prompt patch(es) failed — returning to Stage 4A")

                    # Collect feedback from failed patches
                    previous_feedback = {}
                    for p in failed_prompt:
                        pd = self.run_dir / "per_patch_evals" / p.patch_id
                        fb = self._collect_regression_feedback(pd)
                        if fb:
                            previous_feedback[p.patch_id] = fb

                    if not previous_feedback:
                        self._log("  No regression feedback to feed back — stopping refine")
                        break

                    # Accumulate version history: previous versions' content + feedback
                    for p in failed_prompt:
                        pa = p.prompt_artifacts[0] if p.prompt_artifacts else None
                        fb = previous_feedback.get(p.patch_id)
                        if pa and fb:
                            version_history.append({
                                "version": 2 + refine_round,
                                "content_snippet": pa.content[:500] if pa.content else "",
                                "feedback": fb,
                            })

                    # Go back to Stage 4A with feedback + history
                    refined_patches = self._stage_4a_mutation_refine(
                        report, success_data, previous_feedback, version_history)

                    if not refined_patches:
                        self._log("  Stage 4A returned no refined patches — stopping refine")
                        break

                    # Re-integrate (only refined patches)
                    refined_integration = self._stage_5_integration(refined_patches, [])

                    # Re-validate
                    refined_validated = self._stage_6_validate(
                        refined_integration.selected, overrides)

                    # Update
                    if refined_validated:
                        validated.extend(refined_validated)
                        self._log(f"  Refine round {refine_round + 1}: "
                                  f"{len(refined_validated)} patch(es) passed")
                        failed_prompt = [p for p in refined_integration.selected
                                       if p.patch_type == "prompt_content_update"
                                       and p not in refined_validated]
                        if not failed_prompt:
                            self._log("  All prompt patches passed — refine complete")
                            break
                    else:
                        self._log("  No patches passed this refine round")

            if mode == RunMode.VALIDATE_ONLY:
                self._log(f"\n--validate-only: {len(validated)} patches validated.")
                self._save_json(self.run_dir / "final_report.json", self._build_final_report(report, integration, validated))
                self._write_log()
                return {"status": "validate_only", "validated": validated}

            # ── Stage 7: Apply ───────────────────────────────────────────
            if mode == RunMode.APPLY:
                apply_result = self._stage_7_apply(validated)
                self._log(f"\nAPPLY: {apply_result}")
                self._write_metrics(report, validated)
                self._write_log()
                return {"status": "applied", "apply": apply_result}

        except Exception as e:
            self._log(f"\nFATAL: {e}")
            self._log(traceback.format_exc())
            self._write_log()
            raise

        return {"status": "unknown"}

    # ── Stage implementations ─────────────────────────────────────────────

    def _stage_1_to_3(self) -> TraceReport:
        """Stages 1-3: Trace → Evaluate → Attribute."""
        if not self.eval_results_path.exists():
            raise FileNotFoundError(f"eval_results.json not found: {self.eval_results_path}")

        skill_configs = self._load_skill_configs()
        report = generate_trace_report(
            self.eval_results_path, self.react_logs_dir, skill_configs,
        )
        self._save_json(self.run_dir / "trace_report.json",
                        report.model_dump(mode="json"))
        self._log(f"Trace report: {len(report.failures)} failures, "
                  f"{len(report.partial_successes)} partial, "
                  f"{len(report.success_patterns)} success")
        return report

    def _run_success_analysis(self, report: TraceReport) -> dict:
        """Run LLM-driven success pattern analysis (shared by 4A + 4B)."""
        self._log("\n[SuccessAnalysis] Analyzing successful traces...")
        # Build success trace dicts from report
        success_traces = []
        for sp in report.success_patterns:
            success_traces.append({
                "intent": sp.intent,
                "pattern": sp.pattern,
                "query_structure": sp.query_structure,
                "evidence_type": sp.evidence_type,
                "retrieval_rounds": [],  # SuccessPattern doesn't carry full trace
            })
        if not success_traces:
            return {}

        from .success_analysis import analyze_success_patterns
        failure_by_cluster = {}
        for cluster in report.analysis_clusters:
            failure_traces = []
            for f in cluster.get("failures", []):
                qid = f.question_id if hasattr(f, 'question_id') else f.get("question_id", "")
                failure_traces.append({
                    "question_id": qid,
                    "intent": getattr(f, 'skill_name', ''),
                    "retrieval_rounds": getattr(f, 'retrieval_rounds', []),
                    "assessment_result": getattr(f, 'assessment_result', ''),
                    "react_rounds_used": getattr(f, 'react_rounds_used', 1),
                })
            if failure_traces:
                failure_by_cluster[cluster.get("cluster_id", "unknown")] = failure_traces

        result = analyze_success_patterns(success_traces, failure_by_cluster)
        self._log(f"  Global: {len(result.get('global', {}).get('by_intent', {}))} intents analyzed")
        self._log(f"  Contrastive: {len(result.get('contrastive', {}))} clusters analyzed")
        return result

    def _stage_4a_mutation(self, report: TraceReport, success_data: dict | None = None) -> list[PatchSchema]:
        """Stage 4A: LLM-directed mutation for each analysis cluster."""
        all_patches: list[PatchSchema] = []
        skill_configs = self._load_skill_configs()
        current_prompts = self._load_current_prompts()

        # Build failure_records_map for LLM analysis
        failure_records_map: dict[str, dict] = {}
        for fr in report.failures:
            failure_records_map[fr.question_id] = fr.model_dump(mode="json")

        for cluster in report.analysis_clusters:
            self._log(f"\n[Mutation] Cluster: {cluster['cluster_id']}")
            self._log(f"  Failure type: {cluster['failure_type']}, "
                      f"Skills: {cluster['affected_skills']}")

            # Collect failure records for this cluster
            cluster_failures = []
            for f in cluster.get("failures", []):
                qid = f.question_id if hasattr(f, 'question_id') else f.get("question_id", "")
                if qid in failure_records_map:
                    cluster_failures.append(failure_records_map[qid])

            # Collect affected skill configs
            affected_configs = {
                s: skill_configs[s] for s in cluster["affected_skills"]
                if s in skill_configs
            }

            patches = generate_all_mutations(
                cluster=cluster,
                failure_records=cluster_failures,
                skill_configs=affected_configs,
                current_prompts=current_prompts,
                failure_traces=cluster_failures,
                success_analysis_data=success_data,
            )
            self._log(f"  Generated {len(patches)} mutation patch(es)")
            all_patches.extend(patches)

        self._save_json(self.run_dir / "mutation_patches.json",
                        [p.model_dump(mode="json") for p in all_patches])
        return all_patches

    def _stage_4a_mutation_refine(self, report: TraceReport, success_data: dict,
                                   previous_feedback: dict,
                                   version_history: list[dict] | None = None) -> list[PatchSchema]:
        """Stage 4A refine: re-run mutation with previous regression feedback.

        The previous version's regression feedback is fed back to Stage 2 LLM,
        which analyzes why the previous version failed and generates better rules.
        Stage 3 then produces the next version (V3, V4, ...).
        """
        all_patches: list[PatchSchema] = []
        skill_configs = self._load_skill_configs()
        current_prompts = self._load_current_prompts()

        for cluster in report.analysis_clusters:
            cluster_id = cluster.get("cluster_id", "")

            # Find feedback matching this cluster's patches
            cluster_feedback = {}
            for pid, fb in previous_feedback.items():
                cluster_feedback[pid] = fb

            if not cluster_feedback:
                continue

            # Collect failure records
            failure_records_map = {}
            for fr in report.failures:
                failure_records_map[fr.question_id] = fr.model_dump(mode="json")
            cluster_failures = []
            for f in cluster.get("failures", []):
                qid = f.question_id if hasattr(f, 'question_id') else f.get("question_id", "")
                if qid in failure_records_map:
                    cluster_failures.append(failure_records_map[qid])

            affected_configs = {
                s: skill_configs[s] for s in cluster["affected_skills"]
                if s in skill_configs
            }

            self._log(f"  [4A Refine] Cluster: {cluster_id}, "
                      f"feedback from {len(cluster_feedback)} failed patch(es)")

            # Attach version history to cluster for Stage 2 LLM analysis
            enriched_cluster = dict(cluster)
            enriched_cluster["version_history"] = version_history or []
            patches = generate_all_mutations(
                cluster=enriched_cluster,
                failure_records=cluster_failures,
                skill_configs=affected_configs,
                current_prompts=current_prompts,
                failure_traces=cluster_failures,
                success_analysis_data=success_data,
                previous_feedback=cluster_feedback,
            )
            self._log(f"  Generated {len(patches)} refined mutation patch(es)")
            all_patches.extend(patches)

        return all_patches

    def _stage_4b_distillation(self, report: TraceReport, success_data: dict | None = None) -> list[PatchSchema]:
        """Stage 4B: Template distillation from success patterns."""
        skill_configs = self._load_skill_configs()

        # ADD
        add_patches = distill_new_templates(report.success_patterns, success_data)

        # Cross-skill promotion
        cross_patches = promote_to_cross_skill(report.success_patterns, skill_configs)

        all_patches = add_patches + cross_patches
        self._log(f"\n[Distillation] {len(add_patches)} template_add, "
                  f"{len(cross_patches)} cross-skill promotions")
        self._save_json(self.run_dir / "distillation_patches.json",
                        [p.model_dump(mode="json") for p in all_patches])
        return all_patches

    def _stage_5_integration(
        self, mutation: list[PatchSchema], distillation: list[PatchSchema],
    ) -> IntegrationResult:
        """Stage 5: Integrate candidates."""
        skill_configs = self._load_skill_configs()
        result = integrate_candidates(mutation, distillation, skill_configs)
        self._log(f"\n[Integration] {len(result.selected)} selected, "
                  f"{len(result.rejected)} rejected, "
                  f"{len(result.conflicts_resolved)} conflicts resolved")
        self._save_json(self.run_dir / "integration_result.json", {
            "selected": [p.model_dump(mode="json") for p in result.selected],
            "rejected": result.rejected,
            "conflicts": result.conflicts_resolved,
        })
        return result

    def _stage_6_validate(
        self, patches: list[PatchSchema], overrides: dict,
    ) -> list[PatchSchema]:
        """Stage 6: Schema + semantic + regression validation."""
        skill_configs = self._load_skill_configs()

        # Schema + semantic validation
        accepted: list[PatchSchema] = []
        rejected: list[dict] = []
        for patch in patches:
            config = skill_configs.get(patch.skill_name, {})
            vresult = validate_patch(
                patch, config,
                skill_filename=f"{patch.skill_name}.yaml",
                prompts_dir=self.prompts_dir,
                all_patches=list(patches),
            )
            if vresult.valid:
                accepted.append(patch)
            else:
                entry = patch.model_dump(mode="json")
                entry["rejection_reason"] = "; ".join(vresult.errors)
                rejected.append(entry)

        self._log(f"Schema validation: {len(accepted)} accepted, {len(rejected)} rejected")
        self._save_json(self.run_dir / "selected_patches.json",
                        [p.model_dump(mode="json") for p in accepted])
        self._save_json(self.run_dir / "rejected_patches.json", rejected)

        # Regression validation (delegated to evolve.py for backward compat)
        # Only runs if workers > 0 and dataset exists
        if overrides.get("skip_regression"):
            self._log("Skipping regression (--skip-regression)")
        elif self.dataset_path.exists() and overrides.get("workers", 0) > 0:
            # Call existing validate_individual_patch in runtime_validation
            from .runtime_validation import validate_individual_patch
            workers = overrides.get("workers", 1)
            regression_limit = overrides.get("regression_limit")
            regression_seed = overrides.get("regression_seed", 42)
            stream_output = not overrides.get("quiet_regression", False)

            baseline_data = json.loads(self.eval_results_path.read_text("utf-8")) if self.eval_results_path.exists() else {}
            baseline_result = {
                "average_score": baseline_data.get("summary", {}).get("average_score", 0.0),
                "total_questions": len(baseline_data.get("results", [])),
                "by_intent": {
                    k: {"average_score": v}
                    for k, v in baseline_data.get("summary", {}).get("by_intent", {}).items()
                },
                "results_path": str(self.eval_results_path),
            } if baseline_data else None

            still_valid: list[PatchSchema] = []
            for idx, patch in enumerate(accepted, 1):
                self._log(f"[{idx}/{len(accepted)}] Validating {patch.patch_id}")
                config = skill_configs.get(patch.skill_name, {})
                persist_dir = self.run_dir / "per_patch_evals" / patch.patch_id
                result = validate_individual_patch(
                    patch, config,
                    project_root=self.project_root,
                    skills_dir=self.skills_dir,
                    prompts_dir=self.prompts_dir,
                    regression_dataset=self.dataset_path,
                    baseline_result=baseline_result,
                    skill_filename=f"{patch.skill_name}.yaml",
                    workers=workers,
                    regression_limit=regression_limit,
                    regression_seed=regression_seed,
                    stream_output=stream_output,
                    persist_dir=persist_dir,
                    original_eval_path=self.eval_results_path,
                    all_patches=list(accepted),
                )
                if result["passed"]:
                    still_valid.append(patch)
                    self._log(f"  [OK] score={result.get('average_score', 0):.4f}, "
                              f"delta={result.get('score_delta', 0):+.4f}")
                else:
                    entry = patch.model_dump(mode="json")
                    entry["rejection_reason"] = result.get("errors", [])[:3]
                    rejected.append(entry)
                    self._log(f"  [FAIL] {result.get('errors', [])[:2]}")

            accepted = still_valid

        # Composition validation
        if len(accepted) > 1:
            from .runtime_validation import validate_composition
            comp = validate_composition(
                accepted, skill_configs,
                project_root=self.project_root,
                skills_dir=self.skills_dir,
                prompts_dir=self.prompts_dir,
                regression_dataset=self.dataset_path,
                workers=overrides.get("workers", 1),
                regression_limit=overrides.get("regression_limit"),
                regression_seed=overrides.get("regression_seed", 42),
                stream_output=not overrides.get("quiet_regression", False),
            )
            if not comp.get("passed"):
                self._log("Composition validation FAILED")

        return accepted

    def _stage_7_apply(self, patches: list[PatchSchema]) -> dict[str, Any]:
        """Stage 7: Writeback with snapshot + rollback."""
        from .apply import PatchApplier
        from .rollback import SnapshotManager

        snapshot_mgr = SnapshotManager(self.run_dir, self.skills_dir, self.prompts_dir)
        snapshot_mgr.snapshot_before()

        applier = PatchApplier(self.skills_dir, self.prompts_dir)
        applied = []

        for patch in patches:
            try:
                config = self._load_skill_configs().get(patch.skill_name, {})
                applier.apply_to_disk(config, [patch], round_id="auto")
                applied.append(patch.patch_id)
                self._log(f"  APPLIED {patch.patch_id}")
            except Exception as e:
                self._log(f"  FAILED {patch.patch_id}: {e}")

        snapshot_mgr.snapshot_after()
        self._save_json(self.run_dir / "applied_patches.json", applied)

        return {"applied": len(applied), "total": len(patches)}

    # ── Helpers ───────────────────────────────────────────────────────────

    def _load_skill_configs(self) -> dict[str, dict]:
        configs: dict[str, dict] = {}
        if not self.skills_dir.is_dir():
            return configs
        import yaml
        for f in self.skills_dir.glob("*.yaml"):
            try:
                data = yaml.safe_load(f.read_text("utf-8"))
                intent = data.get("trigger", {}).get("intent", "")
                if intent:
                    configs[intent] = data
            except Exception:
                pass
        return configs

    def _load_current_prompts(self) -> dict[str, dict[str, str]]:
        """Load prompts needed for LLM analysis. {role: {prompt_ref, content}}."""
        prompts: dict[str, dict[str, str]] = {}
        if not self.prompts_dir.is_dir():
            return prompts
        import yaml
        for f in self.prompts_dir.glob("*.yaml"):
            try:
                data = yaml.safe_load(f.read_text("utf-8"))
                ref = data.get("prompt_ref", "")
                content = data.get("content", "")
                if ref and content:
                    role = "evidence_assessment" if "ASSESSMENT" in ref else "answer_generation" if "ANSWER" in ref or "GENERATION" in ref else ""
                    if role:
                        prompts[role] = {"prompt_ref": ref, "content": content}
            except Exception:
                pass
        return prompts

    def _print_diff(self, patches: list[PatchSchema]) -> None:
        """Print dry-run diff for each patch."""
        for p in patches:
            self._log(f"\n--- {p.patch_id} ---")
            self._log(f"  Type: {p.patch_type}")
            self._log(f"  Skill: {p.skill_name}")
            self._log(f"  Target: {p.target_path}")
            self._log(f"  Op: {p.operation.value}")
            if hasattr(p, 'current_value') and p.current_value:
                self._log(f"  Before: {json.dumps(p.current_value, ensure_ascii=False)}")
            if hasattr(p, 'proposed_value') and p.proposed_value:
                self._log(f"  After:  {json.dumps(p.proposed_value, ensure_ascii=False)}")
            self._log(f"  Reason: {p.rationale[:200]}")
            self._log(f"  Risk: {p.risk_level} | Confidence: {p.confidence:.2f}")

    def _build_final_report(
        self, report: TraceReport, integration: IntegrationResult,
        validated: list[PatchSchema] | None = None,
    ) -> dict[str, Any]:
        return {
            "run_id": report.run_id,
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "total_questions": report.total_questions,
                "average_score": report.average_score,
                "failures": len(report.failures),
                "partial_successes": len(report.partial_successes),
                "coverage_gaps": len(report.coverage_gaps),
                "success_patterns": len(report.success_patterns),
            },
            "candidate_patches": len(integration.selected) + len(integration.rejected),
            "selected_patches": [p.patch_id for p in integration.selected],
            "validated_patches": [p.patch_id for p in (validated or [])],
            "rejected_patches": integration.rejected,
            "conflicts_resolved": integration.conflicts_resolved,
        }

    def _write_metrics(self, report: TraceReport, applied: list[PatchSchema]) -> None:
        metrics_path = self.evolution_dir / "metrics.json"
        rm = RoundMetrics(
            run_id=report.run_id,
            before_score=report.average_score,
            applied_patches=[p.patch_id for p in applied],
        )
        write_metrics(metrics_path, rm)

    def _collect_regression_feedback(self, persist_dir: Path) -> dict | None:
        """Build regression feedback by comparing original eval (baseline) vs after-eval output."""
        # Baseline = original eval results (fixed, never re-run)
        baseline_path = self.eval_results_path

        # After-eval = patch regression output (first of 3 parallel runs)
        after_path = persist_dir / "eval_output_0" / "eval_results.json"

        # Fall back to searching for any after-eval output
        if not after_path.exists():
            for d in persist_dir.iterdir():
                if d.is_dir() and d.name.startswith("eval_output"):
                    after_path = d / "eval_results.json"
                    if after_path.exists():
                        break

        if not persist_dir.exists():
            return None
        if not baseline_path.exists() or not after_path.exists():
            return None

        try:
            baseline = json.loads(baseline_path.read_text("utf-8"))
            after = json.loads(after_path.read_text("utf-8"))

            v1_scores: dict[str, float] = {}
            v2_scores: dict[str, float] = {}
            v1_answers: dict[str, str] = {}
            v2_answers: dict[str, str] = {}
            questions: dict[str, str] = {}

            for r in baseline.get("results", []):
                key = f"{r.get('id', '')}_{r.get('source_paper', '')}"
                v1_scores[key] = r.get("score", 0.0)
                v1_answers[key] = r.get("system_answer", "")[:300]
                questions[key] = r.get("question", "")

            for r in after.get("results", []):
                key = f"{r.get('id', '')}_{r.get('source_paper', '')}"
                v2_scores[key] = r.get("score", 0.0)
                v2_answers[key] = r.get("system_answer", "")[:300]
                if key not in questions:
                    questions[key] = r.get("question", "")

            from .mutation import _extract_regression_feedback
            return _extract_regression_feedback(v1_scores, v2_scores, v1_answers, v2_answers, questions)
        except Exception:
            return None

    def _make_regression_callback(self, workers, regression_limit, regression_seed, stream_output, baseline_result, accepted):
        """Create a regression callback for mutation.py's 4A-internal refine loop."""
        from .runtime_validation import validate_individual_patch

        def _callback(patch):
            persist_dir = self.run_dir / "per_patch_evals" / f"{patch.patch_id}_4a_refine"
            config = self._load_skill_configs().get(patch.skill_name, {})
            result = validate_individual_patch(
                patch, config,
                project_root=self.project_root,
                skills_dir=self.skills_dir,
                prompts_dir=self.prompts_dir,
                regression_dataset=self.dataset_path,
                baseline_result=baseline_result,
                skill_filename=f"{patch.skill_name}.yaml",
                workers=workers,
                regression_limit=regression_limit or 5,
                regression_seed=regression_seed,
                stream_output=stream_output,
                persist_dir=persist_dir,
                original_eval_path=self.eval_results_path,
                all_patches=list(accepted),
            )
            # Collect feedback for refine
            feedback = self._collect_regression_feedback(persist_dir)
            return {"passed": result.get("passed", False), "feedback": feedback, "errors": result.get("errors", [])}

        return _callback

    def _load_v1_prompt_content(self, prompt_ref: str) -> str:
        """Load the content of a prompt from config/prompts/ by its prompt_ref."""
        if not prompt_ref or not self.prompts_dir.is_dir():
            return ""
        import yaml
        for f in self.prompts_dir.glob("*.yaml"):
            try:
                data = yaml.safe_load(f.read_text("utf-8"))
                if data and data.get("prompt_ref") == prompt_ref:
                    return data.get("content", "")
            except Exception:
                pass
        return ""

    def _write_log(self) -> None:
        if self.run_dir:
            log_path = self.run_dir / "evolution_log.txt"
            log_path.write_text("\n".join(self.log_lines) + "\n", encoding="utf-8")
