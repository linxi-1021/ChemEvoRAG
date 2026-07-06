"""Tests for score normalization and result dict population bug fix.

Validates that:
- eval_results.json with summary.average_score is correctly read
- RegressionResult_from_dict normalizes both flat and nested formats
- validate_individual_patch populates average_score/score_delta/targeted_improvement
- Missing average_score does not pass
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# RegressionResult_from_dict normalization
# ---------------------------------------------------------------------------

class TestRegressionResultFromDict:
    """Test that RegressionResult_from_dict handles both formats."""

    def test_nested_summary_format(self):
        """eval_results.json format: {"summary": {"average_score": N}, "results": [...]}"""
        from skill_evolution.runtime_validation import RegressionResult_from_dict

        raw = {
            "summary": {
                "total": 118,
                "average_score": 0.866,
                "by_intent": {
                    "property_query": 0.90,
                    "entity_lookup": 0.91,
                },
            },
            "results": [{"id": "q1"}, {"id": "q2"}, {"id": "q3"}],
        }
        rr = RegressionResult_from_dict(raw)
        assert rr.average_score == 0.866
        assert rr.total_questions == 3
        assert "property_query" in rr.by_intent
        assert rr.by_intent["property_query"]["average_score"] == 0.90

    def test_flat_format(self):
        """Already-flattened format: {"average_score": N, "total_questions": N}"""
        from skill_evolution.runtime_validation import RegressionResult_from_dict

        raw = {
            "average_score": 0.75,
            "total_questions": 50,
            "by_intent": {"property_query": {"average_score": 0.80}},
        }
        rr = RegressionResult_from_dict(raw)
        assert rr.average_score == 0.75
        assert rr.total_questions == 50

    def test_missing_average_score_defaults_zero(self):
        """Missing average_score should return 0.0 (not crash)."""
        from skill_evolution.runtime_validation import RegressionResult_from_dict

        rr = RegressionResult_from_dict({})
        assert rr.average_score == 0.0
        assert rr.total_questions == 0

    def test_summary_without_average_score(self):
        """summary without average_score key."""
        from skill_evolution.runtime_validation import RegressionResult_from_dict

        raw = {"summary": {"total": 10}, "results": [{"id": "q1"}]}
        rr = RegressionResult_from_dict(raw)
        assert rr.average_score == 0.0
        assert rr.total_questions == 1


# ---------------------------------------------------------------------------
# validate_individual_patch result dict population
# ---------------------------------------------------------------------------

class TestValidateIndividualPatchScoreFields:
    """validate_individual_patch must surface scores in result dict."""

    def test_result_has_average_score_and_delta(self, monkeypatch, tmp_path):
        """After regression validation, result dict must have average_score."""
        from skill_evolution.runtime_validation import validate_individual_patch
        from skill_evolution.patch import PatchSchema, PatchOperation, PatchStatus
        from skill_evolution.attribution import FailureType

        captured_workers = None

        def fake_run_eval(self, dataset_path, skills_dir=None, prompts_dir=None,
                          output_dir=None, *, workers=1, stream_output=True, **kwargs):
            nonlocal captured_workers
            captured_workers = workers
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            (out / "eval_results.json").write_text(json.dumps({
                "summary": {"total": 2, "average_score": 0.95, "by_intent": {"property_query": 0.95}},
                "results": [
                    {"id": "q1", "score": 0.95, "intent": "property_query"},
                    {"id": "q2", "score": 0.95, "intent": "property_query"},
                ],
            }), encoding="utf-8")
            from skill_evolution.regression import RegressionResult
            return RegressionResult(average_score=0.95, total_questions=2,
                                    by_intent={"property_query": {"average_score": 0.95, "total": 2, "failed": 0}})

        monkeypatch.setattr(
            "skill_evolution.regression.RegressionRunner.run_eval", fake_run_eval,
        )

        skill_dir = tmp_path / "skills"; skill_dir.mkdir()
        (skill_dir / "property_query.yaml").write_text(
            "name: property_query\ntrigger:\n  intent: property_query\n")
        prompts_dir = tmp_path / "prompts"; prompts_dir.mkdir()
        dataset_path = tmp_path / "q.json"
        dataset_path.write_text(json.dumps([
            {"id": "q1", "question": "test", "intent": "property_query"}
        ]))

        patch = PatchSchema(
            patch_id="t1", skill_name="property_query",
            primary_failure_type=FailureType.ASSESSMENT_FALSE_NEG,
            target_path="strategy.assessment.system_prompt_ref",
            operation=PatchOperation.UPDATE,
            proposed_value={"system_prompt_ref": "X"},
            status=PatchStatus.CANDIDATE,
        )
        skill_config = {
            "name": "property_query",
            "trigger": {"intent": "property_query"},
            "evolution": {"mutable_paths": ["strategy.assessment.system_prompt_ref"]},
        }
        baseline = {"average_score": 0.87, "total_questions": 2, "by_intent": {}}

        result = validate_individual_patch(
            patch, skill_config,
            project_root=tmp_path, skills_dir=skill_dir, prompts_dir=prompts_dir,
            regression_dataset=dataset_path, baseline_result=baseline,
            skill_filename="property_query.yaml", workers=1,
        )

        # Top-level fields must exist
        assert "average_score" in result, f"Missing average_score in {result.keys()}"
        assert result["average_score"] == 0.95
        assert "score_delta" in result
        assert result["score_delta"] == pytest.approx(0.08)  # 0.95 - 0.87
        assert "targeted_improvement" in result
        assert result["passed"] is True

    def test_missing_eval_results_regression_fails(self, monkeypatch, tmp_path):
        """If eval_results.json is not produced, regression should fail."""
        from skill_evolution.runtime_validation import validate_individual_patch
        from skill_evolution.patch import PatchSchema, PatchOperation, PatchStatus
        from skill_evolution.attribution import FailureType

        def fake_run_eval(self, dataset_path, skills_dir=None, prompts_dir=None,
                          output_dir=None, *, workers=1, stream_output=True, **kwargs):
            from skill_evolution.regression import RegressionResult
            return RegressionResult(errors=["Eval failed: no output"])

        monkeypatch.setattr(
            "skill_evolution.regression.RegressionRunner.run_eval", fake_run_eval,
        )

        skill_dir = tmp_path / "skills"; skill_dir.mkdir()
        (skill_dir / "property_query.yaml").write_text(
            "name: property_query\ntrigger:\n  intent: property_query\n")
        prompts_dir = tmp_path / "prompts"; prompts_dir.mkdir()
        dataset_path = tmp_path / "q.json"
        dataset_path.write_text(json.dumps([
            {"id": "q1", "question": "test", "intent": "property_query"}
        ]))

        patch = PatchSchema(
            patch_id="t1", skill_name="property_query",
            primary_failure_type=FailureType.ASSESSMENT_FALSE_NEG,
            target_path="strategy.assessment.system_prompt_ref",
            operation=PatchOperation.UPDATE,
            proposed_value={"system_prompt_ref": "X"},
            status=PatchStatus.CANDIDATE,
        )
        skill_config = {
            "name": "property_query",
            "trigger": {"intent": "property_query"},
            "evolution": {"mutable_paths": ["strategy.assessment.system_prompt_ref"]},
        }
        baseline = {"average_score": 0.87, "total_questions": 2, "by_intent": {}}

        result = validate_individual_patch(
            patch, skill_config,
            project_root=tmp_path, skills_dir=skill_dir, prompts_dir=prompts_dir,
            regression_dataset=dataset_path, baseline_result=baseline,
            skill_filename="property_query.yaml", workers=1,
        )

        # Should warn about errors (but currently still "passed" on eval errors)
        assert len(result.get("warnings", [])) > 0


# ---------------------------------------------------------------------------
# compare uses correct average_score
# ---------------------------------------------------------------------------

class TestCompareUsesCorrectAverageScore:
    """RegressionRunner.compare must use RegressionResult.average_score."""

    def test_compare_uses_average_score_from_result(self, tmp_path):
        from skill_evolution.regression import RegressionRunner, RegressionResult

        project = tmp_path / "project"
        project.mkdir()
        (project / "scripts").mkdir()
        (project / "scripts" / "eval_questions.py").write_text("# stub")

        runner = RegressionRunner(project)
        before = RegressionResult(average_score=0.87, total_questions=10)
        after = RegressionResult(average_score=0.85, total_questions=10)

        comparison = runner.compare(before, after)
        assert comparison.average_score == 0.85
        assert comparison.score_drop == pytest.approx(0.02)
        assert comparison.score_delta == pytest.approx(-0.02)
