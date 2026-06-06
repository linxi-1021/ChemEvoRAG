"""Tests for output isolation in regression validation.

Ensures that --validate-only and individual/composition regression
never overwrite the baseline data/eval/eval_results.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_project(tmp_path: Path, baseline_total: int = 118) -> Path:
    """Create a minimal fake project with baseline eval_results.json."""
    project = tmp_path / "project"
    (project / "data" / "eval").mkdir(parents=True)
    (project / "data" / "eval" / "eval_results.json").write_text(
        json.dumps({
            "summary": {"total": baseline_total, "average_score": 0.87, "by_intent": {}},
            "results": [{"id": f"q{i}", "score": 0.9} for i in range(baseline_total)],
        }),
        encoding="utf-8",
    )
    # Minimal scripts dir
    (project / "scripts").mkdir(parents=True)
    (project / "scripts" / "eval_questions.py").write_text("# stub\n", encoding="utf-8")
    return project


def _make_fake_run_evaluation():
    """Return a fake run_evaluation that writes eval_results.json to output_dir."""
    def _fake_run(**kwargs):
        out_dir = kwargs.get("output_dir")
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "eval_results.json").write_text(
                json.dumps({
                    "summary": {"total": 2, "average_score": 0.85, "by_intent": {}},
                    "results": [{"id": "regression_q1", "score": 0.8}, {"id": "regression_q2", "score": 0.9}],
                }),
                encoding="utf-8",
            )
        return 0
    return _fake_run


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRegressionRunnerOutputIsolation:
    """RegressionRunner.run_eval must not write to baseline data/eval."""

    def test_run_eval_uses_passed_output_dir(self, tmp_path):
        project = _make_fake_project(tmp_path)
        regression_output = tmp_path / "regression_output"
        regression_output.mkdir()

        from skill_evolution.regression import RegressionRunner
        runner = RegressionRunner(project)

        with patch("eval_questions.run_evaluation", _make_fake_run_evaluation()):
            result = runner.run_eval(
                dataset_path=project / "data" / "eval" / "eval_questions.json",
                output_dir=regression_output,
                stream_output=False,
            )

        assert result.total_questions == 2
        assert result.average_score == 0.85

    def test_run_eval_rejects_none_output_dir(self, tmp_path):
        project = _make_fake_project(tmp_path)

        from skill_evolution.regression import RegressionRunner
        runner = RegressionRunner(project)

        with pytest.raises(ValueError, match="output_dir is required"):
            runner.run_eval(
                dataset_path=project / "data" / "eval" / "eval_questions.json",
                output_dir=None,
            )

    def test_run_eval_rejects_baseline_output_dir(self, tmp_path):
        project = _make_fake_project(tmp_path)

        from skill_evolution.regression import RegressionRunner
        runner = RegressionRunner(project)

        with pytest.raises(ValueError, match="cannot write to baseline"):
            runner.run_eval(
                dataset_path=project / "data" / "eval" / "eval_questions.json",
                output_dir=project / "data" / "eval",
            )

    def test_run_eval_does_not_corrupt_baseline(self, tmp_path):
        project = _make_fake_project(tmp_path, baseline_total=118)
        regression_output = tmp_path / "regression_output"
        regression_output.mkdir()

        from skill_evolution.regression import RegressionRunner
        runner = RegressionRunner(project)

        with patch("eval_questions.run_evaluation", _make_fake_run_evaluation()):
            runner.run_eval(
                dataset_path=project / "data" / "eval" / "eval_questions.json",
                output_dir=regression_output,
                stream_output=False,
            )

        # Baseline must still be 118
        baseline = json.loads((project / "data" / "eval" / "eval_results.json").read_text("utf-8"))
        assert baseline["summary"]["total"] == 118
        assert len(baseline["results"]) == 118


class TestValidateIndividualPatchOutputIsolation:
    """validate_individual_patch must not overwrite baseline."""

    def test_individual_patch_regression_writes_to_temp(self, tmp_path):
        project = _make_fake_project(tmp_path, baseline_total=118)
        skills_dir = project / "config" / "skills"
        prompts_dir = project / "config" / "prompts"
        skills_dir.mkdir(parents=True)
        prompts_dir.mkdir(parents=True)

        # Write a minimal skill YAML
        import yaml
        skill_data = {
            "name": "property_query",
            "trigger": {"intent": "property_query"},
            "strategy": {
                "assessment": {"system_prompt_ref": "TEST"},
                "answer_generation": {"system_prompt_ref": "TEST"},
            },
        }
        (skills_dir / "property_query.yaml").write_text(yaml.dump(skill_data), encoding="utf-8")

        # Write baseline eval_results.json as dict
        baseline_result = {
            "average_score": 0.87,
            "total_questions": 118,
            "by_intent": {"property_query": {"average_score": 0.86, "total": 23, "failed": 0}},
        }
        dataset = project / "data" / "eval" / "all_questions.json"
        dataset.write_text("[]", encoding="utf-8")

        # Create a simple patch
        from skill_evolution.patch import PatchSchema, PatchOperation
        test_patch = PatchSchema(
            patch_id="test_patch_1",
            skill_name="property_query",
            skill_version="1.0.0",
            source_failure_ids=["q1"],
            primary_failure_type="assessment_false_negative",
            target_path="strategy.assessment",
            operation=PatchOperation.UPDATE,
            current_value={"system_prompt_ref": "OLD"},
            proposed_value={"system_prompt_ref": "NEW"},
            rationale="test",
            risk_level="low",
            confidence=0.9,
        )

        # Mock run_eval to write to a temp dir, not data/eval
        def _mock_run_eval(self, dataset_path, skills_dir=None, prompts_dir=None, output_dir=None, **kwargs):
            from skill_evolution.regression import RegressionResult
            # Verify output_dir is NOT the baseline
            if output_dir and Path(output_dir).resolve() == (project / "data" / "eval").resolve():
                raise AssertionError("run_eval called with baseline output_dir!")
            if output_dir is None:
                raise AssertionError("run_eval called without output_dir!")
            return RegressionResult(average_score=0.86, total_questions=118, by_intent={})

        from skill_evolution.runtime_validation import validate_individual_patch
        with patch(
            "skill_evolution.regression.RegressionRunner.run_eval",
            _mock_run_eval,
        ):
            result = validate_individual_patch(
                test_patch, skill_data,
                project_root=project,
                skills_dir=skills_dir,
                prompts_dir=prompts_dir,
                regression_dataset=dataset,
                baseline_result=baseline_result,
                skill_filename="property_query.yaml",
            )

        # Baseline must still be 118
        baseline = json.loads((project / "data" / "eval" / "eval_results.json").read_text("utf-8"))
        assert baseline["summary"]["total"] == 118
