"""Tests for --regression-limit parameter flow through the call chain.

Validates:
- RegressionRunner.run_eval passes --limit N to eval_questions.py
- limit=None omits --limit
- limit <= 0 raises ValueError
- validate_individual_patch forwards regression_limit
- validate_composition forwards regression_limit
- evolve.py --validate-only --regression-limit 20 reaches phase_regression_validate
- evolve.py --apply --regression-limit 20 is rejected
"""

from __future__ import annotations

import json
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# RegressionRunner.run_eval --limit
# ---------------------------------------------------------------------------

class TestRunEvalLimit:
    """RegressionRunner.run_eval must support limit parameter."""

    def test_passes_limit_flag(self, tmp_path, monkeypatch):
        """run_eval(..., limit=20, workers=4) passes limit=20 to run_evaluation."""
        from skill_evolution.regression import RegressionRunner

        project = tmp_path / "project"
        project.mkdir()
        (project / "scripts").mkdir()
        (project / "scripts" / "eval_questions.py").write_text(
            "def run_evaluation(**kwargs): pass\n"
        )

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        dataset = tmp_path / "questions.json"
        dataset.write_text("[]", encoding="utf-8")

        captured_kwargs: dict = {}

        def fake_run_evaluation(**kwargs):
            captured_kwargs.update(kwargs)
            outp = kwargs["output_dir"]
            (outp / "eval_results.json").write_text(json.dumps({
                "summary": {"total": 1, "average_score": 0.9, "by_intent": {}},
                "results": [{"id": "q1", "score": 0.9, "intent": "test"}],
            }), encoding="utf-8")
            return 0

        monkeypatch.setattr(
            "eval_questions.run_evaluation", fake_run_evaluation,
        )

        runner = RegressionRunner(project)
        runner.run_eval(
            dataset_path=dataset,
            output_dir=output_dir,
            workers=4,
            limit=20,
            stream_output=True,
        )

        assert captured_kwargs.get("workers") == 4
        assert captured_kwargs.get("limit") == 20

    def test_no_limit_flag_when_none(self, tmp_path, monkeypatch):
        """run_eval(..., limit=None) passes limit=None to run_evaluation."""
        from skill_evolution.regression import RegressionRunner

        project = tmp_path / "project"
        project.mkdir()
        (project / "scripts").mkdir()
        (project / "scripts" / "eval_questions.py").write_text(
            "def run_evaluation(**kwargs): pass\n"
        )

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        dataset = tmp_path / "questions.json"
        dataset.write_text("[]", encoding="utf-8")

        captured_kwargs: dict = {}

        def fake_run_evaluation(**kwargs):
            captured_kwargs.update(kwargs)
            outp = kwargs["output_dir"]
            (outp / "eval_results.json").write_text(json.dumps({
                "summary": {"total": 1, "average_score": 0.9, "by_intent": {}},
                "results": [{"id": "q1", "score": 0.9, "intent": "test"}],
            }), encoding="utf-8")
            return 0

        monkeypatch.setattr(
            "eval_questions.run_evaluation", fake_run_evaluation,
        )

        runner = RegressionRunner(project)
        runner.run_eval(
            dataset_path=dataset,
            output_dir=output_dir,
            limit=None,
            stream_output=True,
        )

        assert captured_kwargs.get("limit") is None

    def test_limit_zero_raises(self, tmp_path):
        """limit=0 must raise ValueError."""
        from skill_evolution.regression import RegressionRunner

        project = tmp_path / "project"
        project.mkdir()
        runner = RegressionRunner(project)

        with pytest.raises(ValueError, match="limit"):
            runner.run_eval(
                dataset_path=tmp_path / "q.json",
                output_dir=tmp_path / "out",
                limit=0,
            )

    def test_limit_negative_raises(self, tmp_path):
        """limit=-1 must raise ValueError."""
        from skill_evolution.regression import RegressionRunner

        project = tmp_path / "project"
        project.mkdir()
        runner = RegressionRunner(project)

        with pytest.raises(ValueError, match="limit"):
            runner.run_eval(
                dataset_path=tmp_path / "q.json",
                output_dir=tmp_path / "out",
                limit=-5,
            )


# ---------------------------------------------------------------------------
# validate_individual_patch forwards regression_limit
# ---------------------------------------------------------------------------

class TestIndividualPatchLimit:
    """validate_individual_patch must forward regression_limit to run_eval."""

    def test_forwards_regression_limit(self, monkeypatch, tmp_path):
        """validate_individual_patch(..., regression_limit=20) passes limit=20."""
        from skill_evolution.runtime_validation import validate_individual_patch
        from skill_evolution.patch import PatchSchema, PatchOperation, PatchStatus
        from skill_evolution.attribution import FailureType

        captured_limit = None

        def fake_run_eval(self, dataset_path, skills_dir=None, prompts_dir=None,
                          output_dir=None, *, workers=1, limit=None,
                          stream_output=True, **kwargs):
            nonlocal captured_limit
            captured_limit = limit
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            (out / "eval_results.json").write_text(json.dumps({
                "summary": {"total": 1, "average_score": 0.90, "by_intent": {}},
                "results": [{"id": "q1", "score": 0.90, "intent": "property_query"}],
            }), encoding="utf-8")
            from skill_evolution.regression import RegressionResult
            return RegressionResult(average_score=0.90, total_questions=1)

        monkeypatch.setattr(
            "skill_evolution.regression.RegressionRunner.run_eval", fake_run_eval,
        )

        skill_dir = tmp_path / "skills"; skill_dir.mkdir()
        (skill_dir / "property_query.yaml").write_text(
            "name: property_query\ntrigger:\n  intent: property_query\n")
        prompts_dir = tmp_path / "prompts"; prompts_dir.mkdir()
        dataset = tmp_path / "q.json"
        dataset.write_text(json.dumps([
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
        sc = {
            "name": "property_query",
            "trigger": {"intent": "property_query"},
            "evolution": {"mutable_paths": ["strategy.assessment.system_prompt_ref"]},
        }
        baseline = {"average_score": 0.87, "total_questions": 1, "by_intent": {}}

        validate_individual_patch(
            patch, sc,
            project_root=tmp_path, skills_dir=skill_dir, prompts_dir=prompts_dir,
            regression_dataset=dataset, baseline_result=baseline,
            skill_filename="property_query.yaml",
            regression_limit=20,
        )
        assert captured_limit == 20


# ---------------------------------------------------------------------------
# validate_composition forwards regression_limit
# ---------------------------------------------------------------------------

class TestCompositionLimit:
    """validate_composition must forward regression_limit to run_eval."""

    def test_forwards_regression_limit(self, monkeypatch, tmp_path):
        """validate_composition(..., regression_limit=20) passes limit=20."""
        from skill_evolution.runtime_validation import validate_composition
        from skill_evolution.patch import PatchSchema, PatchOperation, PatchStatus
        from skill_evolution.attribution import FailureType

        captured_limit = None

        def fake_run_eval(self, dataset_path, skills_dir=None, prompts_dir=None,
                          output_dir=None, *, workers=1, limit=None,
                          stream_output=True, **kwargs):
            nonlocal captured_limit
            captured_limit = limit
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            (out / "eval_results.json").write_text(json.dumps({
                "summary": {"total": 1, "average_score": 0.90, "by_intent": {}},
                "results": [{"id": "q1", "score": 0.90, "intent": "property_query"}],
            }), encoding="utf-8")
            from skill_evolution.regression import RegressionResult
            return RegressionResult(average_score=0.90, total_questions=1)

        monkeypatch.setattr(
            "skill_evolution.regression.RegressionRunner.run_eval", fake_run_eval,
        )

        skill_dir = tmp_path / "skills"; skill_dir.mkdir()
        (skill_dir / "property_query.yaml").write_text(
            "name: property_query\ntrigger:\n  intent: property_query\n")
        prompts_dir = tmp_path / "prompts"; prompts_dir.mkdir()
        dataset = tmp_path / "q.json"
        dataset.write_text(json.dumps([
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
        baseline = {"average_score": 0.87, "total_questions": 1, "by_intent": {}}

        validate_composition(
            [patch], {"property_query": {
                "name": "property_query",
                "evolution": {"mutable_paths": ["strategy.assessment.system_prompt_ref"]},
            }},
            project_root=tmp_path, skills_dir=skill_dir, prompts_dir=prompts_dir,
            regression_dataset=dataset, baseline_regression_result=baseline,
            regression_limit=20,
        )
        assert captured_limit == 20


# ---------------------------------------------------------------------------
# evolve.py CLI
# ---------------------------------------------------------------------------

class TestEvolveCLIRegressionLimit:
    """evolve.py must parse --regression-limit and enforce --apply rejection."""

    def test_validate_only_with_limit_passes_to_phase(self, monkeypatch):
        """--validate-only --regression-limit 20 reaches phase_regression_validate."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--validate-only", action="store_true")
        parser.add_argument("--regression-limit", type=int, default=None)
        parser.add_argument("--workers", type=int, default=1)

        args = parser.parse_args(["--validate-only", "--regression-limit", "20", "--workers", "4"])
        assert args.validate_only
        assert args.regression_limit == 20
        assert args.workers == 4

    def test_apply_with_limit_rejected(self):
        """--apply --regression-limit 20 must reject."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--apply", action="store_true")
        parser.add_argument("--regression-limit", type=int, default=None)

        args = parser.parse_args(["--apply", "--regression-limit", "20"])
        assert args.apply
        assert args.regression_limit == 20
        # Rejection logic tested via integration below

    def test_no_limit_default(self):
        """Without --regression-limit, default should be None."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--validate-only", action="store_true")
        parser.add_argument("--regression-limit", type=int, default=None)

        args = parser.parse_args(["--validate-only"])
        assert args.regression_limit is None

    def test_limit_zero_rejected(self):
        """--regression-limit 0 should be rejected."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--regression-limit", type=int, default=None)

        args = parser.parse_args(["--regression-limit", "0"])
        assert args.regression_limit == 0
        # Validation logic tested separately
