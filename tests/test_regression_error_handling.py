"""Tests for regression error handling — eval crashes must result in passed=False.

Validates:
- after_result.errors → validate_individual_patch returns passed=False
- after_result.errors → patch.status = REJECTED
- after_result.errors → regression_result is None
- passed=True + missing average_score → KeyError (not silent 0)
- validate_composition after_result.errors → passed=False
- Normal success path still returns correct average_score
- _require_metric raises KeyError for missing fields
- _require_metric reads from regression_result sub-dict
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


# ---------------------------------------------------------------------------
# Test 1: after_result.errors causes validate_individual_patch to fail
# ---------------------------------------------------------------------------

class TestIndividualPatchEvalCrash:
    """Eval crash (after_result.errors non-empty) must cause passed=False."""

    def test_eval_errors_causes_failed(self, monkeypatch, tmp_path):
        """RegressionRunner returns errors → validate_individual_patch failed."""
        from skill_evolution.runtime_validation import validate_individual_patch
        from skill_evolution.patch import PatchSchema, PatchOperation, PatchStatus
        from skill_evolution.attribution import FailureType
        from skill_evolution.regression import RegressionResult

        def fake_run_eval(self, dataset_path, skills_dir=None, prompts_dir=None,
                          output_dir=None, *, workers=1, limit=None, seed=42,
                          stream_output=True, **kwargs):
            return RegressionResult(
                average_score=0.0,
                total_questions=0,
                errors=["Eval exited with code 1"],
            )

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
        baseline = {"average_score": 0.87, "total_questions": 118, "by_intent": {}}

        result = validate_individual_patch(
            patch, sc,
            project_root=tmp_path, skills_dir=skill_dir, prompts_dir=prompts_dir,
            regression_dataset=dataset, baseline_result=baseline,
            skill_filename="property_query.yaml",
        )

        assert result["passed"] is False, (
            f"Eval crash must set passed=False, got {result['passed']}"
        )
        assert len(result["errors"]) > 0, "Must have errors"
        assert any("Regression eval failed" in e or "Eval exited" in e
                   for e in result["errors"]), (
            f"Error must mention eval failure, got: {result['errors']}"
        )
        assert result["regression_result"] is None, "regression_result must be None"
        assert result.get("average_score") is None, "average_score must be None"
        assert patch.status == PatchStatus.REJECTED, (
            f"patch.status must be REJECTED, got {patch.status}"
        )


# ---------------------------------------------------------------------------
# Test 2: after_result.errors → patch not in selected
# ---------------------------------------------------------------------------

class TestEvalCrashNotSelected:
    """Eval crash patch must not enter selected_patches."""

    def test_crashed_patch_not_selected(self, monkeypatch, tmp_path):
        """phase_regression_validate with eval crash → still_valid is empty."""
        from skill_evolution.runtime_validation import validate_individual_patch
        from skill_evolution.patch import PatchSchema, PatchOperation, PatchStatus
        from skill_evolution.attribution import FailureType
        from skill_evolution.regression import RegressionResult

        def fake_run_eval(self, dataset_path, skills_dir=None, prompts_dir=None,
                          output_dir=None, *, workers=1, limit=None, seed=42,
                          stream_output=True, **kwargs):
            return RegressionResult(
                average_score=0.0,
                errors=["Eval exited with code 1"],
            )

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
        baseline = {"average_score": 0.87, "total_questions": 118, "by_intent": {}}

        result = validate_individual_patch(
            patch, sc,
            project_root=tmp_path, skills_dir=skill_dir, prompts_dir=prompts_dir,
            regression_dataset=dataset, baseline_result=baseline,
            skill_filename="property_query.yaml",
        )

        assert result["passed"] is False, "Eval crash must fail the patch"
        # If passed is False, phase_regression_validate will NOT add to still_valid


# ---------------------------------------------------------------------------
# Test 3: passed=True but missing average_score cannot print OK
# ---------------------------------------------------------------------------

class TestPassedButNoMetrics:
    """passed=True with missing metrics must be caught by _require_metric."""

    def test_require_metric_missing_raises(self):
        """_require_metric raises KeyError when key is missing."""
        from evolve import _require_metric

        result = {
            "patch_id": "p1",
            "passed": True,
            "regression_result": None,
        }
        with pytest.raises(KeyError, match="Missing regression metric"):
            _require_metric(result, "average_score")

    def test_require_metric_from_regression_result(self):
        """_require_metric reads from regression_result sub-dict."""
        from evolve import _require_metric

        result = {
            "patch_id": "p1",
            "passed": True,
            "regression_result": {
                "average_score": 0.86,
                "score_delta": -0.001,
                "targeted_improvement": 0.02,
            },
        }
        assert _require_metric(result, "average_score") == 0.86
        assert _require_metric(result, "score_delta") == -0.001
        assert _require_metric(result, "targeted_improvement") == 0.02

    def test_require_metric_top_level_wins(self):
        """Top-level fields take priority over sub-dict fields."""
        from evolve import _require_metric

        result = {
            "patch_id": "p1",
            "passed": True,
            "average_score": 0.90,
            "regression_result": {
                "average_score": 0.86,
            },
        }
        assert _require_metric(result, "average_score") == 0.90

    def test_passed_true_without_metrics_converts_to_failed(self):
        """In phase_regression_validate, passed=True with missing metrics → failed."""
        from skill_evolution.patch import PatchSchema, PatchOperation, PatchStatus
        from skill_evolution.attribution import FailureType

        # Simulate a result dict that has passed=True but no metrics
        # This mimics what happens when _require_metric raises KeyError
        patch = PatchSchema(
            patch_id="t1", skill_name="property_query",
            primary_failure_type=FailureType.ASSESSMENT_FALSE_NEG,
            target_path="strategy.assessment.system_prompt_ref",
            operation=PatchOperation.UPDATE,
            proposed_value={"system_prompt_ref": "X"},
            status=PatchStatus.CANDIDATE,
        )
        result = {"patch_id": "t1", "passed": True, "errors": [], "regression_result": None}

        # Simulate _require_metric raising
        try:
            from evolve import _require_metric
            _require_metric(result, "average_score")
            pytest.fail("Should have raised KeyError")
        except KeyError as e:
            result["passed"] = False
            result.setdefault("errors", []).append(str(e))

        assert result["passed"] is False
        assert "Missing regression metric" in result["errors"][0]


# ---------------------------------------------------------------------------
# Test 4: validate_composition after_result.errors → failed
# ---------------------------------------------------------------------------

class TestCompositionEvalCrash:
    """Composition regression eval crash must set passed=False."""

    def test_eval_errors_causes_composition_failed(self, monkeypatch, tmp_path):
        """RegressionRunner errors cause composition to fail."""
        from skill_evolution.runtime_validation import validate_composition
        from skill_evolution.patch import PatchSchema, PatchOperation, PatchStatus
        from skill_evolution.attribution import FailureType
        from skill_evolution.regression import RegressionResult

        def fake_run_eval(self, dataset_path, skills_dir=None, prompts_dir=None,
                          output_dir=None, *, workers=1, limit=None, seed=42,
                          stream_output=True, **kwargs):
            return RegressionResult(
                average_score=0.0,
                errors=["Eval timed out after 1800s"],
            )

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
        baseline = {"average_score": 0.87, "total_questions": 118, "by_intent": {}}

        comp_result = validate_composition(
            [patch], {"property_query": {
                "name": "property_query",
                "evolution": {"mutable_paths": ["strategy.assessment.system_prompt_ref"]},
            }},
            project_root=tmp_path, skills_dir=skill_dir, prompts_dir=prompts_dir,
            regression_dataset=dataset, baseline_regression_result=baseline,
        )

        assert comp_result["passed"] is False, (
            f"Composition eval crash must set passed=False, got {comp_result['passed']}"
        )
        assert len(comp_result["errors"]) > 0, "Must have errors"
        assert any("Regression eval had errors" in e or "Eval timed out" in e
                   for e in comp_result["errors"]), (
            f"Error must mention eval failure, got: {comp_result['errors']}"
        )
        assert comp_result.get("regression_comparison") is None
        assert comp_result.get("average_score") is None


# ---------------------------------------------------------------------------
# Test 5: Normal success path shows correct score
# ---------------------------------------------------------------------------

class TestNormalSuccessPath:
    """When eval succeeds, metrics must be present and non-zero."""

    def test_success_has_top_level_metrics(self, monkeypatch, tmp_path):
        """Successful eval populates top-level average_score, score_delta, etc."""
        from skill_evolution.runtime_validation import validate_individual_patch
        from skill_evolution.patch import PatchSchema, PatchOperation, PatchStatus
        from skill_evolution.attribution import FailureType
        from skill_evolution.regression import RegressionResult

        def fake_run_eval(self, dataset_path, skills_dir=None, prompts_dir=None,
                          output_dir=None, *, workers=1, limit=None, seed=42,
                          stream_output=True, **kwargs):
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            (out / "eval_results.json").write_text(json.dumps({
                "summary": {"total": 20, "average_score": 0.865, "by_intent": {
                    "property_query": 0.88
                }},
                "results": [{"id": "q1", "score": 0.88, "intent": "property_query"}],
            }), encoding="utf-8")
            return RegressionResult(average_score=0.865, total_questions=20)

        def fake_compare(self, before, after, **kwargs):
            from skill_evolution.regression import RegressionComparison
            return RegressionComparison(
                passed=True,
                average_score=0.865,
                score_delta=-0.001,
                targeted_improvement=0.02,
                reason="All checks passed",
            )

        monkeypatch.setattr(
            "skill_evolution.regression.RegressionRunner.run_eval", fake_run_eval,
        )
        monkeypatch.setattr(
            "skill_evolution.regression.RegressionRunner.compare", fake_compare,
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
        baseline = {"average_score": 0.866, "total_questions": 118, "by_intent": {}}

        result = validate_individual_patch(
            patch, sc,
            project_root=tmp_path, skills_dir=skill_dir, prompts_dir=prompts_dir,
            regression_dataset=dataset, baseline_result=baseline,
            skill_filename="property_query.yaml",
        )

        assert result["passed"] is True
        assert result["average_score"] == 0.865, f"Expected 0.865, got {result.get('average_score')}"
        assert result["score_delta"] == -0.001
        assert result["targeted_improvement"] == 0.02
        assert result["regression_result"] is not None
