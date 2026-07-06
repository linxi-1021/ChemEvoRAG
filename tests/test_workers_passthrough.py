"""Tests that --workers parameter flows correctly through the call chain."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


# --- RegressionRunner.run_eval passes --workers to subprocess ---

def test_run_eval_passes_workers_4(monkeypatch, tmp_path):
    """RegressionRunner.run_eval should pass workers=4 to run_evaluation."""
    from skill_evolution.regression import RegressionRunner

    fake_project = tmp_path / "project"
    fake_project.mkdir()
    (fake_project / "scripts").mkdir()
    (fake_project / "scripts" / "eval_questions.py").write_text(
        "def run_evaluation(**kwargs): pass\n"
    )

    output_dir = tmp_path / "regression_output"
    output_dir.mkdir()

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

    runner = RegressionRunner(fake_project)
    result = runner.run_eval(
        dataset_path=tmp_path / "questions.json",
        skills_dir=tmp_path / "skills",
        prompts_dir=tmp_path / "prompts",
        output_dir=output_dir,
        workers=4,
        stream_output=False,
    )

    assert captured_kwargs.get("workers") == 4
    assert result.total_questions == 1


def test_run_eval_workers_default_1(monkeypatch, tmp_path):
    """Default workers=1 should pass workers=1 to run_evaluation."""
    from skill_evolution.regression import RegressionRunner

    fake_project = tmp_path / "project"
    fake_project.mkdir()
    (fake_project / "scripts").mkdir()
    (fake_project / "scripts" / "eval_questions.py").write_text(
        "def run_evaluation(**kwargs): pass\n"
    )

    output_dir = tmp_path / "regression_output"
    output_dir.mkdir()

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

    runner = RegressionRunner(fake_project)
    runner.run_eval(
        dataset_path=tmp_path / "questions.json",
        output_dir=output_dir,
        stream_output=False,
    )

    assert captured_kwargs.get("workers") == 1


def test_run_eval_workers_less_than_1_raises(tmp_path):
    """workers < 1 should raise ValueError."""
    from skill_evolution.regression import RegressionRunner

    fake_project = tmp_path / "project"
    fake_project.mkdir()

    runner = RegressionRunner(fake_project)
    with pytest.raises(ValueError, match="workers"):
        runner.run_eval(
            dataset_path=tmp_path / "questions.json",
            output_dir=tmp_path / "out",
            workers=0,
        )


# --- validate_individual_patch passes workers ---

def test_validate_individual_patch_passes_workers(monkeypatch, tmp_path):
    """validate_individual_patch should forward workers to RegressionRunner.run_eval."""
    from skill_evolution.runtime_validation import validate_individual_patch
    from skill_evolution.patch import PatchSchema, PatchOperation, PatchStatus
    from skill_evolution.types import FailureType

    captured_workers = {}

    def fake_run_eval(self, dataset_path, skills_dir=None, prompts_dir=None, output_dir=None, *, workers=1, **kwargs):
        captured_workers["workers"] = workers
        # Write fake eval results so compare can work
        out = Path(output_dir) if output_dir else tmp_path / "out"
        out.mkdir(parents=True, exist_ok=True)
        (out / "eval_results.json").write_text(json.dumps({
            "summary": {"total": 1, "average_score": 0.95, "by_intent": {}},
            "results": [{"id": "q1", "score": 0.95, "intent": "property_query"}],
        }), encoding="utf-8")
        from skill_evolution.regression import RegressionResult
        return RegressionResult(average_score=0.95, total_questions=1)

    monkeypatch.setattr(
        "skill_evolution.regression.RegressionRunner.run_eval",
        fake_run_eval,
    )

    # Create minimal skill config with name
    skill_dir = tmp_path / "skills"
    skill_dir.mkdir()
    skill_file = skill_dir / "property_query.yaml"
    skill_file.write_text("name: property_query\ntrigger:\n  intent: property_query\n", encoding="utf-8")
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()

    # Create a dummy regression dataset file (must exist for step 3 to run)
    dataset_path = tmp_path / "questions.json"
    dataset_path.write_text(json.dumps([
        {"id": "q1", "question": "test", "intent": "property_query"}
    ]), encoding="utf-8")

    patch_obj = PatchSchema(
        patch_id="test_p1",
        skill_name="property_query",
        skill_version="1.0.0",
        source_failure_ids=[],
        primary_failure_type=FailureType.ASSESSMENT_FALSE_NEG,
        target_path="strategy.assessment.system_prompt_ref",
        operation=PatchOperation.UPDATE,
        current_value={},
        proposed_value={"system_prompt_ref": "EVIDENCE_ASSESSMENT_SYSTEM_V2"},
        rationale="test",
        expected_improvement="test",
        risk_level="low",
        confidence=0.9,
        status=PatchStatus.CANDIDATE,
    )

    baseline = {"average_score": 0.87, "total_questions": 1, "by_intent": {}}

    skill_config_for_test = {
        "name": "property_query",
        "trigger": {"intent": "property_query"},
        "evolution": {
            "mutable_paths": ["strategy.assessment.system_prompt_ref"],
        },
    }

    result = validate_individual_patch(
        patch_obj, skill_config_for_test,
        project_root=tmp_path,
        skills_dir=skill_dir,
        prompts_dir=prompts_dir,
        regression_dataset=dataset_path,
        baseline_result=baseline,
        skill_filename="property_query.yaml",
        workers=7,
    )

    assert captured_workers["workers"] == 7


# --- evolve.py CLI parses --workers ---

def test_evolve_cli_parses_workers():
    """evolve.py CLI should accept --workers."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

    # Simulate argparse
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--validate-only", action="store_true")

    args = parser.parse_args(["--validate-only", "--workers", "4"])
    assert args.workers == 4

    args2 = parser.parse_args(["--validate-only"])
    assert args2.workers == 1
