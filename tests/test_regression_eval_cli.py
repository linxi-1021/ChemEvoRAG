"""Tests for RegressionRunner parameter passing to run_evaluation.

All tests monkeypatch eval_questions.run_evaluation to verify kwargs
without actually running eval.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _stub_run_evaluation(monkeypatch, output_dir=None):
    """Helper: install a fake run_evaluation that writes results and captures kwargs."""
    captured: dict = {}

    def fake_run_evaluation(**kwargs):
        captured.update(kwargs)
        out = kwargs.get("output_dir") or output_dir
        if out:
            (out / "eval_results.json").write_text(json.dumps({
                "summary": {"total": 1, "average_score": 0.9, "by_intent": {}},
                "results": [{"id": "q1", "score": 0.9, "intent": "test"}],
            }), encoding="utf-8")
        return 0

    monkeypatch.setattr(
        "eval_questions.run_evaluation", fake_run_evaluation,
    )
    return captured


class TestRegressionRunnerCLI:
    """Verify RegressionRunner.run_eval passes correct kwargs to run_evaluation."""

    def test_passes_skills_dir_and_prompts_dir(self, tmp_path, monkeypatch):
        """skills_dir and prompts_dir should be passed to run_evaluation."""
        from skill_evolution.regression import RegressionRunner

        dataset = tmp_path / "questions.json"
        dataset.write_text("[]", encoding="utf-8")

        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()

        eval_script = tmp_path / "scripts" / "eval_questions.py"
        eval_script.parent.mkdir(parents=True)
        eval_script.write_text("def run_evaluation(**kwargs): pass\n", encoding="utf-8")

        output_dir = tmp_path / "output"
        captured = _stub_run_evaluation(monkeypatch, output_dir)

        runner = RegressionRunner(tmp_path)
        runner.run_eval(
            dataset,
            skills_dir=skills_dir,
            prompts_dir=prompts_dir,
            output_dir=output_dir,
            stream_output=False,
        )

        assert captured.get("use_skills") is True
        assert str(captured.get("skills_dir")) == str(skills_dir)
        assert str(captured.get("prompts_dir")) == str(prompts_dir)

    def test_passes_dataset_and_output_dir(self, tmp_path, monkeypatch):
        """dataset and output_dir should be passed to run_evaluation."""
        from skill_evolution.regression import RegressionRunner

        dataset = tmp_path / "my_questions.json"
        dataset.write_text("[]", encoding="utf-8")

        eval_script = tmp_path / "scripts" / "eval_questions.py"
        eval_script.parent.mkdir(parents=True)
        eval_script.write_text("def run_evaluation(**kwargs): pass\n", encoding="utf-8")

        output_dir = tmp_path / "my_output"
        captured = _stub_run_evaluation(monkeypatch, output_dir)

        runner = RegressionRunner(tmp_path)
        runner.run_eval(
            dataset,
            output_dir=output_dir,
            stream_output=False,
        )

        assert str(captured.get("dataset")) == str(dataset)
        assert str(captured.get("output_dir")) == str(output_dir)

    def test_uses_default_dirs_when_none(self, tmp_path, monkeypatch):
        """When skills_dir/prompts_dir are None, use_skills should be False."""
        from skill_evolution.regression import RegressionRunner

        dataset = tmp_path / "questions.json"
        dataset.write_text("[]", encoding="utf-8")

        eval_script = tmp_path / "scripts" / "eval_questions.py"
        eval_script.parent.mkdir(parents=True)
        eval_script.write_text("def run_evaluation(**kwargs): pass\n", encoding="utf-8")

        output_dir = tmp_path / "output"
        captured = _stub_run_evaluation(monkeypatch, output_dir)

        runner = RegressionRunner(tmp_path)
        runner.run_eval(dataset, output_dir=output_dir, stream_output=False)

        assert captured.get("use_skills") is False

    def test_passes_react_flag(self, tmp_path, monkeypatch):
        """use_react=True (default) should be passed to run_evaluation."""
        from skill_evolution.regression import RegressionRunner

        dataset = tmp_path / "questions.json"
        dataset.write_text("[]", encoding="utf-8")

        eval_script = tmp_path / "scripts" / "eval_questions.py"
        eval_script.parent.mkdir(parents=True)
        eval_script.write_text("def run_evaluation(**kwargs): pass\n", encoding="utf-8")

        output_dir = tmp_path / "output"
        captured = _stub_run_evaluation(monkeypatch, output_dir)

        runner = RegressionRunner(tmp_path)
        runner.run_eval(dataset, output_dir=output_dir, stream_output=False)

        assert captured.get("use_react") is True
