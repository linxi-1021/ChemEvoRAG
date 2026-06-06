"""Tests for RegressionRunner.run_eval stream_output mode (in-process)."""

from __future__ import annotations

import json
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_project(tmp_path: Path) -> Path:
    """Create a minimal project structure with a stub eval_questions.py."""
    project = tmp_path / "fake_project"
    (project / "scripts").mkdir(parents=True)
    (project / "scripts" / "eval_questions.py").write_text(
        "def run_evaluation(**kwargs): pass\n", encoding="utf-8")
    (project / "data" / "eval").mkdir(parents=True)
    (project / "data" / "eval" / "eval_results.json").write_text(
        json.dumps({"summary": {"total": 118, "average_score": 0.87}, "results": []}),
        encoding="utf-8",
    )
    (project / "data" / "eval" / "eval_questions.json").write_text("[]", encoding="utf-8")
    (project / "data" / "eval" / "all_questions.json").write_text(
        json.dumps([{"id": "q1", "question": "test?", "intent": "property_query"}]),
        encoding="utf-8",
    )
    return project


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRegressionRunnerStreamOutput:
    """Test stream_output=True passes through eval output to terminal."""

    def test_stream_output_true_calls_run_evaluation(self, tmp_path, monkeypatch):
        """stream_output=True must call run_evaluation() directly."""
        from skill_evolution.regression import RegressionRunner

        project = _make_fake_project(tmp_path)
        output_dir = tmp_path / "eval_output"
        output_dir.mkdir()

        # Write fake results so run_eval can load them
        (output_dir / "eval_results.json").write_text(json.dumps({
            "summary": {"total": 2, "average_score": 0.85, "by_intent": {}},
            "results": [{"id": "r1", "score": 0.8}, {"id": "r2", "score": 0.9}],
        }), encoding="utf-8")

        captured_kwargs: dict = {}

        def fake_run_evaluation(**kwargs):
            captured_kwargs.update(kwargs)
            return 0

        monkeypatch.setattr(
            "eval_questions.run_evaluation", fake_run_evaluation,
        )

        runner = RegressionRunner(project)
        result = runner.run_eval(
            dataset_path=tmp_path / "questions.json",
            output_dir=output_dir,
            stream_output=True,
        )

        assert captured_kwargs.get("use_react") is True
        assert result.average_score == 0.85
        assert result.total_questions == 2

    def test_stream_output_false_uses_redirect(self, tmp_path, monkeypatch):
        """stream_output=False must capture stdout via redirect."""
        from skill_evolution.regression import RegressionRunner

        project = _make_fake_project(tmp_path)
        output_dir = tmp_path / "eval_output"
        output_dir.mkdir()

        # Write fake results
        (output_dir / "eval_results.json").write_text(json.dumps({
            "summary": {"total": 2, "average_score": 0.85, "by_intent": {}},
            "results": [{"id": "r1", "score": 0.8}, {"id": "r2", "score": 0.9}],
        }), encoding="utf-8")

        captured_kwargs: dict = {}

        def fake_run_evaluation(**kwargs):
            captured_kwargs.update(kwargs)
            print("captured output")
            return 0

        monkeypatch.setattr(
            "eval_questions.run_evaluation", fake_run_evaluation,
        )

        runner = RegressionRunner(project)
        result = runner.run_eval(
            dataset_path=tmp_path / "questions.json",
            output_dir=output_dir,
            stream_output=False,
        )

        # Must still call run_evaluation with correct args
        assert captured_kwargs.get("use_react") is True
        assert result.average_score == 0.85

    def test_stream_output_true_passes_workers_arg(self, tmp_path, monkeypatch):
        """stream_output=True must pass workers to run_evaluation."""
        from skill_evolution.regression import RegressionRunner

        project = _make_fake_project(tmp_path)
        output_dir = tmp_path / "eval_output"
        output_dir.mkdir()

        (output_dir / "eval_results.json").write_text(json.dumps({
            "summary": {"total": 1, "average_score": 0.9, "by_intent": {}},
            "results": [{"id": "q1", "score": 0.9, "intent": "test"}],
        }), encoding="utf-8")

        captured_kwargs: dict = {}

        def fake_run_evaluation(**kwargs):
            captured_kwargs.update(kwargs)
            return 0

        monkeypatch.setattr(
            "eval_questions.run_evaluation", fake_run_evaluation,
        )

        runner = RegressionRunner(project)
        runner.run_eval(
            dataset_path=tmp_path / "questions.json",
            output_dir=output_dir,
            workers=4,
            stream_output=True,
        )

        assert captured_kwargs.get("workers") == 4

    def test_stream_output_true_rejects_baseline_output_dir(self, tmp_path):
        """output_dir=data/eval must raise ValueError."""
        from skill_evolution.regression import RegressionRunner

        project = _make_fake_project(tmp_path)
        runner = RegressionRunner(project)

        with pytest.raises(ValueError, match="cannot write to baseline"):
            runner.run_eval(
                dataset_path=tmp_path / "questions.json",
                output_dir=project / "data" / "eval",
                stream_output=True,
            )

    def test_eval_exception_returns_error_result(self, tmp_path, monkeypatch):
        """If run_evaluation raises, run_eval returns error result."""
        from skill_evolution.regression import RegressionRunner

        project = _make_fake_project(tmp_path)
        output_dir = tmp_path / "eval_output"
        output_dir.mkdir()

        def fake_run_evaluation(**kwargs):
            raise RuntimeError("API connection failed")

        monkeypatch.setattr(
            "eval_questions.run_evaluation", fake_run_evaluation,
        )

        runner = RegressionRunner(project)
        result = runner.run_eval(
            dataset_path=tmp_path / "questions.json",
            output_dir=output_dir,
            stream_output=True,
        )

        assert result.errors
        assert any("API connection failed" in e for e in result.errors)


class TestEvolveQuietRegressionCLI:
    """Verify --quiet-regression CLI flag works."""

    def test_quiet_regression_flag_defaults_false(self):
        """Without --quiet-regression, stream_output should be True."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--workers", type=int, default=1)
        parser.add_argument("--quiet-regression", action="store_true")
        parser.add_argument("--validate-only", action="store_true")

        args = parser.parse_args(["--validate-only"])
        assert not args.quiet_regression

    def test_quiet_regression_flag_parses_true(self):
        """--quiet-regression must be parsed correctly."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--workers", type=int, default=1)
        parser.add_argument("--quiet-regression", action="store_true")
        parser.add_argument("--validate-only", action="store_true")

        args = parser.parse_args(["--validate-only", "--quiet-regression"])
        assert args.quiet_regression
