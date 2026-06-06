"""Tests for RegressionRunner CLI parameter passing.

All tests monkeypatch subprocess.run to verify command construction
without actually running eval_questions.py.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


class TestRegressionRunnerCLI:
    """Verify RegressionRunner.run_eval passes correct CLI args to eval_questions.py."""

    def test_passes_skills_dir_and_prompts_dir(self, tmp_path, monkeypatch):
        """When skills_dir and prompts_dir are provided, they should appear
        in the subprocess command."""
        from skill_evolution.regression import RegressionRunner

        captured_cmd = []

        def mock_run(cmd, **kwargs):
            captured_cmd.extend(cmd)
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stderr = ""
            return mock_result

        monkeypatch.setattr("subprocess.run", mock_run)

        # Create fake dataset file so it passes existence check
        dataset = tmp_path / "questions.json"
        dataset.write_text("[]", encoding="utf-8")

        # Create fake skills/prompts dirs
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()

        # Create the eval script so path exists
        eval_script = tmp_path / "scripts" / "eval_questions.py"
        eval_script.parent.mkdir(parents=True)
        eval_script.write_text("# stub", encoding="utf-8")

        runner = RegressionRunner(tmp_path)
        runner.run_eval(
            dataset,
            skills_dir=skills_dir,
            prompts_dir=prompts_dir,
            output_dir=tmp_path / "output",
        )

        cmd_str = " ".join(captured_cmd)
        assert "--skills" in cmd_str
        assert "--skills-dir" in cmd_str
        assert "--prompts-dir" in cmd_str
        assert str(skills_dir) in cmd_str
        assert str(prompts_dir) in cmd_str

    def test_passes_dataset_and_output_dir(self, tmp_path, monkeypatch):
        """When dataset and output_dir are provided, they should appear
        in the subprocess command."""
        from skill_evolution.regression import RegressionRunner

        captured_cmd = []

        def mock_run(cmd, **kwargs):
            captured_cmd.extend(cmd)
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stderr = ""
            return mock_result

        monkeypatch.setattr("subprocess.run", mock_run)

        dataset = tmp_path / "my_questions.json"
        dataset.write_text("[]", encoding="utf-8")

        eval_script = tmp_path / "scripts" / "eval_questions.py"
        eval_script.parent.mkdir(parents=True)
        eval_script.write_text("# stub", encoding="utf-8")

        runner = RegressionRunner(tmp_path)
        runner.run_eval(
            dataset,
            output_dir=tmp_path / "my_output",
        )

        cmd_str = " ".join(captured_cmd)
        assert "--dataset" in cmd_str
        assert str(dataset) in cmd_str
        assert "--output-dir" in cmd_str
        assert str(tmp_path / "my_output") in cmd_str

    def test_uses_default_dirs_when_none(self, tmp_path, monkeypatch):
        """When skills_dir/prompts_dir are None, --skills should not be passed."""
        from skill_evolution.regression import RegressionRunner

        captured_cmd = []

        def mock_run(cmd, **kwargs):
            captured_cmd.extend(cmd)
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stderr = ""
            return mock_result

        monkeypatch.setattr("subprocess.run", mock_run)

        dataset = tmp_path / "questions.json"
        dataset.write_text("[]", encoding="utf-8")

        eval_script = tmp_path / "scripts" / "eval_questions.py"
        eval_script.parent.mkdir(parents=True)
        eval_script.write_text("# stub", encoding="utf-8")

        runner = RegressionRunner(tmp_path)
        runner.run_eval(dataset, output_dir=tmp_path / "output")  # no skills_dir, no prompts_dir

        cmd_str = " ".join(captured_cmd)
        assert "--skills" not in cmd_str
        assert "--skills-dir" not in cmd_str
        assert "--prompts-dir" not in cmd_str
        assert "--dataset" in cmd_str
        assert "--output-dir" in cmd_str

    def test_passes_react_flag(self, tmp_path, monkeypatch):
        """When use_react=True (default), --react should be in the command."""
        from skill_evolution.regression import RegressionRunner

        captured_cmd = []

        def mock_run(cmd, **kwargs):
            captured_cmd.extend(cmd)
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stderr = ""
            return mock_result

        monkeypatch.setattr("subprocess.run", mock_run)

        dataset = tmp_path / "questions.json"
        dataset.write_text("[]", encoding="utf-8")

        eval_script = tmp_path / "scripts" / "eval_questions.py"
        eval_script.parent.mkdir(parents=True)
        eval_script.write_text("# stub", encoding="utf-8")

        runner = RegressionRunner(tmp_path)
        runner.run_eval(dataset, output_dir=tmp_path / "output")

        cmd_str = " ".join(captured_cmd)
        assert "--react" in cmd_str
