"""Tests for skill_evolution.rollback module."""
import pytest
from pathlib import Path
from skill_evolution.rollback import SnapshotManager, SnapshotError


class TestSnapshotManager:
    def test_snapshot_before_copies_files(self, tmp_path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        (skills_dir / "test.yaml").write_text("original content", encoding="utf-8")
        (prompts_dir / "prompt.yaml").write_text("original prompt", encoding="utf-8")

        mgr = SnapshotManager(run_dir, skills_dir, prompts_dir)
        mgr.snapshot_before()

        assert (run_dir / "skill_snapshot_before" / "test.yaml").read_text("utf-8") == "original content"
        assert (run_dir / "prompt_snapshot_before" / "prompt.yaml").read_text("utf-8") == "original prompt"

    def test_snapshot_after_copies_files(self, tmp_path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        (skills_dir / "test.yaml").write_text("modified content", encoding="utf-8")

        mgr = SnapshotManager(run_dir, skills_dir, prompts_dir)
        mgr.snapshot_after()

        assert (run_dir / "skill_snapshot_after" / "test.yaml").read_text("utf-8") == "modified content"

    def test_rollback_restores_old_content(self, tmp_path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        # Write original
        (skills_dir / "test.yaml").write_text("original", encoding="utf-8")
        (prompts_dir / "p.yaml").write_text("original prompt", encoding="utf-8")

        mgr = SnapshotManager(run_dir, skills_dir, prompts_dir)
        mgr.snapshot_before()

        # Modify
        (skills_dir / "test.yaml").write_text("modified", encoding="utf-8")
        (prompts_dir / "p.yaml").write_text("modified prompt", encoding="utf-8")

        # Rollback
        mgr.rollback()

        assert (skills_dir / "test.yaml").read_text("utf-8") == "original"
        assert (prompts_dir / "p.yaml").read_text("utf-8") == "original prompt"

    def test_rollback_fails_without_snapshot(self, tmp_path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        mgr = SnapshotManager(run_dir, skills_dir, prompts_dir)
        with pytest.raises(SnapshotError):
            mgr.rollback()

    def test_has_before_snapshot(self, tmp_path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        mgr = SnapshotManager(run_dir, skills_dir, prompts_dir)
        assert mgr.has_before_snapshot() is False

        mgr.snapshot_before()
        assert mgr.has_before_snapshot() is True
