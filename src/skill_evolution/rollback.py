"""Snapshot and Rollback Manager for ChemEvoRAG Skill Evolution.

Saves snapshots of config/skills and config/prompts before and after
patch application, and supports rollback to the before state.

Usage:
    from skill_evolution.rollback import SnapshotManager
    mgr = SnapshotManager(run_dir, skills_dir, prompts_dir)
    mgr.snapshot_before()
    # ... apply patches ...
    mgr.snapshot_after()
    # ... eval shows regression ...
    mgr.rollback()
"""

from __future__ import annotations

import shutil
from pathlib import Path


class SnapshotError(Exception):
    """Raised when snapshot or rollback operations fail."""
    pass


class SnapshotManager:
    """Manages before/after snapshots of skill and prompt configs."""

    def __init__(
        self,
        run_dir: Path,
        skills_dir: Path,
        prompts_dir: Path,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.skills_dir = Path(skills_dir)
        self.prompts_dir = Path(prompts_dir)

    def _copy_dir(self, src: Path, dst: Path) -> None:
        """Copy directory contents, replacing dst if it exists."""
        if dst.exists():
            shutil.rmtree(dst)
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            dst.mkdir(parents=True, exist_ok=True)

    def snapshot_before(self) -> None:
        """Save current skill and prompt configs as 'before' snapshot."""
        self._copy_dir(
            self.skills_dir,
            self.run_dir / "skill_snapshot_before",
        )
        self._copy_dir(
            self.prompts_dir,
            self.run_dir / "prompt_snapshot_before",
        )

    def snapshot_after(self) -> None:
        """Save current skill and prompt configs as 'after' snapshot."""
        self._copy_dir(
            self.skills_dir,
            self.run_dir / "skill_snapshot_after",
        )
        self._copy_dir(
            self.prompts_dir,
            self.run_dir / "prompt_snapshot_after",
        )

    def rollback(self) -> None:
        """Restore skill and prompt configs from the 'before' snapshot.

        Raises SnapshotError if the before snapshot doesn't exist.
        Does NOT depend on git.
        """
        before_skills = self.run_dir / "skill_snapshot_before"
        before_prompts = self.run_dir / "prompt_snapshot_before"

        if not before_skills.is_dir() and not before_prompts.is_dir():
            raise SnapshotError(
                f"No before snapshot found in {self.run_dir}. "
                f"Cannot rollback without a prior snapshot."
            )

        # Restore skills
        if before_skills.is_dir():
            if self.skills_dir.exists():
                shutil.rmtree(self.skills_dir)
            shutil.copytree(before_skills, self.skills_dir)

        # Restore prompts
        if before_prompts.is_dir():
            if self.prompts_dir.exists():
                shutil.rmtree(self.prompts_dir)
            shutil.copytree(before_prompts, self.prompts_dir)

    def has_before_snapshot(self) -> bool:
        """Check whether a before snapshot exists."""
        return (
            (self.run_dir / "skill_snapshot_before").is_dir()
            or (self.run_dir / "prompt_snapshot_before").is_dir()
        )

    def has_after_snapshot(self) -> bool:
        """Check whether an after snapshot exists."""
        return (
            (self.run_dir / "skill_snapshot_after").is_dir()
            or (self.run_dir / "prompt_snapshot_after").is_dir()
        )
