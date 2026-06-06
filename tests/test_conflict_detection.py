"""Tests for detect_conflicts and _get_effective_patch_path."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _make_patch(
    patch_id: str,
    target_path: str,
    proposed_value: dict,
    operation: str = "add",
    skill_name: str = "test_skill",
):
    """Create a minimal PatchSchema for testing."""
    from skill_evolution.patch import PatchSchema
    return PatchSchema(
        patch_id=patch_id,
        skill_name=skill_name,
        skill_version="1.0.0",
        source_failure_ids=[],
        primary_failure_type="unknown_failure",
        target_path=target_path,
        operation=operation,
        current_value={},
        proposed_value=proposed_value,
        rationale="test",
        expected_improvement="test",
        risk_level="low",
        confidence=0.8,
        status="candidate",
    )


class TestEffectivePatchPath:
    """Test _get_effective_patch_path helper."""

    def test_template_add_with_name(self):
        from skill_evolution.runtime_validation import _get_effective_patch_path
        patch = _make_patch("p1", "templates", {"name": "my_template", "content": "..."})
        assert _get_effective_patch_path(patch) == "templates.my_template"

    def test_template_add_different_names_different_paths(self):
        from skill_evolution.runtime_validation import _get_effective_patch_path
        p1 = _make_patch("p1", "templates", {"name": "template_a", "content": "..."})
        p2 = _make_patch("p2", "templates", {"name": "template_b", "content": "..."})
        assert _get_effective_patch_path(p1) != _get_effective_patch_path(p2)

    def test_template_add_no_name_uses_target(self):
        from skill_evolution.runtime_validation import _get_effective_patch_path
        patch = _make_patch("p1", "templates", {"content": "..."})
        assert _get_effective_patch_path(patch) == "templates"

    def test_update_expands_to_field(self):
        from skill_evolution.runtime_validation import _get_effective_patch_path
        patch = _make_patch(
            "p1", "strategy.assessment",
            {"system_prompt_ref": "NEW_PROMPT"},
            operation="update",
        )
        assert _get_effective_patch_path(patch) == "strategy.assessment.system_prompt_ref"


class TestDetectConflicts:
    """Test detect_conflicts."""

    def test_template_add_different_names_no_conflict(self):
        from skill_evolution.runtime_validation import detect_conflicts
        p1 = _make_patch("p1", "templates", {"name": "template_a", "content": "A"})
        p2 = _make_patch("p2", "templates", {"name": "template_b", "content": "B"})
        conflicts = detect_conflicts([p1, p2])
        assert conflicts == []

    def test_template_add_same_name_conflict(self):
        from skill_evolution.runtime_validation import detect_conflicts
        p1 = _make_patch("p1", "templates", {"name": "template_a", "content": "A"})
        p2 = _make_patch("p2", "templates", {"name": "template_a", "content": "B"})
        conflicts = detect_conflicts([p1, p2])
        assert len(conflicts) == 1
        assert conflicts[0][2] == "templates.template_a"

    def test_update_same_path_conflict(self):
        from skill_evolution.runtime_validation import detect_conflicts
        p1 = _make_patch(
            "p1", "strategy.assessment",
            {"system_prompt_ref": "PROMPT_A"}, operation="update",
        )
        p2 = _make_patch(
            "p2", "strategy.assessment",
            {"system_prompt_ref": "PROMPT_B"}, operation="update",
        )
        conflicts = detect_conflicts([p1, p2])
        assert len(conflicts) == 1

    def test_update_different_paths_no_conflict(self):
        from skill_evolution.runtime_validation import detect_conflicts
        p1 = _make_patch(
            "p1", "strategy.assessment",
            {"system_prompt_ref": "PROMPT_A"}, operation="update",
        )
        p2 = _make_patch(
            "p2", "strategy.answer_generation",
            {"system_prompt_ref": "PROMPT_B"}, operation="update",
        )
        conflicts = detect_conflicts([p1, p2])
        assert conflicts == []

    def test_add_no_name_conservative_conflict(self):
        """When proposed_value has no 'name' field, same target_path = conflict."""
        from skill_evolution.runtime_validation import detect_conflicts
        p1 = _make_patch("p1", "templates", {"content": "A"})
        p2 = _make_patch("p2", "templates", {"content": "B"})
        conflicts = detect_conflicts([p1, p2])
        assert len(conflicts) == 1

    def test_mixed_update_and_add_no_conflict(self):
        """An update to strategy.assessment and an add to templates don't conflict."""
        from skill_evolution.runtime_validation import detect_conflicts
        p1 = _make_patch(
            "p1", "strategy.assessment",
            {"system_prompt_ref": "NEW"}, operation="update",
        )
        p2 = _make_patch("p2", "templates", {"name": "new_template", "content": "..."})
        conflicts = detect_conflicts([p1, p2])
        assert conflicts == []

    def test_five_template_adds_different_names_no_conflict(self):
        """Five template adds with different names should all be compatible."""
        from skill_evolution.runtime_validation import detect_conflicts
        patches = [
            _make_patch(f"p{i}", "templates", {"name": f"template_{i}", "content": f"C{i}"})
            for i in range(5)
        ]
        conflicts = detect_conflicts(patches)
        assert conflicts == []
