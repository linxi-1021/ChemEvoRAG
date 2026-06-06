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
        assert "templates.template_a" in conflicts[0][2]  # format: "skill/path"

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


# ---------------------------------------------------------------------------
# NEW: Cross-skill conflict tests (the core fix)
# ---------------------------------------------------------------------------

class TestCrossSkillConflicts:
    """Patches targeting the same field path in DIFFERENT skills are NOT conflicts."""

    def test_same_path_different_skill_no_conflict(self):
        """strategy.assessment in reaction_condition_query vs reaction_comparison → no conflict."""
        from skill_evolution.runtime_validation import detect_conflicts
        p1 = _make_patch(
            "reaction_condition_query_round1_patch1",
            "strategy.assessment",
            {"system_prompt_ref": "NEW_V2"},
            operation="update",
            skill_name="reaction_condition_query",
        )
        p2 = _make_patch(
            "reaction_comparison_round1_patch1",
            "strategy.assessment",
            {"system_prompt_ref": "NEW_V2"},
            operation="update",
            skill_name="reaction_comparison",
        )
        conflicts = detect_conflicts([p1, p2])
        assert conflicts == [], (
            f"Different skills should NOT conflict on same path, got: {conflicts}"
        )

    def test_same_path_same_skill_conflict(self):
        """Same path + same skill → conflict."""
        from skill_evolution.runtime_validation import detect_conflicts
        p1 = _make_patch(
            "p1", "strategy.assessment",
            {"system_prompt_ref": "A"}, operation="update",
            skill_name="property_query",
        )
        p2 = _make_patch(
            "p2", "strategy.assessment",
            {"system_prompt_ref": "B"}, operation="update",
            skill_name="property_query",
        )
        conflicts = detect_conflicts([p1, p2])
        assert len(conflicts) == 1

    def test_different_path_same_skill_no_conflict(self):
        """Different paths in same skill → no conflict."""
        from skill_evolution.runtime_validation import detect_conflicts
        p1 = _make_patch(
            "p1", "strategy.assessment",
            {"system_prompt_ref": "A"}, operation="update",
            skill_name="property_query",
        )
        p2 = _make_patch(
            "p2", "strategy.answer_generation",
            {"max_evidence_items": 20}, operation="update",
            skill_name="property_query",
        )
        conflicts = detect_conflicts([p1, p2])
        assert conflicts == []

    def test_real_world_scenario_7_patches(self):
        """7 patches across 5 skills with some shared paths → only true conflicts detected."""
        from skill_evolution.runtime_validation import detect_conflicts
        patches = [
            _make_patch("p1", "strategy.assessment", {"system_prompt_ref": "V2"},
                        operation="update", skill_name="reaction_condition_query"),
            _make_patch("p2", "strategy.answer_generation", {"max_evidence_items": 20},
                        operation="update", skill_name="reaction_condition_query"),
            _make_patch("p3", "strategy.assessment", {"system_prompt_ref": "V2"},
                        operation="update", skill_name="reaction_comparison"),
            _make_patch("p4", "strategy.answer_generation", {"max_evidence_items": 20},
                        operation="update", skill_name="reaction_comparison"),
            _make_patch("p5", "templates", {"name": "entity_template", "slots": ["a"]},
                        operation="add", skill_name="entity_lookup"),
            _make_patch("p6", "templates", {"name": "property_template", "slots": ["b"]},
                        operation="add", skill_name="property_query"),
            _make_patch("p7", "templates", {"name": "alias_template", "slots": ["c"]},
                        operation="add", skill_name="alias_resolution"),
        ]
        conflicts = detect_conflicts(patches)
        assert conflicts == [], (
            f"7 patches across different skills should have 0 conflicts, got: {conflicts}"
        )


# ---------------------------------------------------------------------------
# resolve_conflicts tests
# ---------------------------------------------------------------------------

class TestResolveConflicts:
    """Test the conflict auto-resolution logic."""

    def test_no_conflicts_returns_all(self):
        from skill_evolution.runtime_validation import resolve_conflicts
        p1 = _make_patch("p1", "strategy.assessment", {"system_prompt_ref": "A"},
                         operation="update", skill_name="skill_a")
        p2 = _make_patch("p2", "strategy.assessment", {"system_prompt_ref": "B"},
                         operation="update", skill_name="skill_b")
        kept, rejected = resolve_conflicts([p1, p2])
        assert len(kept) == 2
        assert len(rejected) == 0

    def test_conflict_keeps_highest_score(self):
        from skill_evolution.runtime_validation import resolve_conflicts
        p1 = _make_patch("p1", "strategy.assessment", {"system_prompt_ref": "A"},
                         operation="update", skill_name="property_query")
        p2 = _make_patch("p2", "strategy.assessment", {"system_prompt_ref": "B"},
                         operation="update", skill_name="property_query")
        individual_results = [
            {"patch_id": "p1", "average_score": 0.85},
            {"patch_id": "p2", "average_score": 0.92},
        ]
        kept, rejected = resolve_conflicts([p1, p2], individual_results=individual_results)
        assert len(kept) == 1
        assert kept[0].patch_id == "p2"  # higher score kept
        assert len(rejected) == 1
        assert rejected[0]["patch_id"] == "p1"

    def test_conflict_fallback_to_confidence(self):
        """When no individual_results, fallback to confidence."""
        from skill_evolution.runtime_validation import resolve_conflicts
        p1 = _make_patch("p1", "strategy.assessment", {"system_prompt_ref": "A"},
                         operation="update", skill_name="property_query")
        p1.confidence = 0.7
        p2 = _make_patch("p2", "strategy.assessment", {"system_prompt_ref": "B"},
                         operation="update", skill_name="property_query")
        p2.confidence = 0.9
        kept, rejected = resolve_conflicts([p1, p2])
        assert len(kept) == 1
        assert kept[0].patch_id == "p2"

    def test_non_conflicting_always_kept(self):
        from skill_evolution.runtime_validation import resolve_conflicts
        p1 = _make_patch("p1", "strategy.assessment", {"system_prompt_ref": "A"},
                         operation="update", skill_name="skill_a")
        p2 = _make_patch("p2", "strategy.assessment", {"system_prompt_ref": "A"},
                         operation="update", skill_name="skill_a")  # conflicts with p1
        p3 = _make_patch("p3", "templates", {"name": "t1", "content": "C"},
                         operation="add", skill_name="skill_c")  # no conflict
        individual_results = [
            {"patch_id": "p1", "average_score": 0.90},
            {"patch_id": "p2", "average_score": 0.80},
            {"patch_id": "p3", "average_score": 0.85},
        ]
        kept, rejected = resolve_conflicts([p1, p2, p3], individual_results=individual_results)
        kept_ids = {p.patch_id for p in kept}
        assert "p3" in kept_ids  # always kept — no conflict
        assert "p1" in kept_ids  # higher score kept
        assert "p2" not in kept_ids  # lower score rejected
        assert len(rejected) == 1
