"""Tests for skill_evolution.apply module."""
import pytest
from pathlib import Path
from skill_evolution.apply import PatchApplier, PatchApplyError
from skill_evolution.patch import PatchSchema, PatchOperation, FailureType, PatchStatus


def _make_patch(
    target_path: str,
    operation: PatchOperation,
    proposed_value: dict,
    **kwargs,
) -> PatchSchema:
    return PatchSchema(
        skill_name="test_skill",
        primary_failure_type=FailureType.GENERATION_ERROR,
        target_path=target_path,
        operation=operation,
        proposed_value=proposed_value,
        **kwargs,
    )


class TestUpdateOperation:
    def test_update_system_prompt_ref(self, tmp_path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()

        skill = {
            "name": "test_skill",
            "skill_version": "1.0.0",
            "strategy": {"assessment": {"system_prompt_ref": "OLD_PROMPT"}},
        }
        patch = _make_patch(
            "strategy.assessment",
            PatchOperation.UPDATE,
            {"system_prompt_ref": "NEW_PROMPT"},
        )
        applier = PatchApplier(skills_dir, prompts_dir)
        result = applier.apply_to_memory(skill, patch)
        assert result["strategy"]["assessment"]["system_prompt_ref"] == "NEW_PROMPT"

    def test_update_creates_missing_path(self, tmp_path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()

        skill = {"name": "test_skill", "skill_version": "1.0.0"}
        patch = _make_patch(
            "strategy.assessment",
            PatchOperation.UPDATE,
            {"system_prompt_ref": "NEW_PROMPT"},
        )
        applier = PatchApplier(skills_dir, prompts_dir)
        result = applier.apply_to_memory(skill, patch)
        assert result["strategy"]["assessment"]["system_prompt_ref"] == "NEW_PROMPT"


class TestAddOperation:
    def test_add_template(self, tmp_path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()

        skill = {
            "name": "test_skill",
            "skill_version": "1.0.0",
            "templates": [{"name": "existing_template"}],
        }
        patch = _make_patch(
            "templates",
            PatchOperation.ADD,
            {"name": "new_template", "description": "A new template"},
        )
        applier = PatchApplier(skills_dir, prompts_dir)
        result = applier.apply_to_memory(skill, patch)
        assert len(result["templates"]) == 2
        assert result["templates"][1]["name"] == "new_template"

    def test_add_deduplicates_by_name(self, tmp_path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()

        skill = {
            "name": "test_skill",
            "skill_version": "1.0.0",
            "templates": [{"name": "existing"}],
        }
        patch = _make_patch(
            "templates",
            PatchOperation.ADD,
            {"name": "existing", "description": "duplicate"},
        )
        applier = PatchApplier(skills_dir, prompts_dir)
        result = applier.apply_to_memory(skill, patch)
        assert len(result["templates"]) == 1  # no duplicate added


class TestForbiddenOperations:
    def test_delete_raises_error(self, tmp_path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()

        skill = {"name": "test_skill", "skill_version": "1.0.0"}
        patch = _make_patch(
            "strategy.assessment",
            PatchOperation.DELETE,
            {"system_prompt_ref": "TO_DELETE"},
        )
        applier = PatchApplier(skills_dir, prompts_dir)
        with pytest.raises(PatchApplyError, match="DELETE"):
            applier.apply_to_memory(skill, patch)

    def test_replace_raises_error(self, tmp_path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()

        skill = {"name": "test_skill", "skill_version": "1.0.0"}
        patch = _make_patch(
            "strategy.assessment",
            PatchOperation.REPLACE,
            {"system_prompt_ref": "REPLACEMENT"},
        )
        applier = PatchApplier(skills_dir, prompts_dir)
        with pytest.raises(PatchApplyError, match="REPLACE"):
            applier.apply_to_memory(skill, patch)


class TestApplyToDisk:
    def test_writes_yaml_and_updates_version(self, tmp_path):
        import yaml
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()

        skill = {
            "name": "test_skill",
            "skill_version": "1.0.0",
            "trigger": {"intent": "test_skill"},
            "lifecycle": {"updated_at": "2026-01-01"},
            "provenance": {},
            "changelog": [],
            "strategy": {"answer_generation": {"max_evidence_items": 10}},
        }
        # Write initial skill YAML
        (skills_dir / "test_skill.yaml").write_text(
            yaml.dump(skill, default_flow_style=False), encoding="utf-8")

        patch = _make_patch(
            "strategy.answer_generation",
            PatchOperation.UPDATE,
            {"max_evidence_items": 15},
            rationale="Increase evidence limit.",
        )
        applier = PatchApplier(skills_dir, prompts_dir)
        result = applier.apply_to_disk(skill, [patch], round_id="round1")

        assert result["skill_version"] == "1.0.1"
        assert len(result["changelog"]) > 0
        assert result["provenance"]["last_modified_by"] == "skill_evolution_pipeline"
        assert patch.status == PatchStatus.APPLIED

        # Verify file written
        written = yaml.safe_load((skills_dir / "test_skill.yaml").read_text("utf-8"))
        assert written["skill_version"] == "1.0.1"
        assert written["strategy"]["answer_generation"]["max_evidence_items"] == 15
