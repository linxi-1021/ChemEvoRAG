"""Patch Applier for ChemEvoRAG Skill Evolution.

Applies validated patches to Skill YAML files and Prompt Registry.

Usage:
    from skill_evolution.apply import PatchApplier, PatchApplyError
    applier = PatchApplier(skills_dir, prompts_dir)
    applier.apply_to_disk(skill_config, patches, round_id="round1")
"""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

from .attribution import FailureType
from .patch import PatchOperation, PatchSchema, PatchStatus


class PatchApplyError(Exception):
    """Raised when a patch cannot be applied."""
    pass


def _get_nested(d: dict, path: str) -> Any:
    """Get nested value by dot-separated path."""
    parts = path.split(".")
    current = d
    for p in parts:
        if isinstance(current, dict) and p in current:
            current = current[p]
        else:
            return None
    return current


def _set_nested(d: dict, path: str, value: Any) -> None:
    """Set nested value by dot-separated path."""
    parts = path.split(".")
    current = d
    for p in parts[:-1]:
        if p not in current:
            current[p] = {}
        elif not isinstance(current[p], dict):
            current[p] = {}
        current = current[p]
    current[parts[-1]] = value


def _bump_version(version: str) -> str:
    """Bump the patch component of a semver string: 1.0.0 -> 1.0.1."""
    parts = version.split(".")
    if len(parts) >= 3:
        try:
            parts[2] = str(int(parts[2]) + 1)
        except ValueError:
            parts.append("1")
    else:
        parts.append("1")
    return ".".join(parts)


class PatchApplier:
    """Applies validated patches to Skill YAML files and Prompt Registry."""

    def __init__(self, skills_dir: Path, prompts_dir: Path) -> None:
        self.skills_dir = Path(skills_dir)
        self.prompts_dir = Path(prompts_dir)

    def apply_to_memory(
        self,
        skill_config: dict,
        patch: PatchSchema,
    ) -> dict:
        """Apply a single patch to a skill config dict (in memory).

        Returns the modified skill config dict.
        Does NOT write to disk.
        """
        if yaml is None:
            raise PatchApplyError("pyyaml is required for patch application")

        target = patch.target_path
        operation = patch.operation

        if operation == PatchOperation.DELETE:
            raise PatchApplyError(
                f"DELETE operation is not allowed in automatic mode. "
                f"Patch {patch.patch_id} targets '{target}'."
            )

        if operation == PatchOperation.REPLACE:
            raise PatchApplyError(
                f"REPLACE operation is not allowed in automatic mode. "
                f"Patch {patch.patch_id} targets '{target}'."
            )

        if operation == PatchOperation.UPDATE:
            return self._apply_update(skill_config, patch)

        if operation == PatchOperation.ADD:
            return self._apply_add(skill_config, patch)

        if operation == PatchOperation.MERGE:
            return self._apply_merge(skill_config, patch)

        raise PatchApplyError(f"Unknown operation: {operation}")

    def _apply_update(self, skill_config: dict, patch: PatchSchema) -> dict:
        """Update operation: set fields in proposed_value at target_path."""
        target = patch.target_path
        current = _get_nested(skill_config, target)
        if current is None:
            # Create the target if it doesn't exist
            _set_nested(skill_config, target, {})
            current = _get_nested(skill_config, target)

        if not isinstance(current, dict):
            raise PatchApplyError(
                f"Target '{target}' is not a dict (type: {type(current).__name__}). "
                f"Cannot apply UPDATE."
            )

        for key, value in patch.proposed_value.items():
            current[key] = value
        return skill_config

    def _apply_add(self, skill_config: dict, patch: PatchSchema) -> dict:
        """Add operation: append to list or add key to dict at target_path."""
        target = patch.target_path
        current = _get_nested(skill_config, target)

        if current is None:
            # Create as list if proposed_value looks like a list item
            if isinstance(patch.proposed_value, dict) and "name" in patch.proposed_value:
                _set_nested(skill_config, target, [patch.proposed_value])
            else:
                _set_nested(skill_config, target, patch.proposed_value)
            return skill_config

        if isinstance(current, list):
            # Check for duplicates by name
            new_name = patch.proposed_value.get("name", "") if isinstance(patch.proposed_value, dict) else ""
            if new_name:
                for existing in current:
                    if isinstance(existing, dict) and existing.get("name") == new_name:
                        return skill_config  # already exists, skip
            current.append(patch.proposed_value)
        elif isinstance(current, dict):
            for key, value in patch.proposed_value.items():
                if key not in current:
                    current[key] = value
        else:
            raise PatchApplyError(
                f"Target '{target}' is neither a list nor a dict. Cannot ADD."
            )
        return skill_config

    def _apply_merge(self, skill_config: dict, patch: PatchSchema) -> dict:
        """Merge operation: only allowed for templates (list merge)."""
        target = patch.target_path
        if "template" not in target.lower():
            raise PatchApplyError(
                f"MERGE operation only allowed for templates. Target: '{target}'."
            )
        return self._apply_add(skill_config, patch)

    def _write_prompt_artifacts(self, patch: PatchSchema) -> list[str]:
        """Write prompt artifacts to the prompts directory.

        Returns list of written prompt file paths.
        """
        written: list[str] = []
        for artifact in patch.prompt_artifacts:
            if not artifact.prompt_ref or not artifact.registry_path:
                continue
            # Create the prompt file
            prompt_path = self.prompts_dir / Path(artifact.registry_path).name
            prompt_path.parent.mkdir(parents=True, exist_ok=True)

            # If the source prompt exists, copy it as base
            source_ref = artifact.created_from
            if source_ref:
                source_files = list(self.prompts_dir.glob(f"*{source_ref.lower()}*.yaml"))
                if source_files:
                    shutil.copy2(source_files[0], prompt_path)
                    written.append(str(prompt_path))
                    continue

            # Otherwise create a minimal placeholder
            placeholder = {
                "prompt_ref": artifact.prompt_ref,
                "version": "1.0.0",
                "status": "experimental",
                "created_from": artifact.created_from,
                "created_by": "skill_evolution_pipeline",
                "created_at": datetime.now().strftime("%Y-%m-%d"),
                "content": f"# TODO: Generate content for {artifact.prompt_ref}\n# Based on: {artifact.created_from}\n# Diff: {artifact.diff_summary}\n",
                "diff_summary": artifact.diff_summary,
            }
            prompt_path.write_text(
                yaml.dump(placeholder, allow_unicode=True, default_flow_style=False),
                encoding="utf-8",
            )
            written.append(str(prompt_path))

        return written

    def apply_to_disk(
        self,
        skill_config: dict,
        patches: list[PatchSchema],
        *,
        round_id: str,
    ) -> dict:
        """Apply patches to skill config and write back to disk.

        Updates skill_version, lifecycle.updated_at, provenance, and changelog.
        Writes prompt artifacts to config/prompts/.
        """
        if yaml is None:
            raise PatchApplyError("pyyaml is required")

        # Apply patches in memory
        for patch in patches:
            skill_config = self.apply_to_memory(skill_config, patch)
            patch.status = PatchStatus.APPLIED

        # Bump skill version
        old_version = skill_config.get("skill_version", "1.0.0")
        skill_config["skill_version"] = _bump_version(old_version)

        # Update lifecycle
        if "lifecycle" not in skill_config:
            skill_config["lifecycle"] = {}
        skill_config["lifecycle"]["updated_at"] = datetime.now().strftime("%Y-%m-%d")

        # Update provenance
        if "provenance" not in skill_config:
            skill_config["provenance"] = {}
        skill_config["provenance"]["last_modified_by"] = "skill_evolution_pipeline"
        skill_config["provenance"]["last_modified_reason"] = (
            f"Applied {len(patches)} patch(es) in {round_id}"
        )

        # Append to changelog
        if "changelog" not in skill_config:
            skill_config["changelog"] = []
        for patch in patches:
            affected = [patch.target_path]
            skill_config["changelog"].append({
                "version": skill_config["skill_version"],
                "round": round_id,
                "change_type": f"patch_{patch.operation.value}",
                "reason": patch.rationale[:200],
                "affected_paths": affected,
            })

        # Write prompt artifacts
        for patch in patches:
            self._write_prompt_artifacts(patch)

        # Write skill YAML to disk
        skill_name = skill_config.get("name", "unknown")
        skill_path = self.skills_dir / f"{skill_name}.yaml"
        skill_path.parent.mkdir(parents=True, exist_ok=True)
        skill_path.write_text(
            yaml.dump(skill_config, allow_unicode=True, default_flow_style=False, sort_keys=False),
            encoding="utf-8",
        )

        return skill_config
