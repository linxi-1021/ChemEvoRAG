"""PromptContext: explicit prompt injection for Skill YAML / Prompt Registry.

Replaces monkey-patching of solver.prompts module variables.
PromptContext is passed explicitly through the solver call chain,
so each question can use different prompts based on its intent + Skill YAML.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PromptContext:
    """Container for resolved prompt overrides from Skill YAML + Prompt Registry.

    Fields are None when no override is available; the solver should fallback
    to the default prompt from src/solver/prompts.py.
    """
    evidence_assessment_system: str | None = None
    answer_generation_system: str | None = None
    answer_generation_examples: list[tuple[str, str]] | None = None

    @property
    def has_assessment_override(self) -> bool:
        return bool(self.evidence_assessment_system)

    @property
    def has_answer_override(self) -> bool:
        return bool(self.answer_generation_system)


def resolve_prompt_context(
    skill_config: dict,
    prompt_registry: dict[str, str],
    *,
    warnings: list[str] | None = None,
) -> PromptContext:
    """Resolve Skill YAML + Prompt Registry into a PromptContext.

    Reads strategy.assessment.system_prompt_ref and
    strategy.answer_generation.system_prompt_ref from skill_config,
    looks them up in prompt_registry, and returns the resolved prompts.

    If a ref is missing or not found in registry, the corresponding field
    is left None (solver will fallback to default).

    Args:
        skill_config: Parsed Skill YAML dict (keyed by intent).
        prompt_registry: Dict mapping prompt_ref → prompt content.
        warnings: Optional list to append warning messages to.

    Returns:
        PromptContext with resolved overrides (None where fallback needed).
    """
    ctx = PromptContext()

    strategy = skill_config.get("strategy", {})

    # Assessment prompt
    assessment_ref = (
        strategy.get("assessment", {}).get("system_prompt_ref", "")
    )
    if assessment_ref:
        if assessment_ref in prompt_registry:
            ctx.evidence_assessment_system = prompt_registry[assessment_ref]
        else:
            if warnings is not None:
                warnings.append(
                    f"Prompt ref '{assessment_ref}' not found in registry, "
                    f"falling back to default EVIDENCE_ASSESSMENT_SYSTEM"
                )

    # Answer generation prompt
    answer_ref = (
        strategy.get("answer_generation", {}).get("system_prompt_ref", "")
    )
    if answer_ref:
        if answer_ref in prompt_registry:
            ctx.answer_generation_system = prompt_registry[answer_ref]
        else:
            if warnings is not None:
                warnings.append(
                    f"Prompt ref '{answer_ref}' not found in registry, "
                    f"falling back to default ANSWER_GENERATION_SYSTEM"
                )

    return ctx
