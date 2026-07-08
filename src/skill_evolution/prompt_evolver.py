"""Prompt evolution: generate improved prompt candidates from failure analysis.

Implements a simplified Reflexion-style prompt mutation:
  1. Read current prompt (V1)
  2. Read failure reflections from memory
  3. Read bootstrapped few-shot examples from success traces
  4. Ask LLM to produce 2 improved prompt candidates (V2a, V2b)

Usage:
    from skill_evolution.prompt_evolver import generate_prompt_candidates

    candidates = generate_prompt_candidates(
        prompt_role="evidence_assessment",
        skill_name="alias_resolution",
        current_prompt=v1_content,
        reflections=recent_memories,
        failures=failure_samples,
        bootstrapped_examples=examples,
    )
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from .patch import PatchSchema, PatchOperation, PromptArtifact
from .types import FailureType

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def generate_prompt_candidates(
    prompt_role: str,
    skill_name: str,
    current_prompt: str,
    reflections: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    bootstrapped_examples: list[tuple[str, str]] | None = None,
    num_candidates: int = 2,
) -> list[dict[str, Any]]:
    """Generate improved prompt candidates via LLM.

    Each candidate dict contains:
      - prompt_ref: str (V2 ref name, e.g. "EVIDENCE_ASSESSMENT_SYSTEM_V2")
      - content: str (the full new prompt text)
      - change_summary: list[str] (what changed)
      - rationale: str

    If LLM is unavailable, falls back to heuristic improvement.
    """
    key = os.environ.get("API_KEY")

    if key:
        candidates = _llm_generate(
            prompt_role=prompt_role,
            skill_name=skill_name,
            current_prompt=current_prompt,
            reflections=reflections,
            failures=failures,
            bootstrapped_examples=bootstrapped_examples,
            num_candidates=num_candidates,
        )
    else:
        candidates = _heuristic_generate(
            prompt_role=prompt_role,
            skill_name=skill_name,
            current_prompt=current_prompt,
            reflections=reflections,
            failures=failures,
        )

    return candidates


def build_prompt_update_patches(
    candidates: list[dict[str, Any]],
    skill_name: str,
    skill_version: str,
    failure_ids: list[str],
    prompt_role: str,
    current_prompt_ref: str,
) -> list[PatchSchema]:
    """Convert prompt candidates into PatchSchema objects.

    Generates:
      - 1 prompt_content_update patch per candidate (writes new prompt file)
      - 1 prompt_ref_update patch per candidate (updates skill YAML ref)
    """
    patches: list[PatchSchema] = []

    for i, cand in enumerate(candidates):
        new_ref = cand["prompt_ref"]
        content = cand.get("content", "")

        # Content patch
        content_patch = PatchSchema(
            patch_id=f"{skill_name}_{new_ref.lower()}_v{i+1}",
            patch_type="prompt_content_update",
            skill_name=skill_name,
            skill_version=skill_version,
            target_file=f"config/prompts/{new_ref.lower()}.yaml",
            source_failure_ids=failure_ids,
            primary_failure_type=FailureType.ASSESSMENT_FALSE_NEG,
            target_path=f"config/prompts/{new_ref.lower()}.yaml",
            operation=PatchOperation.ADD,
            prompt_artifacts=[PromptArtifact(
                prompt_ref=new_ref,
                base_prompt_ref=current_prompt_ref,
                prompt_role=prompt_role,
                version="2.0",
                prompt_filename=f"{new_ref.lower()}.yaml",
                created_from=current_prompt_ref,
                diff_summary="; ".join(cand.get("change_summary", [])),
                registry_path=f"config/prompts/{new_ref.lower()}.yaml",
                content=content,
                change_summary=cand.get("change_summary", []),
                preserved_constraints=[],
                targeted_failure_ids=failure_ids,
                validation_requirements=["targeted_regression", "global_regression"],
            )],
            rationale=cand.get("rationale", ""),
            expected_improvement=cand.get("rationale", ""),
            risk_level="medium",
            confidence=cand.get("confidence", 0.7),
        )
        patches.append(content_patch)

        # Ref update patch
        ref_patch = PatchSchema(
            patch_id=f"{skill_name}_{new_ref.lower()}_ref_v{i+1}",
            patch_type="prompt_ref_update",
            skill_name=skill_name,
            skill_version=skill_version,
            target_file=f"config/skills/{skill_name}.yaml",
            source_failure_ids=failure_ids,
            primary_failure_type=FailureType.ASSESSMENT_FALSE_NEG,
            target_path=f"strategy.{'assessment' if 'ASSESSMENT' in prompt_role.upper() else 'answer_generation'}",
            operation=PatchOperation.UPDATE,
            current_value={"system_prompt_ref": current_prompt_ref},
            proposed_value={"system_prompt_ref": new_ref},
            rationale=f"Point to V2 prompt '{new_ref}'",
            expected_improvement=cand.get("rationale", ""),
            risk_level="medium",
            confidence=cand.get("confidence", 0.7),
            dependencies=[content_patch.patch_id],
        )
        # Store the content_patch as an "auxiliary" — the ref_update implicitly
        # requires it. The validation sandbox reads this field to know which
        # content patches to co-apply.
        ref_patch._auxiliary_content_patch = content_patch
        patches.append(ref_patch)

    return patches


# ── Internal: LLM-driven generation ──────────────────────────────────────

def _llm_generate(
    prompt_role: str,
    skill_name: str,
    current_prompt: str,
    reflections: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    bootstrapped_examples: list[tuple[str, str]] | None = None,
    num_candidates: int = 2,
) -> list[dict[str, Any]]:
    """Call LLM to generate improved prompt candidates."""
    key = os.environ.get("API_KEY")
    try:
        from openai import OpenAI
    except ImportError:
        return _heuristic_generate(prompt_role, skill_name, current_prompt, reflections, failures)

    # Build context
    reflection_text = _format_reflections(reflections)
    failure_text = _format_failures(failures)
    example_text = _format_examples(bootstrapped_examples)

    prompt = _CANDIDATE_PROMPT.format(
        prompt_role=prompt_role,
        skill_name=skill_name,
        current_prompt=current_prompt[:3000],
        reflection_context=reflection_text,
        failure_context=failure_text,
        example_context=example_text,
    )

    candidates = []
    client = OpenAI(api_key=key, base_url=os.environ.get("BASE_URL") or None)
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")

    for i in range(num_candidates):
        temp = 0.3 if i > 0 else 0.0  # Slight diversity for second candidate
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temp,
                max_tokens=3000,
                timeout=120,
            )
            raw = (response.choices[0].message.content or "").strip()
            if raw.startswith("```"):
                # Strip markdown code fences
                lines = raw.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                raw = "\n".join(lines)

            parsed = json.loads(raw)
            # Normalize: some LLMs return flat camelCase, others return nested
            if isinstance(parsed, dict) and "content" in parsed:
                candidates.append(parsed)
            elif isinstance(parsed, dict) and "prompt_ref" in parsed:
                # LLM returned the flat dict directly — wrap it
                candidates.append(parsed)
        except Exception:
            continue

    if not candidates:
        return _heuristic_generate(prompt_role, skill_name, current_prompt, reflections, failures)

    return candidates


def _heuristic_generate(
    prompt_role: str,
    skill_name: str,
    current_prompt: str,
    reflections: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Fallback: append reflection-driven rules to the current prompt."""
    if not current_prompt:
        return []

    # Derive a V2 ref name
    role_upper = prompt_role.upper().replace(" ", "_")
    new_ref = f"{role_upper}_V2"

    # Collect improvement signals from reflections
    signals: list[str] = []
    for r in reflections[:10]:
        text = r.get("reflection", "")
        if "insufficient" in text.lower() or "assessment" in text.lower():
            signals.append("insufficient_assessment")
        if "numeric" in text.lower() or "value" in text.lower():
            signals.append("numeric_extraction")
        if "table" in text.lower() or "entry" in text.lower():
            signals.append("table_parsing")
        if "compound" in text.lower() or "entity" in text.lower():
            signals.append("entity_disambiguation")

    unique_signals = list(dict.fromkeys(signals))
    if not unique_signals:
        unique_signals = ["general"]

    # Build improvement rules
    rules: list[str] = []
    rule_labels: list[str] = []

    if "numeric_extraction" in unique_signals:
        rules.append(
            "NUMERIC VALUE DETECTION: Before declaring sufficient=false, "
            "carefully check if the evidence contains numeric values "
            "(yields, temperatures, percentages, ratios) that answer the "
            "question. A table cell containing '85' under a 'Yield' column "
            "means 85% yield."
        )
        rule_labels.append("numeric extraction enhancement")

    if "insufficient_assessment" in unique_signals:
        rules.append(
            "EVIDENCE SUFFICIENCY: Only declare insufficient if NO evidence "
            "item contains ANY relevant data. If partial data exists, "
            "mark sufficient=true with appropriate confidence."
        )
        rule_labels.append("relaxed assessment threshold")

    if "table_parsing" in unique_signals:
        rules.append(
            "TABLE PARSING: Parse pipe-delimited and Markdown tables carefully. "
            "Match entry numbers to correct rows. Extract values with column "
            "header context."
        )
        rule_labels.append("table parsing instruction")

    if "entity_disambiguation" in unique_signals:
        rules.append(
            "ENTITY DISAMBIGUATION: Verify the evidence mentions the SAME "
            "compound/entity that the question asks about before declaring "
            "sufficient=true."
        )
        rule_labels.append("entity disambiguation")

    if "general" in unique_signals:
        rules.append(
            "GENERAL: Re-read each evidence item before concluding. "
            "Check if the answer can be derived by combining multiple items."
        )
        rule_labels.append("general assessment improvement")

    improved = current_prompt.rstrip() + "\n\n" + "\n\n".join(rules) + "\n"

    return [{
        "prompt_ref": new_ref,
        "content": improved,
        "change_summary": rule_labels,
        "rationale": f"Heuristic improvement based on failure patterns: {'; '.join(rule_labels)}.",
        "confidence": 0.7,
    }]


# ── Formatting helpers ────────────────────────────────────────────────────

def _format_reflections(reflections: list[dict[str, Any]]) -> str:
    if not reflections:
        return "No prior failure reflections available."
    lines = []
    for i, r in enumerate(reflections[-5:], 1):
        lines.append(
            f"[{i}] Score: {r.get('score', '?')}. "
            f"Reflection: {r.get('reflection', '')[:300]}"
        )
    return "\n".join(lines)


def _format_failures(failures: list[dict[str, Any]]) -> str:
    if not failures:
        return "No failure samples provided."
    lines = []
    for i, f in enumerate(failures[:5], 1):
        lines.append(
            f"[{i}] Q: {f.get('question', '')[:200]}\n"
            f"    System: {f.get('system_answer', '')[:200]}\n"
            f"    Truth: {f.get('ground_truth', '')[:200]}\n"
            f"    Score: {f.get('score', f.get('judge_score', 0)):.2f}"
        )
    return "\n".join(lines)


def _format_examples(examples: list[tuple[str, str]] | None) -> str:
    if not examples:
        return "No bootstrapped examples available."
    lines = []
    for i, (q, a) in enumerate(examples[:3], 1):
        lines.append(f"[{i}] {q[:300]}\n    → {a[:300]}")
    return "\n".join(lines)


_CANDIDATE_PROMPT = """\
You are optimizing a prompt for a chemistry RAG system. The current prompt
is used for: {prompt_role} (skill: {skill_name}).

=== CURRENT PROMPT (V1) ===
{current_prompt}

=== REFLECTION MEMORY (lessons from past failures) ===
{reflection_context}

=== RECENT FAILURE SAMPLES ===
{failure_context}

=== SUCCESS EXAMPLES (for reference) ===
{example_context}

=== TASK ===
Generate an improved V2 version of this prompt. Your improvements should
directly address the failure patterns described above.

Return ONLY a JSON object:
{{
  "prompt_ref": "EVIDENCE_ASSESSMENT_SYSTEM_V2" (or ANSWER_GENERATION_SYSTEM_V2),
  "content": "<the full V2 prompt text>",
  "change_summary": ["change 1", "change 2", ...],
  "rationale": "<why these changes address the failures>",
  "confidence": 0.7
}}

Rules:
- The content field must contain the COMPLETE V2 prompt, not just additions.
- Preserve all working parts of V1. Only modify what needs fixing.
- Be specific. Don't say "improve assessment" — say HOW.
- If failures show that the assessment rejects evidence with numeric data,
  add explicit instructions to extract and check numeric values first.
- If failures show table parsing errors, add table parsing rules.
"""
