"""LLM-driven Failure Analysis and Prompt Generation for ChemEvoRAG Skill Evolution.

Implements Stage 2 (LLM Failure Analysis) and Stage 3 (LLM Prompt Generation)
of the Skill Evolution design spec.

Stage 2: Analyze failure clusters -> FailureAnalysisReport
Stage 3: Generate V2 prompt based on V1 + FailureAnalysisReport

Usage:
    from skill_evolution.llm_analysis import (
        analyze_failure_cluster,
        generate_v2_prompt,
    )
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

try:
    from pydantic import BaseModel, ConfigDict, Field
except ImportError:
    raise ImportError("pydantic is required for skill_evolution.llm_analysis")


def _call_llm(system_prompt: str, user_prompt: str, temperature: float = 0.3) -> str | None:
    """Call LLM API (same config as the main system)."""
    try:
        from openai import OpenAI
    except ImportError:
        return None

    api_key = os.environ.get("API_KEY")
    base_url = os.environ.get("BASE_URL")
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")

    if not api_key:
        return None

    client = OpenAI(api_key=api_key, base_url=base_url or None)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=4096,
        )
        return resp.choices[0].message.content
    except Exception as e:
        print(f"[LLM Analysis] API call failed: {e}")
        return None


def _extract_json(text: str) -> dict | None:
    """Extract JSON from LLM response (handles markdown code blocks)."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    m = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass

    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    return None


# ---------------------------------------------------------------------------
# Failure sample builder
# ---------------------------------------------------------------------------

def _build_failure_sample_text(
    failure_record: dict[str, Any],
    react_log: str,
) -> str:
    """Build a text summary of one failure sample for LLM analysis."""
    lines = []
    lines.append(f"Question ID: {failure_record.get('question_id', 'unknown')}")
    lines.append(f"Skill: {failure_record.get('skill_name', 'unknown')}")
    lines.append(f"Score: {failure_record.get('score', 0.0)}")
    lines.append(f"Failure type (rule): {failure_record.get('primary_failure_type', 'unknown')}")
    lines.append(f"Attribution confidence: {failure_record.get('attribution_confidence', 0.0)}")
    lines.append(f"Attribution explanation: {failure_record.get('attribution_explanation', '')}")
    lines.append("")
    lines.append(f"Question: {failure_record.get('question', 'N/A')}")
    lines.append(f"Gold answer: {failure_record.get('gold_answer', 'N/A')}")
    lines.append(f"Model answer: {failure_record.get('predicted_answer', 'N/A')}")
    lines.append("")

    # Assessment info
    assessment_result = failure_record.get('assessment_result', '')
    assessment_reason = failure_record.get('assessment_reason', '')
    if assessment_result:
        lines.append(f"Assessment result: {assessment_result}")
    if assessment_reason:
        lines.append(f"Assessment reason: {assessment_reason[:300]}")
    lines.append("")

    # Evidence containment
    lines.append(f"Evidence contained answer: {failure_record.get('evidence_contained_answer', False)}")
    lines.append(f"Retrieved evidence IDs: {failure_record.get('retrieved_evidence_ids', [])}")
    lines.append(f"Missing slots: {failure_record.get('missing_slots', [])}")
    lines.append("")

    # Retrieval rounds with evidence content
    retrieval_rounds = failure_record.get('retrieval_rounds', [])
    if retrieval_rounds:
        lines.append("Retrieval rounds:")
        for rr in retrieval_rounds[:3]:
            lines.append(f"  Round {rr.get('round', '?')}: query=\"{rr.get('query', '')[:150]}\"")
            if rr.get('refined_query'):
                lines.append(f"    Refined: \"{rr['refined_query'][:150]}\"")
            ev_items = rr.get('retrieved_evidence', [])
            if ev_items:
                lines.append(f"    Evidence ({len(ev_items)} items):")
                for ev in ev_items[:5]:
                    gold_flag = " [GOLD ANSWER HERE]" if ev.get('contains_gold_answer') else ""
                    lines.append(f"      [{ev.get('evidence_type', '?')}] {ev.get('evidence_id', '?')}: {ev.get('summary', '')[:200]}{gold_flag}")
            lines.append(f"    New evidence found: {rr.get('new_evidence_found', False)}")
            if rr.get('skipped_due_to_similarity'):
                lines.append("    SKIPPED due to query similarity")
        lines.append("")

    # React log evidence (fallback)
    if not retrieval_rounds and react_log:
        evidence_lines = re.findall(
            r"\[ReAct\]   \[\d+\] (\w+) \| ([^|]+) \| (.+?)(?:\n|$)", react_log
        )
        if evidence_lines:
            lines.append("Retrieved evidence (from react log):")
            for etype, eid, summary in evidence_lines[:8]:
                lines.append(f"  [{etype}] {eid}: {summary[:200]}")

    return "\n".join(lines)


def _build_failure_samples_for_prompt_gen(
    failure_records: list[dict[str, Any]],
) -> str:
    """Build a compact failure-focused section for Stage 3 prompt generation.
    Includes SPECIFIC evidence that was available but assessment missed."""
    lines = []
    lines.append("## Failure Evidence Details")
    lines.append("Below are the specific failures this improved prompt must fix:")
    lines.append("")
    for fr in failure_records[:5]:
        qid = fr.get('question_id', 'unknown')
        question = fr.get('question', '')[:200]
        gold = fr.get('gold_answer', '')[:200]
        model = fr.get('predicted_answer', '')[:200]
        lines.append(f"### Failure {qid}")
        lines.append(f"**Question:** {question}")
        lines.append(f"**Gold answer:** {gold}")
        lines.append(f"**Model answered:** {model}")
        lines.append(f"**Error:** Assessment judged evidence insufficient, but evidence DID contain the answer")
        # Show evidence that contained the answer
        retrieval_rounds = fr.get('retrieval_rounds', [])
        if retrieval_rounds:
            for rr in retrieval_rounds[:2]:
                ev_items = rr.get('retrieved_evidence', [])
                gold_ev = [e for e in ev_items if e.get('contains_gold_answer')]
                if gold_ev:
                    lines.append(f"**Evidence with gold answer:**")
                    for ev in gold_ev[:3]:
                        lines.append(f"  - [{ev.get('evidence_type', '?')}] {ev.get('evidence_id', '?')}: {ev.get('summary', '')[:250]}")
        # Check react log for evidence
        react_log = fr.get("react_log", "")
        if react_log:
            ev_lines = re.findall(
                r"\[ReAct\]   \[\d+\] (\w+) \| ([^|]+) \| (.+)", react_log
            )
            if ev_lines:
                lines.append(f"**All retrieved evidence in react log:**")
                for etype, eid, summary in ev_lines[:10]:
                    lines.append(f"  - [{etype}] {eid}: {summary[:200]}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Skill configs builder
# ---------------------------------------------------------------------------

def _build_skill_configs_section(skill_configs: dict[str, dict]) -> str:
    """Build a text section showing all affected skill configs."""
    lines = []
    for skill_name, config in skill_configs.items():
        lines.append(f"### {skill_name}")
        strategy = config.get('strategy', {})
        rr = strategy.get('retrieval_routing', {})
        if rr:
            lines.append(f"Retrieval: channels={rr.get('primary_channels', [])}, top_k={json.dumps(rr.get('top_k', {}))}")
            lines.append(f"  boost_weights: {json.dumps(rr.get('boost_weights', {}))}")
        qr = strategy.get('query_rewrite', {})
        if qr:
            lines.append(f"Query rewrite: fallback={qr.get('fallback_terms', [])}, div_threshold={qr.get('diversity_threshold', 0.8)}, max_variants={qr.get('max_variants', 3)}")
        ee = strategy.get('evidence_expansion', {})
        if ee:
            lines.append(f"Evidence expansion: max_blocks={ee.get('max_expanded_blocks', 5)}, block_neighbors={ee.get('expand_from_block_neighbors', True)}")
        assessment = strategy.get('assessment', {})
        if assessment:
            lines.append(f"Assessment: prompt_ref={assessment.get('system_prompt_ref', 'N/A')}")
        ag = strategy.get('answer_generation', {})
        if ag:
            lines.append(f"Answer gen: prompt_ref={ag.get('system_prompt_ref', 'N/A')}, max_items={ag.get('max_evidence_items', 15)}")
        evolution = config.get('evolution', {})
        if evolution:
            lines.append(f"Mutable: {evolution.get('mutable_paths', [])}")
            lines.append(f"Frozen: {evolution.get('frozen_paths', [])}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stage 2: LLM Failure Analysis
# ---------------------------------------------------------------------------

_ANALYSIS_SYSTEM_PROMPT = """You are a failure analysis expert for a chemistry RAG system.

Your task: Analyze a cluster of failure samples that share the same failure type.
Determine the common root causes, identify sub-clusters, and recommend improvements.

RULES:
1. Be SPECIFIC about WHY each failure occurred. Reference the actual evidence and answers.
2. Do NOT make vague suggestions like "improve assessment" — specify exactly what rule is missing.
3. At least ONE component MUST have should_change=true (unless it's a coverage gap).
4. Strictly follow the output JSON schema.
5. primary_failure_type MUST be a non-empty string from: assessment_false_negative, routing_error, entity_miss, alias_miss, local_id_miss, evidence_expansion_error, planner_error, table_reasoning_error, generation_error.
6. Output valid JSON only — no markdown wrappers, no extra text."""


def _build_analysis_user_prompt(
    cluster: dict[str, Any],
    failure_samples: list[str],
    current_prompts: dict[str, dict],
    current_skill_configs: dict[str, dict],
) -> str:
    lines = []
    lines.append("## Analysis Cluster")
    lines.append(f"Cluster ID: {cluster.get('cluster_id', 'unknown')}")
    lines.append(f"Failure type: {cluster.get('failure_type', 'unknown')}")
    lines.append(f"Target path: {cluster.get('normalized_target_path', 'unknown')}")
    lines.append(f"Prompt role: {cluster.get('prompt_role', 'unknown')}")
    lines.append(f"Affected skills: {cluster.get('affected_skills', [])}")
    lines.append(f"Cluster size: {cluster.get('cluster_size', 0)}")

    # Previous version regression feedback (refine round)
    prev_fb = cluster.get("previous_version_feedback")
    version_history = cluster.get("version_history", [])
    if prev_fb or version_history:
        lines.append("")
        lines.append("## Previous Version Regression Feedback (ALL rounds)")
        lines.append("The previous patch versions were tested and FAILED. Below is the FULL history:")
        for vh in version_history:
            lines.append(f"### V{vh.get('version', '?')}")
            fb = vh.get('feedback', {})
            lines.append(f"  Pass: {fb.get('pass_count', 0)}, Fail: {fb.get('fail_count', 0)}")
            for q in fb.get("regressed", [])[:5]:
                lines.append(f"  REGRESSED: {q.get('id','?')}: {q.get('before',0):.1f}→{q.get('after',0):.1f}")
            for q in fb.get("improved", [])[:3]:
                lines.append(f"  IMPROVED: {q.get('id','?')}: {q.get('before',0):.1f}→{q.get('after',0):.1f}")
        # Also show the latest feedback from prev_fb
        if prev_fb:
            for pid, fb in prev_fb.items():
                lines.append(f"### Latest attempt (Patch: {pid})")
                for q in fb.get("regressed", [])[:5]:
                    lines.append(f"  REGRESSED: {q.get('id','?')}: {q.get('before',0):.1f}→{q.get('after',0):.1f}")
                for q in fb.get("improved", [])[:3]:
                    lines.append(f"  IMPROVED: {q.get('id','?')}: {q.get('before',0):.1f}→{q.get('after',0):.1f}")
        lines.append("Your task: analyze ALL previous versions' failures and avoid repeating their mistakes.")
        lines.append("Generate a refined version that learns from the entire history.")

    lines.append("")
    lines.append("## Current Prompts")
    for role, prompt_info in current_prompts.items():
        lines.append(f"### {role} ({prompt_info.get('prompt_ref', 'N/A')})")
        lines.append("```")
        lines.append(prompt_info.get('content', 'N/A')[:2000])
        lines.append("```")
        lines.append("")
    lines.append("## Current Skill Configs")
    lines.append(_build_skill_configs_section(current_skill_configs))
    lines.append("## Failure Samples")
    for i, sample in enumerate(failure_samples, 1):
        lines.append(f"### Sample {i}")
        lines.append(sample)
        lines.append("")
    lines.append("""## Required Output (valid JSON, no markdown)
{
  "primary_failure_type": "assessment_false_negative",
  "contributing_failure_types": [],
  "confidence": 0.8,
  "coverage_gap": false,
  "explanation": "...",
  "supporting_trace_signals": ["..."],
  "common_root_causes": [
    {"name": "...", "description": "...", "supporting_failure_ids": ["q1"], "confidence": 0.8}
  ],
  "subclusters": [],
  "proposed_patch_targets": ["strategy.assessment"],
  "recommended_changes": {
    "prompt": {"should_change": true, "priority": "primary", "prompt_role": "evidence_assessment", "base_prompt_ref": "EVIDENCE_ASSESSMENT_SYSTEM", "change_summary": ["Add specific rule..."], "risk_level": "medium", "confidence": 0.8},
    "retrieval_routing": {"should_change": false, "priority": "secondary", "changes": [], "risk_level": "low"},
    "query_rewrite": {"should_change": false, "priority": "secondary", "changes": [], "risk_level": "low"},
    "evidence_expansion": {"should_change": false, "priority": "secondary", "changes": [], "risk_level": "low"},
    "answer_generation": {"should_change": false, "priority": "secondary", "changes": [], "risk_level": "low"}
  },
  "should_generate_patch": true,
  "risk_level": "medium"
}""")
    return "\n".join(lines)


class FailureAnalysisReport(BaseModel):
    """Output of Stage 2: LLM failure analysis."""
    model_config = ConfigDict(populate_by_name=True, validate_assignment=True, extra="forbid")
    cluster_id: str = ""
    primary_failure_type: str = ""
    contributing_failure_types: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    coverage_gap: bool = False
    explanation: str = ""
    supporting_trace_signals: list[str] = Field(default_factory=list)
    common_root_causes: list[dict[str, Any]] = Field(default_factory=list)
    subclusters: list[dict[str, Any]] = Field(default_factory=list)
    proposed_patch_targets: list[str] = Field(default_factory=list)
    recommended_changes: dict[str, Any] = Field(default_factory=dict)
    should_generate_patch: bool = False
    risk_level: str = "medium"
    raw_llm_response: str = ""


def _validate_analysis_output(parsed: dict) -> list[str]:
    """Validate Stage 2 output. Returns list of error messages."""
    errors = []
    if not parsed.get("primary_failure_type"):
        errors.append("primary_failure_type is empty")
    root_causes = parsed.get("common_root_causes", [])
    if not root_causes:
        errors.append("common_root_causes is empty")
    rc = parsed.get("recommended_changes", {})
    has_change = any(
        rc.get(c, {}).get("should_change", False)
        for c in ("prompt", "retrieval_routing", "query_rewrite", "evidence_expansion", "answer_generation")
    )
    if not has_change:
        errors.append("no component has should_change=true")
    return errors


def analyze_failure_cluster(
    cluster: dict[str, Any],
    failure_records: list[dict[str, Any]],
    current_prompts: dict[str, dict],
    current_skill_configs: dict[str, dict],
    *,
    max_retries: int = 3,
) -> FailureAnalysisReport:
    """Stage 2 with retry + structured output validation."""
    sample_texts = []
    for fr in failure_records:
        react_log = fr.get("react_log", "")
        sample_texts.append(_build_failure_sample_text(fr, react_log))

    user_prompt = _build_analysis_user_prompt(
        cluster, sample_texts, current_prompts, current_skill_configs,
    )

    last_errors: list[str] = ["LLM call failed"]
    last_raw: str = ""
    for retry_idx in range(max_retries):
        temperature = 0.1 * (retry_idx + 1)
        llm_response = _call_llm(_ANALYSIS_SYSTEM_PROMPT, user_prompt, temperature=temperature)
        if not llm_response:
            last_errors = ["LLM call returned None"]
            continue
        last_raw = llm_response
        parsed = _extract_json(llm_response)
        if not parsed:
            last_errors = ["Response is not valid JSON"]
            continue
        errors = _validate_analysis_output(parsed)
        if not errors:
            return FailureAnalysisReport(
                cluster_id=parsed.get("cluster_id", cluster.get("cluster_id", "")),
                primary_failure_type=parsed.get("primary_failure_type", ""),
                contributing_failure_types=parsed.get("contributing_failure_types", []),
                confidence=parsed.get("confidence", 0.0),
                coverage_gap=parsed.get("coverage_gap", False),
                explanation=parsed.get("explanation", ""),
                supporting_trace_signals=parsed.get("supporting_trace_signals", []),
                common_root_causes=parsed.get("common_root_causes", []),
                subclusters=parsed.get("subclusters", []),
                proposed_patch_targets=parsed.get("proposed_patch_targets", []),
                recommended_changes=parsed.get("recommended_changes", {}),
                should_generate_patch=parsed.get("should_generate_patch", True),
                risk_level=parsed.get("risk_level", "medium"),
                raw_llm_response=llm_response,
            )
        last_errors = errors
        print(f"  [LLM] Retry {retry_idx + 1}/{max_retries}: {errors}")

    return FailureAnalysisReport(
        cluster_id=cluster.get("cluster_id", ""),
        should_generate_patch=False,
        risk_level="high",
        raw_llm_response=f"Failed after {max_retries} retries. Errors: {json.dumps(last_errors)}\nLast: {last_raw[:500]}",
    )


# ---------------------------------------------------------------------------
# Stage 3: LLM Prompt Generation (with failure-evidence-driven rules)
# ---------------------------------------------------------------------------

_GENERATION_SYSTEM_PROMPT = """You are a prompt engineer fixing a chemistry RAG system.

Your task: The current prompt caused assessment failures on specific questions.
You have the current prompt AND the exact failure evidence. Generate an improved version with TARGETED rules.

CRITICAL INSTRUCTION:
- You may add, remove, or modify any part of the current prompt.
- Each change must be driven by the failure evidence — be SPECIFIC about which failure sample motivates which change.
- Example of a GOOD change: "When a table shows per-entry yields (e.g., Entry 1: 78%, Entry 5: 42%), and the question asks which entry gave a higher yield, the table IS sufficient — extract and compare."
- Example of a BAD change: "Improve evidence assessment."
- MUST preserve: output JSON schema (sufficient/reason/refined_query), anti-hallucination constraints.
- MAY modify: sufficiency rules, table parsing rules, reasoning examples, pre-flight check.
- Output valid JSON only."""


def _build_generation_user_prompt(
    analysis_report: FailureAnalysisReport,
    v1_content: str,
    base_prompt_ref: str,
    new_prompt_ref: str,
    failure_records: list[dict[str, Any]],
) -> str:
    """Build Stage 3 user prompt including specific failure evidence."""
    lines = []
    # Support both FailureAnalysisReport object and dict
    _get = lambda obj, key, default='': obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)

    lines.append("## What Went Wrong")
    lines.append(f"Failure type: {_get(analysis_report, 'primary_failure_type')}")
    lines.append(f"Risk level: {_get(analysis_report, 'risk_level')}")
    lines.append("")
    lines.append("### Root Causes from Analysis")
    for rc in _get(analysis_report, 'common_root_causes', []):
        lines.append(f"- {_get(rc, 'name', '?')}: {_get(rc, 'description', '')}")
    lines.append("")

    # Include specific failure evidence
    lines.append(_build_failure_samples_for_prompt_gen(failure_records))

    lines.append("## Current Prompt (to be improved)")
    lines.append("```")
    lines.append(v1_content)
    lines.append("```")
    lines.append("")
    lines.append(f"""## Required Output (valid JSON, no markdown wrappers)
{{
  "prompt_ref": "{new_prompt_ref}",
  "base_prompt_ref": "{base_prompt_ref}",
  "prompt_role": "evidence_assessment",
  "version": "2.0",
  "prompt_filename": "{new_prompt_ref.lower()}.yaml",
  "content": "THE FULL IMPROVED PROMPT TEXT",
  "change_summary": ["Added rule: ...", "Modified section: ...", "Removed rule: ..."],
  "preserved_constraints": ["Evidence must support the answer", "Do not infer missing facts"],
  "validation_requirements": ["prompt_ref_unique", "content_non_empty", "grounding_constraints_preserved"]
}}""")
    return "\n".join(lines)


class PromptArtifact(BaseModel):
    """Output of Stage 3: complete prompt artifact."""
    model_config = ConfigDict(populate_by_name=True, validate_assignment=True, extra="forbid")
    prompt_ref: str = ""
    base_prompt_ref: str = ""
    prompt_role: str = ""
    version: str = "2.0"
    prompt_filename: str = ""
    content: str = ""
    change_summary: list[str] = Field(default_factory=list)
    preserved_constraints: list[str] = Field(default_factory=list)
    targeted_failure_ids: list[str] = Field(default_factory=list)
    validation_requirements: list[str] = Field(default_factory=list)
    raw_llm_response: str = ""
    success: bool = False


def _validate_v2_quality(v1_content: str, v2_content: str, change_summary: list[str]) -> list[str]:
    """Check V2 has real improvements over V1. Returns error messages."""
    errors = []
    # V2 must differ from V1 significantly
    if v2_content.strip() == v1_content.strip():
        errors.append("V2 content is identical to V1 — no improvement")
    if not change_summary:
        errors.append("V2 change_summary is empty")
    # V2 can be shorter (removed bad rules) as long as it differs meaningfully
    return errors


def generate_v2_prompt(
    analysis_report: FailureAnalysisReport,
    v1_content: str,
    base_prompt_ref: str,
    new_prompt_ref: str,
    targeted_failure_ids: list[str] | None = None,
    failure_records: list[dict[str, Any]] | None = None,
    *,
    max_retries: int = 3,
) -> PromptArtifact:
    """Stage 3 with retry + V2 quality validation."""
    user_prompt = _build_generation_user_prompt(
        analysis_report, v1_content, base_prompt_ref, new_prompt_ref,
        failure_records or [],
    )

    last_raw: str = ""
    for retry_idx in range(max_retries):
        temperature = 0.1 * (retry_idx + 1)
        llm_response = _call_llm(_GENERATION_SYSTEM_PROMPT, user_prompt, temperature=temperature)
        if not llm_response:
            continue
        last_raw = llm_response
        parsed = _extract_json(llm_response)
        if not parsed:
            continue
        content = parsed.get("content", "")
        change_summary = parsed.get("change_summary", [])
        if not content or len(content) < 100:
            print(f"  [LLM] Stage 3 retry {retry_idx + 1}: content too short ({len(content)} chars)")
            continue

        # Quality check
        errors = _validate_v2_quality(v1_content, content, change_summary)
        if errors:
            print(f"  [LLM] Stage 3 retry {retry_idx + 1}: {errors}")
            continue

        return PromptArtifact(
            prompt_ref=parsed.get("prompt_ref", new_prompt_ref),
            base_prompt_ref=parsed.get("base_prompt_ref", base_prompt_ref),
            prompt_role=parsed.get("prompt_role", "evidence_assessment"),
            version=parsed.get("version", "2.0"),
            prompt_filename=parsed.get("prompt_filename", f"{new_prompt_ref.lower()}.yaml"),
            content=content,
            change_summary=change_summary,
            preserved_constraints=parsed.get("preserved_constraints", []),
            targeted_failure_ids=targeted_failure_ids or [],
            validation_requirements=parsed.get("validation_requirements", []),
            raw_llm_response=llm_response,
            success=True,
        )

    return PromptArtifact(
        prompt_ref=new_prompt_ref,
        base_prompt_ref=base_prompt_ref,
        raw_llm_response=f"V2 generation failed after {max_retries} retries.\nLast: {last_raw[:500]}",
        success=False,
    )


# ---------------------------------------------------------------------------
# Refine loop: iterative prompt improvement based on regression feedback
# ---------------------------------------------------------------------------

_REFINE_SYSTEM_PROMPT = """You are a prompt engineer fixing a chemistry RAG system.

Your previous prompt version was tested in regression. You will receive:
- V1 (current stable prompt)
- Your previous version
- Fresh regression results from THIS version: which questions improved, which regressed

Your task:
1. Analyze the regression results. Why did specific questions regress?
2. Based on the evidence, decide whether to add, remove, or modify rules.
3. Generate the next version. It can be more conservative OR more targeted —
   the regression data should guide your decision, not a preset direction.
4. If the regression data shows you cannot fix targets without regressions, set should_apply=false.

Output valid JSON only:
{
  "should_apply": true,
  "regression_analysis": "explain what you changed and why",
  "refined_rules": [
    {"rule": "...", "trigger_condition": "when to apply", "exclusion": "when NOT to apply"}
  ],
  "content": "THE FULL NEXT VERSION PROMPT TEXT",
  "change_summary": ["..."],
  "preserved_constraints": ["..."]
}"""


def _build_regression_feedback_text(feedback: dict) -> str:
    """Build a text summary of regression results for the refine prompt."""
    lines = []
    lines.append(f"Summary: {feedback.get('pass_count', 0)} passed, "
                 f"{feedback.get('fail_count', 0)} failed")
    lines.append("")

    # Improved samples
    improved = feedback.get('improved', [])
    if improved:
        lines.append("## Improved Questions (before → after)")
        for q in improved[:5]:
            lines.append(f"- {q.get('id', '?')}: {q.get('before', 0):.1f} → {q.get('after', 0):.1f}")
    lines.append("")

    # Regressed samples
    regressed = feedback.get('regressed', [])
    if regressed:
        lines.append("## Regressed Questions (before → after)")
        for q in regressed[:10]:
            lines.append(f"- {q.get('id', '?')}: {q.get('before', 0):.1f} → {q.get('after', 0):.1f}")
            if q.get('v1_answer'):
                lines.append(f"  V1 answer: {q['v1_answer'][:200]}")
            if q.get('v2_answer'):
                lines.append(f"  V2 answer: {q['v2_answer'][:200]}")
            if q.get('question'):
                lines.append(f"  Question: {q['question'][:200]}")
    lines.append("")

    # Unchanged failures
    unchanged = feedback.get('unchanged_failures', [])
    if unchanged:
        lines.append("## Unchanged Failures (still wrong)")
        for q in unchanged[:5]:
            lines.append(f"- {q.get('id', '?')}: {q.get('before', 0):.1f} → {q.get('after', 0):.1f}")
    lines.append("")

    return "\n".join(lines)


def build_refine_prompt(
    v1_content: str,
    v2_content: str,
    regression_feedback: dict,
    new_prompt_ref: str,
    base_prompt_ref: str,
    version_history: list[dict] | None = None,
) -> str:
    """Build the LLM prompt for the refine iteration."""
    lines = []
    lines.append("## Current Stable Prompt")
    lines.append("```")
    lines.append(v1_content[:3000])
    lines.append("```")
    lines.append("")

    # Show version history (accumulated)
    version_history = version_history or []
    if version_history:
        lines.append("## Version History")
        for vh in version_history:
            lines.append(f"### V{vh.get('version', '?')} Feedback")
            lines.append(_build_regression_feedback_text(vh.get('feedback', {})))
        lines.append("")
    else:
        lines.append("## Your Previous Attempt")
        lines.append("```")
        lines.append(v2_content[:3000])
        lines.append("```")
        lines.append("")
        lines.append("## Regression Results")
        lines.append(_build_regression_feedback_text(regression_feedback))
        lines.append("")

    lines.append(f"""## Required Output (valid JSON, no markdown)
{{
  "should_apply": true,
  "regression_analysis": "V2 rule X caused question Y to regress because...",
  "refined_rules": [
    {{"rule": "specific text to add", "trigger_condition": "when", "exclusion": "when NOT"}}
  ],
  "content": "THE FULL V3 PROMPT TEXT",
  "change_summary": ["Preserved V1 grounding", "Added conservative rule: ..."],
  "preserved_constraints": ["Do not infer missing facts", "Keep JSON output schema"]
}}""")
    return "\n".join(lines)


def refine_prompt_with_feedback(
    v1_content: str,
    v2_content: str,
    regression_feedback: dict,
    base_prompt_ref: str,
    new_prompt_ref: str,
    *,
    max_refine_iterations: int = 3,
) -> dict | None:
    """Iteratively refine a prompt based on regression feedback.

    Returns the refined prompt dict on success, None if no improvement possible.
    """
    current_v = v2_content
    last_valid = None
    for iteration in range(max_refine_iterations):
        temp = 0.1 * (iteration + 1)
        user_prompt = build_refine_prompt(
            v1_content, current_v, regression_feedback, new_prompt_ref, base_prompt_ref,
        )
        response = _call_llm(_REFINE_SYSTEM_PROMPT, user_prompt, temperature=temp)
        if not response:
            print(f"  [Refine] Iteration {iteration + 1}: LLM call failed")
            continue

        parsed = _extract_json(response)
        if not parsed:
            print(f"  [Refine] Iteration {iteration + 1}: invalid JSON")
            continue

        if not parsed.get("content") or len(parsed.get("content", "")) < 100:
            print(f"  [Refine] Iteration {iteration + 1}: content too short")
            continue

        # Always feed into next iteration — never exit early
        current_v = parsed["content"]
        last_valid = parsed
        print(f"  [Refine] Iteration {iteration + 1}/{max_refine_iterations}: "
              f"V{iteration + 3} ({len(parsed.get('refined_rules', []))} rules)")

    if last_valid:
        print(f"  [Refine] Complete: {max_refine_iterations} iterations, returning latest version")
        return last_valid

    print(f"  [Refine] Failed: no valid response in {max_refine_iterations} iterations")
    return None
