"""Reflexion-style episodic memory for Skill Evolution.

Each evolution round, LLM reflections on failures are saved per skill
and retrieved as context for the next round's prompt mutation.

Storage:
  data/evolution/memory/{skill_name}.jsonl

Each line is a JSON object:
  {
    "round": 1,
    "timestamp": "2026-07-06T12:00:00",
    "question_id": "q1",
    "reflection": "The assessment prompt failed to recognise that ...",
    "score": 0.0
  }

Usage:
    from skill_evolution.memory import (
        generate_reflections,
        load_memory,
        format_memory_for_prompt,
    )

    # Step 1: Generate reflections from failed questions
    reflections = generate_reflections(failures, skill_name="alias_resolution")

    # Step 2: Save to disk
    append_reflections(reflections, skill_name="alias_resolution")

    # Step 3: Load memory for the next evolution round
    memory = load_memory(skill_name="alias_resolution", last_n=10)
    context = format_memory_for_prompt(memory)
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MEMORY_DIR = PROJECT_ROOT / "data" / "evolution" / "memory"

# ── Public API ─────────────────────────────────────────────────────────────

def generate_reflections(
    failures: list[dict[str, Any]],
    skill_name: str = "",
) -> list[dict[str, Any]]:
    """Generate Reflexion-style reflections for failed questions.

    Each failure dict should contain at minimum:
      - question_id (str)
      - question (str)
      - system_answer (str)
      - ground_truth (str)
      - judge_score (float)
      - judge_reasoning (str)

    Returns a list of reflection dicts suitable for append_reflections().
    """
    reflections: list[dict[str, Any]] = []

    for f in failures:
        score = f.get("judge_score", f.get("score", 0.0))
        if score >= 0.8:
            continue  # not a failure

        question = f.get("question", "")
        system_answer = f.get("system_answer", "")[:300]
        ground_truth = f.get("ground_truth", "")[:300]
        judge_reasoning = f.get("judge_reasoning", "")[:300]

        # Generate reflection via LLM or fallback to heuristic
        reflection = _llm_reflect(
            question=question,
            system_answer=system_answer,
            ground_truth=ground_truth,
            judge_reasoning=judge_reasoning,
            score=score,
        ) or _heuristic_reflect(
            question=question,
            system_answer=system_answer,
            ground_truth=ground_truth,
            judge_reasoning=judge_reasoning,
            score=score,
        )

        reflections.append({
            "question_id": f.get("id", f.get("question_id", "")),
            "intent": f.get("intent", skill_name),
            "question": question[:200],
            "system_answer_snippet": system_answer,
            "ground_truth_snippet": ground_truth,
            "score": score,
            "reflection": reflection,
            "timestamp": datetime.now().isoformat(),
        })

    return reflections


def append_reflections(
    reflections: list[dict[str, Any]],
    skill_name: str = "default",
    round_id: int = 0,
) -> Path:
    """Append reflections to the memory file for a skill."""
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    path = MEMORY_DIR / f"{skill_name}.jsonl"

    lines = []
    for r in reflections:
        r["round"] = round_id
        lines.append(json.dumps(r, ensure_ascii=False) + "\n")

    with path.open("a", encoding="utf-8") as f:
        f.writelines(lines)

    return path


def load_memory(
    skill_name: str = "default",
    last_n: int = 10,
) -> list[dict[str, Any]]:
    """Load the most recent reflections for a skill."""
    path = MEMORY_DIR / f"{skill_name}.jsonl"
    if not path.exists():
        return []

    entries: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    # Return most recent entries
    return entries[-last_n:]


def format_memory_for_prompt(
    memories: list[dict[str, Any]],
    max_memories: int = 5,
) -> str:
    """Format prior reflections as context for prompt mutation.

    Returns a string suitable for injecting into an LLM prompt that says
    'Here is what went wrong in previous evaluation rounds...'
    """
    if not memories:
        return "No prior failure memories available."

    lines = ["## Prior Failure Reflections (from previous evolution rounds)\n"]
    lines.append("These are the lessons learned from past evaluation failures:\n")

    for i, m in enumerate(memories[-max_memories:], 1):
        lines.append(
            f"### Memory {i} (round {m.get('round', '?')}, "
            f"score={m.get('score', 0):.2f})\n"
        )
        lines.append(f"Question: {m.get('question', '')[:200]}")
        lines.append(f"Reflection: {m.get('reflection', '')}")
        lines.append("")

    return "\n".join(lines)


def clear_memory(skill_name: str = "") -> None:
    """Clear all memory files (or a specific skill's memory)."""
    if skill_name:
        path = MEMORY_DIR / f"{skill_name}.jsonl"
        if path.exists():
            path.unlink()
    else:
        if MEMORY_DIR.exists():
            for f in MEMORY_DIR.glob("*.jsonl"):
                f.unlink()


def count_memories(skill_name: str = "default") -> int:
    """Count stored reflections for a skill."""
    path = MEMORY_DIR / f"{skill_name}.jsonl"
    if not path.exists():
        return 0
    return sum(1 for _ in path.open("r", encoding="utf-8"))


# ── Internal helpers ─────────────────────────────────────────────────────

def _llm_reflect(
    question: str,
    system_answer: str,
    ground_truth: str,
    judge_reasoning: str,
    score: float,
) -> str | None:
    """Generate reflection via LLM. Returns None if LLM is unavailable."""
    key = os.environ.get("API_KEY")
    if not key:
        return None

    try:
        from openai import OpenAI
    except ImportError:
        return None

    prompt = _REFLECTION_PROMPT.format(
        question=question[:500],
        system_answer=system_answer[:500],
        ground_truth=ground_truth[:500],
        judge_reasoning=judge_reasoning[:300],
        score=score,
    )

    try:
        client = OpenAI(api_key=key, base_url=os.environ.get("BASE_URL") or None)
        response = client.chat.completions.create(
            model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=400,
            timeout=30,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception:
        return None


def _heuristic_reflect(
    question: str,
    system_answer: str,
    ground_truth: str,
    judge_reasoning: str,
    score: float,
) -> str:
    """Fallback: simple pattern-based reflection."""
    answer_lower = system_answer.lower()

    if any(w in answer_lower for w in ["insufficient", "not found", "no evidence",
                                         "cannot determine", "does not contain"]):
        return (
            f"Assessment declared evidence insufficient (score={score:.1f}). "
            f"Ground truth: {ground_truth[:200]}. "
            "The evidence ASSESSMENT prompt may need to be more permissive or "
            "better instructed on how to combine partial evidence."
        )

    if score < 0.3:
        return (
            f"Critical failure (score={score:.1f}). "
            f"System answered: {system_answer[:200]}. "
            f"Expected: {ground_truth[:200]}. "
            "Possible retrieval gap or major hallucination."
        )

    return (
        f"Partial failure (score={score:.1f}). "
        f"System answered: {system_answer[:200]}. "
        f"Expected: {ground_truth[:200]}. "
        f"Judge said: {judge_reasoning[:200]}. "
        "Answer may be partially correct but missing key details or precision."
    )


_REFLECTION_PROMPT = """\
You are analysing a failure in a chemistry RAG system. Given the question,
system answer, ground truth, and judge feedback, produce a short (2-4 sentence)
reflection describing:

1. What went wrong (retrieval gap, evidence misreading, hallucination, or
   overly conservative assessment?)
2. What should change in the system's prompts or strategy to fix this.
3. Be specific and actionable. Do NOT say "improve the prompt" — say WHAT
   specifically the prompt should do differently.

QUESTION: {question}
SYSTEM ANSWER: {system_answer}
GROUND TRUTH: {ground_truth}
JUDGE: {judge_reasoning}
SCORE: {score:.2f}

Reflection:"""
