"""DSPy-style bootstrap: extract few-shot examples from successful traces.

Reads eval_results.json and filters questions with score >= 0.9.
For each successful question, extracts (question, answer, evidence) tuples
that can be used as few-shot examples in the answer generation prompt.

Usage:
    from skill_evolution.bootstrap import bootstrap_examples

    examples = bootstrap_examples(eval_results_path, min_score=0.9, max_per_skill=3)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def bootstrap_examples(
    eval_results_path: Path,
    min_score: float = 0.9,
    max_per_skill: int = 3,
) -> dict[str, list[dict[str, Any]]]:
    """Extract successful QA pairs as few-shot examples, grouped by intent.

    Args:
        eval_results_path: Path to eval_results.json
        min_score: Minimum score to qualify as a "success" worth bootstrapping
        max_per_skill: Maximum examples per intent/skill

    Returns:
        Dict mapping intent → list of example dicts with:
          {question, system_answer, ground_truth, evidence_ids, score}
    """
    if not eval_results_path.exists():
        return {}

    data = json.loads(eval_results_path.read_text("utf-8"))
    results = data.get("results", [])

    by_intent: dict[str, list[dict[str, Any]]] = {}

    for r in results:
        score = r.get("judge_score", r.get("score", 0.0))
        if score < min_score:
            continue

        intent = r.get("intent", "unknown")
        by_intent.setdefault(intent, []).append({
            "question": r.get("question", ""),
            "system_answer": r.get("system_answer", ""),
            "ground_truth": r.get("ground_truth", ""),
            "evidence_ids": r.get("evidence_ids", [])[:5],
            "score": score,
            "judge_reasoning": r.get("judge_reasoning", ""),
        })

    # Cap per skill
    for intent in by_intent:
        by_intent[intent] = by_intent[intent][:max_per_skill]

    return by_intent


def format_examples_for_prompt(
    examples: dict[str, list[dict[str, Any]]],
    skill_name: str = "",
    max_examples: int = 3,
) -> list[tuple[str, str]]:
    """Format bootstrapped examples as (user_msg, assistant_msg) pairs.

    This produces the same format as ANSWER_GENERATION_EXAMPLES in
    src/solver/prompts.py, suitable for injection into the prompt registry.
    """
    pairs: list[tuple[str, str]] = []

    examples_for_skill = examples.get(skill_name, [])
    for ex in examples_for_skill[:max_examples]:
        user_msg = (
            f"QUESTION: {ex['question']}\n\n"
            f"EVIDENCE (top items):\n"
            + "\n".join(f"[{i+1}] {eid}" for i, eid in enumerate(ex['evidence_ids']))
        )
        assistant_msg = json.dumps({
            "answer": ex["system_answer"],
            "confidence": ex["score"],
            "uncertainty": None,
        }, ensure_ascii=False)
        pairs.append((user_msg, assistant_msg))

    return pairs


def bootstrap_summary(
    examples: dict[str, list[dict[str, Any]]],
) -> str:
    """Human-readable summary of bootstrapped examples."""
    lines = ["## Bootstrapped Few-Shot Examples\n"]
    total = 0
    for intent, exs in sorted(examples.items()):
        lines.append(f"- **{intent}**: {len(exs)} examples")
        total += len(exs)
        for ex in exs[:2]:
            lines.append(f"  - Q: {ex['question'][:80]}... (score={ex['score']:.2f})")
    lines.append(f"\nTotal: {total} examples across {len(examples)} intents")
    return "\n".join(lines)
