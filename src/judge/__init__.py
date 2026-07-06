"""LLM Judge module for answer scoring.

Extracted from eval_questions.py for reuse across experiments.

Usage:
    from judge import judge_answer, JUDGE_PROMPT

    result = judge_answer(system_answer, ground_truth, key_entities=["DCE", "yield"])
    # {"score": 0.8, "key_entities_found": ["DCE"], "key_entities_missing": ["yield"], "reasoning": "..."}
"""

from .llm_judge import judge_answer, judge_batch, JUDGE_PROMPT
