"""Result evaluation. Stage 2 of the Skill Evolution pipeline.

Responsibilities:
  - Classify each trace as success / partial_success / system_failure / coverage_gap
  - Detect coverage gaps (answer missing from corpus)
  - Route each outcome to the correct downstream path

Coverage gaps do NOT enter skill patch generation.
They are written to data/evolution/backlog/ for corpus/index improvement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .trace import StandardTrace


# ── Outcome types ────────────────────────────────────────────────────────

class OutcomeType(str, Enum):
    SUCCESS = "success"               # score ≥ 0.8 — enters 4B distillation
    PARTIAL_SUCCESS = "partial"       # 0.3 ≤ score < 0.8 — queued for review
    SYSTEM_FAILURE = "failure"        # score < 0.3 (not coverage gap) — enters 4A mutation
    COVERAGE_GAP = "coverage_gap"     # corpus doesn't contain answer — backlog only


@dataclass
class OutcomeResult:
    trace_id: str
    outcome_type: OutcomeType
    action: str = ""  # record_to_backlog | generate_patch | queue_for_review | distill_template
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Public API ───────────────────────────────────────────────────────────

def evaluate_outcome(trace: StandardTrace) -> OutcomeResult:
    """Classify a trace into one of four outcome types.

    Decision tree:
      score >= 0.8        → SUCCESS
      0.3 <= score < 0.8  → PARTIAL_SUCCESS
      score < 0.3
        ├── is_coverage_gap → COVERAGE_GAP
        └── otherwise       → SYSTEM_FAILURE
    """
    if trace.score >= 0.8:
        return OutcomeResult(
            trace_id=trace.question_id,
            outcome_type=OutcomeType.SUCCESS,
            action="distill_template",
            score=trace.score,
            metadata={"intent": trace.intent},
        )

    if trace.score >= 0.3:
        return OutcomeResult(
            trace_id=trace.question_id,
            outcome_type=OutcomeType.PARTIAL_SUCCESS,
            action=_partial_action(trace.score),
            score=trace.score,
            metadata={"intent": trace.intent},
        )

    # score < 0.3
    if _is_coverage_gap(trace):
        return OutcomeResult(
            trace_id=trace.question_id,
            outcome_type=OutcomeType.COVERAGE_GAP,
            action="record_to_backlog",
            score=trace.score,
            metadata={
                "intent": trace.intent,
                "question": trace.question,
                "gold_answer": trace.gold_answer,
                "missing_entities": _extract_missing_entities(trace),
                "corpus_search_hint": " ".join(trace.question.split()[:20]),
                "source_paper": trace.source_paper,
            },
        )

    return OutcomeResult(
        trace_id=trace.question_id,
        outcome_type=OutcomeType.SYSTEM_FAILURE,
        action="generate_patch",
        score=trace.score,
        metadata={"intent": trace.intent},
    )


def is_tool_error(trace: StandardTrace) -> bool:
    """Check if this failure was caused by a tool error (RDKit, Neo4j, parser)."""
    answer_lower = trace.answer.lower()
    tool_error_signals = [
        "rdkit error", "rdkit could not", "rdkit failed",
        "neo4j error", "neo4j connection", "neo4j timeout",
        "parsing error", "parse error", "could not parse",
        "elementkg error", "elementkg timeout",
    ]
    return any(sig in answer_lower for sig in tool_error_signals)


# ── Internal helpers ─────────────────────────────────────────────────────

def _is_coverage_gap(trace: StandardTrace) -> bool:
    """Determine if this failure is a coverage gap (answer absent from corpus).

    Conservative detection:
    1. All rounds found 0 new evidence (evidence exhaustion)
    2. Final answer says "insufficient" / "not found"
    3. Assessment explicitly marked evidence insufficient
    """
    answer_lower = trace.answer.lower()
    _INSUFFICIENT_SIGNALS = [
        "insufficient", "not enough", "does not include",
        "cannot determine", "unable to", "not sufficient",
        "not found", "no data", "no evidence", "does not contain",
    ]

    is_refusal = any(s in answer_lower for s in _INSUFFICIENT_SIGNALS)
    if not is_refusal:
        return False

    # Check if all rounds exhausted (no new evidence)
    if trace.retrieval_rounds:
        # If round 1 found evidence, it's less likely a coverage gap
        first_round_new = trace.retrieval_rounds[0].new_evidence_count if trace.retrieval_rounds else 0
        if first_round_new >= 5:
            return False
        # All subsequent rounds found nothing
        later_rounds = trace.retrieval_rounds[1:]
        if later_rounds and all(rr.new_evidence_count == 0 for rr in later_rounds):
            return True

    # Assessment marked insufficient
    if "sufficient=false" in trace.assessment_result.lower():
        return True

    return False


def _partial_action(score: float) -> str:
    """Determine action for a partially successful result."""
    if score >= 0.7:
        return "minor_patch_candidate"
    elif score >= 0.5:
        return "uncertain_queue"
    else:
        return "review_next_round"


def _extract_missing_entities(trace: StandardTrace) -> list[str]:
    """Extract entities that appear in gold answer but not in retrieved evidence."""
    # Simple heuristic: words >= 4 chars in gold answer not found in evidence text
    import re as _re
    gold_words = set(_re.findall(r'\b\w{4,}\b', trace.gold_answer.lower()))
    evidence_text = " ".join(
        ev.summary for rr in trace.retrieval_rounds for ev in rr.retrieved_evidence
    ).lower()
    # Filter common chemistry stopwords
    _stopwords = {
        "with", "from", "this", "that", "which", "than", "were", "have",
        "product", "yield", "reaction", "solvent", "conditions", "table",
        "entry", "data", "results", "paper", "authors", "obtained",
        "formed", "showed", "reported",
    }
    return sorted([w for w in gold_words if w not in evidence_text and w not in _stopwords])
