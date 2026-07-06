"""Trace standardization and field audit. Stage 1 of the Skill Evolution pipeline.

Responsibilities:
  - standardize_trace(): Convert eval_results.json + react_logs into a uniform StandardTrace
  - audit_trace_fields(): Check which required fields are available
  - Does NOT perform attribution or outcome classification (those belong to evaluation.py / attribution.py)
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ── Data models ──────────────────────────────────────────────────────────

@dataclass
class RetrievedEvidence:
    evidence_id: str
    evidence_type: str        # document_block, table_block, reaction_event, molecule, etc.
    summary: str = ""
    channel: str = ""         # which retrieval channel found this
    score: float = 0.0
    contains_gold_answer: bool = False
    used_in_final_answer: bool = False


@dataclass
class RetrievalRound:
    round: int
    query: str = ""
    refined_query: str = ""
    retrieval_actions: list[str] = field(default_factory=list)
    retrieved_evidence: list[RetrievedEvidence] = field(default_factory=list)
    new_evidence_count: int = 0
    total_evidence: int = 0
    new_evidence_found: bool = False
    skipped_due_to_similarity: bool = False
    similarity_to_previous_query: float = 1.0


@dataclass
class StandardTrace:
    question_id: str
    question: str = ""
    intent: str = "unknown"
    entities: list[str] = field(default_factory=list)
    retrieval_rounds: list[RetrievalRound] = field(default_factory=list)
    evidence_items: list[str] = field(default_factory=list)
    answer: str = ""
    gold_answer: str = ""
    score: float = 0.0
    stop_reason: str = ""
    eval_metrics: dict[str, Any] = field(default_factory=dict)
    feedback: str = ""
    annotations: list[str] = field(default_factory=list)
    source_paper: str = ""
    assessment_result: str = ""
    assessment_reason: str = ""
    react_rounds_used: int = 1


@dataclass
class FieldAudit:
    question_id: str
    field_status: dict[str, str] = field(default_factory=dict)
    missing_fields: list[str] = field(default_factory=list)
    trace_quality: str = "complete"  # complete | incomplete_trace | minimal


# ── Required / strongly needed fields ────────────────────────────────────

_REQUIRED_FIELDS = [
    "question_id", "question", "gold_answer", "answer", "score", "intent",
]

_STRONGLY_NEEDED_FIELDS = [
    "retrieval_rounds", "evidence_items", "react_log", "assessment_result",
    "assessment_reason", "stop_reason",
]


# ── Public API ───────────────────────────────────────────────────────────

def standardize_trace(eval_result: dict, react_log: str) -> StandardTrace:
    """Convert an eval_results.json entry + its react_log into a uniform StandardTrace.

    This is a pure data transformation — no attribution, no classification.
    """
    trace = StandardTrace(
        question_id=eval_result.get("id", ""),
        question=eval_result.get("question", ""),
        intent=eval_result.get("intent", "unknown"),
        entities=eval_result.get("key_entities", []),
        evidence_items=eval_result.get("evidence_ids", []),
        answer=eval_result.get("system_answer", ""),
        gold_answer=eval_result.get("ground_truth", ""),
        score=eval_result.get("score", 0.0),
        stop_reason=eval_result.get("stop_reason", ""),
        eval_metrics=eval_result.get("metrics", {}),
        feedback=eval_result.get("judge_reasoning", ""),
        annotations=eval_result.get("annotations", []),
        source_paper=eval_result.get("source_paper", ""),
        react_rounds_used=eval_result.get("react_rounds_used", 1),
    )

    # Parse react_log for structured retrieval rounds + assessment
    if react_log:
        trace.retrieval_rounds = _parse_retrieval_rounds(react_log)
        trace.assessment_result = _parse_assessment_result(react_log)
        trace.assessment_reason = _parse_assessment_reason(react_log)
        if not trace.stop_reason:
            trace.stop_reason = _parse_stop_reason(react_log)
        trace.react_rounds_used = max(
            trace.react_rounds_used,
            _count_react_rounds(react_log),
        )

    return trace


def audit_trace_fields(trace: StandardTrace) -> FieldAudit:
    """Check which critical fields are present or missing."""
    field_status: dict[str, str] = {}

    # Check required fields
    for f in _REQUIRED_FIELDS:
        val = getattr(trace, f, None)
        field_status[f] = "available" if val else "missing"

    # Check strongly needed fields
    field_status["retrieval_rounds"] = "available" if trace.retrieval_rounds else "missing"
    field_status["evidence_items"] = "available" if trace.evidence_items else "missing"
    field_status["assessment_result"] = "available" if trace.assessment_result else "missing"
    field_status["assessment_reason"] = "available" if trace.assessment_reason else "missing"

    missing = [k for k, v in field_status.items() if v == "missing"]

    # Classify trace quality
    if not missing:
        quality = "complete"
    elif any(f in missing for f in _REQUIRED_FIELDS):
        quality = "minimal"
    else:
        quality = "incomplete_trace"

    return FieldAudit(
        question_id=trace.question_id,
        field_status=field_status,
        missing_fields=missing,
        trace_quality=quality,
    )


def load_react_log(react_logs_dir: Path, doc_tag: str, query_hash: str) -> str:
    """Load a specific react log file from disk."""
    log_path = react_logs_dir / f"{doc_tag}_{query_hash}.log"
    if log_path.exists():
        return log_path.read_text("utf-8")
    return ""


# ── Internal parsers ─────────────────────────────────────────────────────

def _parse_retrieval_rounds(react_log: str) -> list[RetrievalRound]:
    """Parse structured retrieval rounds from a react log."""
    rounds: list[RetrievalRound] = []
    # Split by round markers
    round_blocks = re.split(r"\[ReAct\] Round \d+/\d+", react_log)
    round_numbers = re.findall(r"\[ReAct\] Round (\d+)/(\d+)", react_log)

    # First block before any round is the header — skip it
    for i, (block, (round_num, _)) in enumerate(
        zip(round_blocks[1:], round_numbers), start=1
    ):
        rr = RetrievalRound(round=i)

        # Query
        query_match = re.search(r"\[ReAct\] Query:\s*(.*?)(?:\n|$)", block)
        if query_match:
            rr.query = query_match.group(1).strip()

        # Refined query
        refined_match = re.search(r"\[ReAct\] Refined query:\s*(.*?)(?:\n|$)", block)
        if refined_match:
            rr.refined_query = refined_match.group(1).strip()

        # Retrieved evidence items
        ev_items = re.findall(
            r"\[ReAct\]\s+\[\d+\]\s+(\w+)\s+\|\s+([^|]+?)\s+\|\s+(.*)",
            block,
        )
        for etype, eid, summary in ev_items:
            rr.retrieved_evidence.append(RetrievedEvidence(
                evidence_id=eid.strip(),
                evidence_type=etype.strip(),
                summary=summary.strip()[:300],
            ))

        # Count new evidence
        new_match = re.search(r"(\d+) new", block)
        if new_match:
            rr.new_evidence_count = int(new_match.group(1))

        total_match = re.search(r"\(total:\s*(\d+)\)", block)
        if total_match:
            rr.total_evidence = int(total_match.group(1))

        rr.new_evidence_found = rr.new_evidence_count > 0

        # Skipped due to similarity
        if "skipping" in block.lower() or "SKIP" in block:
            rr.skipped_due_to_similarity = True

        rounds.append(rr)

    return rounds


def _parse_assessment_result(react_log: str) -> str:
    """Extract assessment sufficiency from react log."""
    m = re.search(r"\[ReAct\] Assessment:\s*sufficient=(\w+)", react_log)
    if m:
        return f"sufficient={m.group(1)}"
    return ""


def _parse_assessment_reason(react_log: str) -> str:
    """Extract assessment reason from react log."""
    m = re.search(r"\[ReAct\] Reason:\s*(.*?)(?:\n|$)", react_log)
    if m:
        return m.group(1).strip()[:500]
    return ""


def _parse_stop_reason(react_log: str) -> str:
    """Infer stop reason from react log."""
    if "Evidence sufficient, stopping" in react_log:
        return "evidence_sufficient"
    if "No refined query, stopping" in react_log:
        return "no_refined_query"
    if "Assessment failed" in react_log:
        return "assessment_failed"
    return "max_rounds_exhausted"


def _count_react_rounds(react_log: str) -> int:
    """Count how many rounds were actually executed."""
    matches = re.findall(r"\[ReAct\] Round (\d+)/(\d+)", react_log)
    if matches:
        return max(int(m[0]) for m in matches)
    return 1
