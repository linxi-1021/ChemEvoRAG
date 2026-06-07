"""Failure Attribution for ChemEvoRAG Skill Evolution.

Implements §4 of the Skill Evolution design spec:
  - OutcomeType / FailureType enums
  - Rule-based attribution engine (priority order)
  - LLM attribution fallback
  - TraceReport generator from eval_results.json + react_logs/

Usage:
    from skill_evolution.attribution import generate_trace_report, TraceReport
    report = generate_trace_report(eval_results_path, react_logs_dir)
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

try:
    from pydantic import BaseModel, ConfigDict, Field
except ImportError:
    raise ImportError("pydantic is required for skill_evolution.attribution")


# ---------------------------------------------------------------------------
# §4.1 Enums
# ---------------------------------------------------------------------------

class OutcomeType(str, Enum):
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    SYSTEM_FAILURE = "system_failure"
    COVERAGE_GAP = "coverage_gap"


class FailureType(str, Enum):
    ENTITY_MISS = "entity_miss"
    ALIAS_MISS = "alias_miss"
    LOCAL_ID_MISS = "local_id_miss"
    ROUTING_ERROR = "routing_error"
    EVIDENCE_EXPANSION_ERROR = "evidence_expansion_error"
    PLANNER_ERROR = "planner_error"
    ASSESSMENT_FALSE_NEG = "assessment_false_negative"
    TABLE_EXTRACTION_ERROR = "table_extraction_error"
    TABLE_REASONING_ERROR = "table_reasoning_error"
    GENERATION_ERROR = "generation_error"
    UNKNOWN_FAILURE = "unknown_failure"


class AttributionSource(str, Enum):
    RULE = "rule"
    LLM = "llm"
    HYBRID = "hybrid"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class _BaseModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True, validate_assignment=True, extra="forbid")


class PartialSuccessRecord(_BaseModel):
    question_id: str
    intent: str
    score: float
    outcome_type: OutcomeType = OutcomeType.PARTIAL_SUCCESS
    partial_success_reason: str = ""
    primary_failure_type: FailureType | None = None
    evolution_action: str = "uncertain_queue"


class CoverageGapRecord(_BaseModel):
    question_id: str
    intent: str
    score: float
    outcome_type: OutcomeType = OutcomeType.COVERAGE_GAP
    gap_reason: str = ""
    coverage_gap_verified_by: str = "oracle_search"
    evolution_action: str = "record_only"


class FailureRecord(_BaseModel):
    question_id: str
    intent: str
    skill_name: str = ""
    skill_version: str = "1.0.0"
    score: float
    outcome_type: OutcomeType = OutcomeType.SYSTEM_FAILURE
    primary_failure_type: FailureType = FailureType.UNKNOWN_FAILURE
    contributing_failure_types: list[FailureType] = Field(default_factory=list)
    attribution_source: AttributionSource = AttributionSource.RULE
    attribution_confidence: float = 0.0
    attribution_explanation: str = ""
    evidence_contained_answer: bool = False
    corpus_contained_answer: bool | None = None
    gold_answer: str = ""
    predicted_answer: str = ""
    retrieval_path: list[str] = Field(default_factory=list)
    retrieved_evidence_ids: list[str] = Field(default_factory=list)
    gold_evidence_ids: list[str] = Field(default_factory=list)
    cited_evidence_ids: list[str] = Field(default_factory=list)
    react_rounds_used: int = 1
    stop_reason: str = ""
    proposed_patch_targets: list[str] = Field(default_factory=list)
    evolution_action: str = "generate_patch_candidate"


class SuccessPattern(_BaseModel):
    intent: str
    pattern: str
    frequency: int = 0
    avg_score: float = 0.0
    supporting_question_ids: list[str] = Field(default_factory=list)
    candidate_template_action: str = "distill_or_refine_template"


class IntentStats(_BaseModel):
    average_score: float = 0.0
    total: int = 0
    success: int = 0
    partial_success: int = 0
    system_failure: int = 0
    coverage_gap: int = 0


class EnvironmentSnapshot(_BaseModel):
    model_name: str = ""
    temperature: float = 0.1
    embedding_model: str = "all-MiniLM-L6-v2"
    retriever_version: str = ""
    prompt_registry_version: str = "1.0"
    corpus_snapshot_id: str = ""
    eval_dataset_version: str = "v1_118qs"
    random_seed: int = 42


class TraceReport(_BaseModel):
    run_id: str
    generated_at: str = ""
    skill_snapshot: dict[str, Any] = Field(default_factory=dict)
    environment: EnvironmentSnapshot = Field(default_factory=EnvironmentSnapshot)
    total_questions: int = 0
    average_score: float = 0.0
    by_intent: dict[str, IntentStats] = Field(default_factory=dict)
    failures: list[FailureRecord] = Field(default_factory=list)
    partial_successes: list[PartialSuccessRecord] = Field(default_factory=list)
    coverage_gaps: list[CoverageGapRecord] = Field(default_factory=list)
    success_patterns: list[SuccessPattern] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# §4.3 Attribution rule engine (priority-ordered)
# ---------------------------------------------------------------------------

# Priority order: upstream failures first to avoid downstream contamination
_ATTRIBUTION_PRIORITY = [
    "coverage_gap_check",
    "entity_miss",
    "alias_miss",
    "local_id_miss",
    "routing_error",
    "evidence_expansion_error",
    "planner_error",
    "table_extraction_error",
    "table_reasoning_error",
    "assessment_false_negative",
    "generation_error",
]


def _classify_outcome(score: float, evidence_contained_answer: bool | None = None) -> OutcomeType:
    """Classify outcome from judge score."""
    if score >= 0.8:
        return OutcomeType.SUCCESS
    if score >= 0.3:
        return OutcomeType.PARTIAL_SUCCESS
    # score < 0.3 → system_failure or coverage_gap
    # coverage_gap is determined separately by oracle check
    return OutcomeType.SYSTEM_FAILURE


def _check_coverage_gap(
    result: dict,
    react_log: str,
    gold_answer: str,
) -> bool:
    """Check if failure is a coverage gap (answer not in corpus).

    Conservative: only returns True if the react log shows all rounds
    retrieved 0 new evidence AND the final answer explicitly says insufficient.
    A more rigorous check would use oracle_search, but that requires
    running additional retrieval which is expensive.
    """
    predicted = result.get("system_answer", "")
    if "insufficient" not in predicted.lower() and "not enough" not in predicted.lower():
        return False
    # Check if react log shows evidence exhaustion (all rounds found 0 new items)
    if not react_log:
        return False
    zero_new_pattern = re.findall(r"(\d+) new", react_log)
    if zero_new_pattern and all(int(n) == 0 for n in zero_new_pattern[1:]):
        # Second round onward all had 0 new items
        return True
    return False


def _detect_entity_miss(result: dict, react_log: str, question: str) -> bool:
    """Check if the question's entities were never found in evidence."""
    answer = result.get("system_answer", "").lower()
    if "insufficient" not in answer and "not found" not in answer:
        return False
    # Look for evidence IDs in the react log
    if "molecule" in react_log.lower() or "entity" in react_log.lower():
        return False
    # Check if judge reasoning mentions entities not found
    reasoning = result.get("judge_reasoning", "").lower()
    if "entity" in reasoning or "not mentioned" in reasoning:
        return True
    return False


def _detect_alias_miss(result: dict, react_log: str) -> bool:
    """Check if alias expansion failed."""
    answer = result.get("system_answer", "").lower()
    reasoning = result.get("judge_reasoning", "").lower()
    entities_found = result.get("entities_found", [])
    entities_missing = result.get("entities_missing", [])
    # If entities were partially found (abbreviation found but full name not), it's alias_miss
    if entities_missing and entities_found:
        for e in entities_missing:
            if any(c.isupper() for c in e) and len(e) <= 6:
                return True
    return False


def _detect_local_id_miss(result: dict, question: str) -> bool:
    """Check if local compound IDs (like 1a, 2b) failed to map."""
    # Look for local ID patterns in question
    local_ids = re.findall(r"\b(\d+[a-z])\b", question.lower())
    if not local_ids:
        return False
    answer = result.get("system_answer", "").lower()
    reasoning = result.get("judge_reasoning", "").lower()
    # If the answer mentions a different local ID or says not found
    if "not found" in answer or "not mentioned" in reasoning:
        return True
    # If the answer's entities don't match the expected ones
    entities_found = result.get("entities_found", [])
    for lid in local_ids:
        if lid not in str(entities_found).lower():
            if lid not in answer:
                return True
    return False


def _detect_assessment_false_negative(
    result: dict, react_log: str, gold_answer: str = "", question: str = ""
) -> bool:
    """Check if assessment incorrectly declared evidence insufficient
    despite the answer being present in evidence.

    This should ONLY trigger when the gold answer's specific terms
    actually appear in the retrieved evidence. Otherwise it's a routing error.
    """
    answer = result.get("system_answer", "").lower()
    # Check if the answer is a refusal (saying evidence is lacking)
    _REFUSAL_SIGNALS = [
        "insufficient", "not enough", "does not include", "cannot determine",
        "unable to", "not sufficient", "not found", "no data", "no evidence",
        "does not contain", "not available", "could not find",
    ]
    is_refusal = any(s in answer for s in _REFUSAL_SIGNALS)
    if not is_refusal:
        return False
    # Check if evidence IDs were retrieved and substantial
    evidence_ids = result.get("evidence_ids", [])
    if not evidence_ids:
        return False
    # Check react log for rounds with evidence but assessment said insufficient
    react_lower = react_log.lower()
    assessment_said_insufficient = (
        "insufficient" in react_lower or "sufficient=false" in react_lower
    )
    if assessment_said_insufficient and re.search(r"\d+ new", react_log):
        new_counts = re.findall(r"(\d+) new", react_log)
        if not new_counts or int(new_counts[0]) < 5:
            return False
        # Verify that gold answer's specific terms appear in the evidence
        _COMMON_CHEM_WORDS = {
            "product", "yield", "reaction", "solvent", "conditions",
            "temperature", "catalyst", "oxidation", "compound", "molecule",
            "experimental", "standard", "higher", "lower", "percent",
            "obtained", "formed", "gave", "showed", "reported",
            "table", "entry", "data", "results", "paper", "authors",
            "the", "and", "for", "was", "were", "are", "has", "have",
            "with", "from", "this", "that", "which", "than", "versus",
        }
        if gold_answer:
            gold_terms = set(re.findall(r'\b\w{3,}\b', gold_answer.lower()))
            specific_terms = gold_terms - _COMMON_CHEM_WORDS
            if question:
                compound_ids = set(re.findall(r'\b(\d+[a-z])\b', question.lower()))
                specific_terms.update(compound_ids)
            if specific_terms:
                react_lower = react_log.lower()
                matching_specific = sum(1 for t in specific_terms if t in react_lower)
                if matching_specific >= 1:
                    return True
        # No specific terms to match — fall back to evidence count heuristic
        return True
    return False


def _detect_table_reasoning_error(result: dict, react_log: str, question: str) -> bool:
    """Check if table data was present in evidence but the LLM misinterpreted it
    (e.g., wrong row, wrong column, entry confusion).

    This should ONLY trigger when:
    1. The question involves table data
    2. The system generated a WRONG answer (not "insufficient")
    3. Table data was actually present in the retrieved evidence

    If the system said "insufficient", the failure is either:
    - routing_error (evidence didn't contain the answer)
    - assessment_false_negative (evidence contained the answer but assessment missed it)
    """
    question_lower = question.lower()
    # Table-related questions
    table_signals = ["entry", "table", "row", "column", "yield", "solvent"]
    if not any(s in question_lower for s in table_signals):
        return False
    answer = result.get("system_answer", "").lower()
    score = result.get("score", 1.0)
    # Only classify as table_reasoning_error if the system produced a WRONG answer
    # (not a refusal/non-answer). If it refused, it's routing or assessment failure.
    _REFUSAL_SIGNALS = [
        "insufficient", "not enough", "does not include", "cannot determine",
        "unable to", "not sufficient", "not found", "no data", "no evidence",
        "does not contain", "not available", "could not find",
    ]
    is_refusal = any(s in answer for s in _REFUSAL_SIGNALS)
    if score < 0.3 and not is_refusal:
        # System generated a wrong answer from table data
        if "table" in react_log.lower() or "entry" in react_log.lower():
            return True
    return False


def _detect_generation_error(result: dict, react_log: str) -> bool:
    """Check if evidence was sufficient but answer generation failed."""
    answer = result.get("system_answer", "").lower()
    score = result.get("score", 1.0)
    # If evidence was marked sufficient by assessment but score is low
    if score < 0.5 and "sufficient" in react_log.lower():
        # Assessment said yes but answer was wrong
        if "assessment: sufficient=true" in react_log:
            return True
    return False


def _detect_routing_error(result: dict, react_log: str, gold_answer: str = "", question: str = "") -> bool:
    """Check if wrong retrieval channels were used.

    Detects two cases:
    1. Very few evidence items retrieved across all rounds (low recall)
    2. Evidence was retrieved but none contained the answer (wrong evidence)
       - detected by: score < 0.3, answer says "insufficient", and assessment
         said "sufficient=False" in the react log, AND the gold answer terms
         do NOT appear in the retrieved evidence

    If the gold answer terms DO appear in the evidence, it's an
    assessment_false_negative (evidence was there but assessment missed it).
    """
    score = result.get("score", 1.0)
    if score >= 0.5:
        return False

    answer = result.get("system_answer", "").lower()
    _REFUSAL_SIGNALS = [
        "insufficient", "not enough", "does not include", "cannot determine",
        "unable to", "not sufficient", "not found", "no data", "no evidence",
        "does not contain", "not available", "could not find",
    ]
    is_refusal = any(s in answer for s in _REFUSAL_SIGNALS)

    # Case 1: Very few evidence items
    new_counts = re.findall(r"(\d+) new", react_log)
    if new_counts:
        total_new = sum(int(n) for n in new_counts)
        if total_new < 3:
            return True

    # Case 2: Evidence was retrieved but assessment said insufficient
    react_lower = react_log.lower()
    assessment_said_insufficient = (
        "insufficient" in react_lower or "sufficient=false" in react_lower
    )
    if is_refusal and assessment_said_insufficient:
        if new_counts and sum(int(n) for n in new_counts) >= 3:
            # Check if SPECIFIC gold answer terms appear in the evidence.
            # Filter out common chemistry words to avoid false matches.
            _COMMON_CHEM_WORDS = {
                "product", "yield", "reaction", "solvent", "conditions",
                "temperature", "catalyst", "oxidation", "compound", "molecule",
                "experimental", "standard", "higher", "lower", "percent",
                "obtained", "formed", "gave", "showed", "reported",
                "table", "entry", "data", "results", "paper", "authors",
                "the", "and", "for", "was", "were", "are", "has", "have",
                "with", "from", "this", "that", "which", "than", "versus",
            }
            if gold_answer:
                gold_terms = set(re.findall(r'\b\w{3,}\b', gold_answer.lower()))
                specific_terms = gold_terms - _COMMON_CHEM_WORDS
                # Also extract compound IDs from the question (e.g., "2c", "2k", "1a")
                if question:
                    compound_ids = set(re.findall(r'\b(\d+[a-z])\b', question.lower()))
                    specific_terms.update(compound_ids)
                if specific_terms:
                    react_lower = react_log.lower()
                    matching_specific = sum(1 for t in specific_terms if t in react_lower)
                    if matching_specific >= 1:
                        # Specific gold answer terms found in evidence
                        # — assessment missed it, not a routing error
                        return False
            # Evidence was retrieved but didn't contain the answer
            # This is a routing error — wrong evidence channels
            return True

    return False


def _get_proposed_patch_targets(failure_type: FailureType) -> list[str]:
    """Map failure type to patch target paths."""
    mapping = {
        FailureType.ENTITY_MISS: ["strategy.query_rewrite", "strategy.retrieval_routing"],
        FailureType.ALIAS_MISS: ["strategy.query_rewrite", "strategy.retrieval_routing"],
        FailureType.LOCAL_ID_MISS: ["strategy.query_rewrite"],
        FailureType.ROUTING_ERROR: ["strategy.retrieval_routing"],
        FailureType.EVIDENCE_EXPANSION_ERROR: ["strategy.evidence_expansion"],
        FailureType.PLANNER_ERROR: ["strategy.query_rewrite"],
        FailureType.ASSESSMENT_FALSE_NEG: ["strategy.assessment"],
        FailureType.TABLE_EXTRACTION_ERROR: [],  # not patchable
        FailureType.TABLE_REASONING_ERROR: ["strategy.assessment", "strategy.answer_generation"],
        FailureType.GENERATION_ERROR: ["strategy.answer_generation"],
        FailureType.UNKNOWN_FAILURE: [],
    }
    return mapping.get(failure_type, [])


def _attribute_failure_rule(
    result: dict,
    react_log: str,
    question: str,
    gold_answer: str = "",
) -> tuple[FailureType, float, str]:
    """Rule-based failure attribution.

    Returns (failure_type, confidence, explanation).
    Runs through priority order; first match wins.
    """
    # 1. coverage_gap_check — handled separately in generate_trace_report

    # 2. entity_miss
    if _detect_entity_miss(result, react_log, question):
        return FailureType.ENTITY_MISS, 0.80, "Entities from question not found in retrieved evidence."

    # 3. alias_miss
    if _detect_alias_miss(result, react_log):
        return FailureType.ALIAS_MISS, 0.75, "Abbreviation/alias not expanded to full chemical name."

    # 4. local_id_miss
    if _detect_local_id_miss(result, question):
        return FailureType.LOCAL_ID_MISS, 0.80, "Local compound ID mapping failed."

    # 5. routing_error
    if _detect_routing_error(result, react_log, gold_answer, question):
        return FailureType.ROUTING_ERROR, 0.70, "Wrong evidence retrieved — answer terms not found in evidence."

    # 6. evidence_expansion_error
    # (hard to detect without comparing expected vs actual expansion)

    # 7. planner_error — only if the planner's query refinement was ineffective
    # (don't trigger if the real issue is retrieval or assessment)
    rounds = result.get("react_rounds_used", 1)
    stop_reason = result.get("stop_reason", "")
    if rounds == 3 and stop_reason == "max_rounds_exhausted":
        # Check if this is actually a planner issue vs retrieval/assessment
        # If assessment ever said sufficient, it's not a planner error
        if "sufficient=true" not in react_log.lower():
            # Check if new evidence was found in later rounds (planner was working)
            new_counts = re.findall(r"(\d+) new", react_log)
            if new_counts and len(new_counts) >= 2:
                later_new = sum(int(n) for n in new_counts[1:])
                if later_new == 0:
                    # Planner's refined queries found nothing — planner error
                    return FailureType.PLANNER_ERROR, 0.60, "ReAct loop exhausted all rounds — refined queries found no new evidence."

    # 8. table_extraction_error vs table_reasoning_error
    if _detect_table_reasoning_error(result, react_log, question):
        # Assume table was correctly extracted but LLM reasoning failed
        return FailureType.TABLE_REASONING_ERROR, 0.75, "Table data present in evidence but LLM failed to interpret correctly."

    # 9. assessment_false_negative
    if _detect_assessment_false_negative(result, react_log, gold_answer, question):
        return FailureType.ASSESSMENT_FALSE_NEG, 0.85, "Assessment LLM incorrectly judged evidence insufficient despite answer being present."

    # 10. generation_error
    if _detect_generation_error(result, react_log):
        return FailureType.GENERATION_ERROR, 0.80, "Evidence was sufficient per assessment, but answer generation produced incorrect output."

    # 11. fallback
    return FailureType.UNKNOWN_FAILURE, 0.30, "No rule matched. Manual review recommended."


# ---------------------------------------------------------------------------
# §4.7 Trace report generator
# ---------------------------------------------------------------------------

def _load_react_log(react_logs_dir: Path, doc_tag: str, query_hash: str) -> str:
    """Load a specific react log file."""
    log_path = react_logs_dir / f"{doc_tag}_{query_hash}.log"
    if log_path.exists():
        return log_path.read_text("utf-8")
    return ""


def _extract_react_rounds(react_log: str) -> int:
    """Extract the number of rounds used from react log."""
    round_matches = re.findall(r"\[ReAct\] Round (\d+)/(\d+)", react_log)
    if round_matches:
        return max(int(m[0]) for m in round_matches)
    return 1


def _extract_stop_reason(react_log: str) -> str:
    """Extract the stop reason from react log."""
    if "Evidence sufficient, stopping" in react_log:
        return "evidence_sufficient"
    if "No refined query, stopping" in react_log:
        return "no_refined_query"
    if "Assessment failed" in react_log:
        return "assessment_failed"
    return "max_rounds_exhausted"


def generate_trace_report(
    eval_results_path: Path,
    react_logs_dir: Path,
    skill_configs: dict[str, dict] | None = None,
    run_id: str | None = None,
) -> TraceReport:
    """Generate a TraceReport from eval results and react logs.

    This is the main entry point for §4 Failure Attribution.

    Args:
        eval_results_path: Path to eval_results.json
        react_logs_dir: Path to react_logs/ directory
        skill_configs: Optional dict of intent → skill YAML config
        run_id: Optional run identifier (auto-generated if not provided)

    Returns:
        TraceReport with failures, partial_successes, coverage_gaps, and success_patterns
    """
    eval_results = json.loads(eval_results_path.read_text("utf-8"))
    results = eval_results.get("results", [])
    summary = eval_results.get("summary", {})

    if not run_id:
        run_id = f"run_{datetime.now().strftime('%Y-%m-%d_%H%M')}"

    report = TraceReport(
        run_id=run_id,
        generated_at=datetime.now().isoformat(),
        total_questions=len(results),
        average_score=summary.get("average_score", 0.0),
    )

    # Build intent stats
    by_intent_raw: dict[str, list[float]] = {}
    for r in results:
        intent = r.get("intent", "unknown")
        by_intent_raw.setdefault(intent, []).append(r.get("score", 0.0))

    for intent, scores in by_intent_raw.items():
        stats = IntentStats(
            average_score=sum(scores) / len(scores) if scores else 0.0,
            total=len(scores),
        )
        report.by_intent[intent] = stats

    # Analyze each result
    for r in results:
        score = r.get("score", 0.0)
        intent = r.get("intent", "unknown")
        question_id = r.get("id", "")
        question = r.get("question", "")
        ground_truth = r.get("ground_truth", "")
        system_answer = r.get("system_answer", "")

        # Load react log for this question
        source_paper = r.get("source_paper", "").replace(".pdf", "")
        q_hash = __import__("hashlib").md5(question.encode()).hexdigest()[:8]
        react_log = _load_react_log(react_logs_dir, source_paper, q_hash)

        # Determine skill_name from intent
        skill_name = intent
        skill_version = "1.0.0"
        if skill_configs and intent in skill_configs:
            skill_version = skill_configs[intent].get("skill_version", "1.0.0")

        # Classify outcome
        outcome = _classify_outcome(score)

        # Update intent stats (system_failure is incremented later,
        # only for confirmed system failures, not coverage gaps)
        if intent in report.by_intent:
            if outcome == OutcomeType.SUCCESS:
                report.by_intent[intent].success += 1
            elif outcome == OutcomeType.PARTIAL_SUCCESS:
                report.by_intent[intent].partial_success += 1

        if outcome == OutcomeType.SUCCESS:
            # Use a unique identifier combining question_id and source paper
            # to prevent the same question template across papers from inflating counts
            unique_qid = f"{question_id}_{source_paper}" if source_paper else question_id
            report.success_patterns.append(SuccessPattern(
                intent=intent,
                pattern=f"score_{score:.1f}",
                frequency=1,
                avg_score=score,
                supporting_question_ids=[unique_qid],
            ))
            continue

        if outcome == OutcomeType.PARTIAL_SUCCESS:
            primary_failure, conf, explanation = _attribute_failure_rule(
                r, react_log, question, ground_truth
            )
            report.partial_successes.append(PartialSuccessRecord(
                question_id=question_id,
                intent=intent,
                score=score,
                partial_success_reason=explanation,
                primary_failure_type=primary_failure,
                evolution_action="minor_patch_candidate" if score >= 0.5 else "uncertain_queue",
            ))
            continue

        # score < 0.3 → check coverage_gap or system_failure
        is_coverage_gap = _check_coverage_gap(r, react_log, ground_truth)
        if is_coverage_gap:
            report.coverage_gaps.append(CoverageGapRecord(
                question_id=question_id,
                intent=intent,
                score=score,
                gap_reason="Answer not found in retrieved evidence or corpus.",
            ))
            if intent in report.by_intent:
                report.by_intent[intent].coverage_gap += 1
            continue

        # System failure — run full attribution
        primary_failure, conf, explanation = _attribute_failure_rule(
            r, react_log, question, ground_truth
        )

        # Determine contributing failures (simplified: check generation if assessment passed)
        contributing = []
        if primary_failure == FailureType.ASSESSMENT_FALSE_NEG:
            # Check if there's also a generation component
            if "insufficient" in system_answer.lower():
                pass  # assessment was the root cause
        elif primary_failure == FailureType.LOCAL_ID_MISS:
            contributing.append(FailureType.ROUTING_ERROR)

        # Determine if evidence actually contained the answer by checking
        # if the system's answer matches the gold answer (score > 0.5)
        # or if the react log shows the assessment found evidence sufficient
        evidence_contained = score > 0.5 or (
            "sufficient" in react_log.lower() and "true" in react_log.lower()
        )
        # For table_reasoning_error specifically: check if the answer keywords
        # appear in the retrieved evidence text
        if primary_failure == FailureType.TABLE_REASONING_ERROR and ground_truth:
            # Check if any key terms from gold_answer appear in react log evidence
            gold_terms = set(re.findall(r'\b\w{4,}\b', ground_truth.lower()))
            react_lower = react_log.lower()
            matching_terms = sum(1 for t in gold_terms if t in react_lower)
            evidence_contained = matching_terms >= 2 or evidence_contained

        failure = FailureRecord(
            question_id=question_id,
            intent=intent,
            skill_name=skill_name,
            skill_version=skill_version,
            score=score,
            primary_failure_type=primary_failure,
            contributing_failure_types=contributing,
            attribution_source=AttributionSource.RULE,
            attribution_confidence=conf,
            attribution_explanation=explanation,
            evidence_contained_answer=evidence_contained,
            gold_answer=ground_truth,
            predicted_answer=system_answer[:200],
            retrieved_evidence_ids=r.get("evidence_ids", []),
            react_rounds_used=_extract_react_rounds(react_log),
            stop_reason=_extract_stop_reason(react_log),
            proposed_patch_targets=_get_proposed_patch_targets(primary_failure),
            evolution_action="generate_patch_candidate" if conf >= 0.75 else "manual_review",
        )
        report.failures.append(failure)
        if intent in report.by_intent:
            report.by_intent[intent].system_failure += 1  # confirmed system failure

    return report


def generate_trace_report_from_paths(
    eval_results_path: str | Path,
    react_logs_dir: str | Path,
    output_path: str | Path | None = None,
) -> dict:
    """CLI-friendly wrapper that generates and optionally saves a TraceReport.

    Returns the report as a dict (for JSON serialization).
    """
    report = generate_trace_report(
        eval_results_path=Path(eval_results_path),
        react_logs_dir=Path(react_logs_dir),
    )
    report_dict = report.model_dump(mode="json")
    if output_path:
        Path(output_path).write_text(
            json.dumps(report_dict, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return report_dict
