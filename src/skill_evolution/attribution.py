"""Failure Attribution, Success Pattern Mining, and Case Library. Stage 3 + E layer.

Responsibilities:
  - Rule-based candidate attribution (not final judgment)
  - Analysis clusters (cross-skill grouping for LLM analysis)
  - Success pattern mining with structural feature extraction
  - Coverage gap recording to backlog
  - CaseLibrary (failures, successes, gaps, analysis notes)

Principle: 先归因，再进化。规则给出候选结论，LLM 复核，regression 最终裁决。
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

try:
    from pydantic import BaseModel, ConfigDict, Field
except ImportError:
    raise ImportError("pydantic is required for skill_evolution.attribution")

from .trace import StandardTrace, FieldAudit, load_react_log, audit_trace_fields
from .evaluation import OutcomeType, OutcomeResult, evaluate_outcome


# ══════════════════════════════════════════════════════════════════════════
# Enums — aligned with architecture diagram
# ══════════════════════════════════════════════════════════════════════════

class FailureType(str, Enum):
    ENTITY_MISS = "entity_miss"
    ALIAS_MISS = "alias_miss"
    LOCAL_ID_MISS = "local_id_miss"
    TOOL_ERROR = "tool_error"                     # RDKit / Neo4j / parser errors — no patch
    ROUTING_ERROR = "routing_error"
    PLANNER_ERROR = "planner_error"
    GRAPH_EXPANSION_ERROR = "graph_expansion_error"  # evidence expansion / graph neighbor
    ASSESSMENT_FALSE_NEG = "assessment_false_negative"
    TABLE_EXTRACTION_ERROR = "table_extraction_error"
    TABLE_REASONING_ERROR = "table_reasoning_error"
    GENERATION_ERROR = "generation_error"
    UNKNOWN_FAILURE = "unknown_failure"

    @property
    def is_patchable(self) -> bool:
        return self not in (FailureType.TOOL_ERROR, FailureType.UNKNOWN_FAILURE,
                            FailureType.TABLE_EXTRACTION_ERROR)

    @property
    def mutation_dimension(self) -> str:
        """Map failure type to the primary mutation dimension."""
        _map = {
            FailureType.ENTITY_MISS: "planning",
            FailureType.ALIAS_MISS: "planning",
            FailureType.LOCAL_ID_MISS: "planning",
            FailureType.ROUTING_ERROR: "routing",
            FailureType.PLANNER_ERROR: "planning",
            FailureType.GRAPH_EXPANSION_ERROR: "graph_expansion",
            FailureType.ASSESSMENT_FALSE_NEG: "prompt",
            FailureType.TABLE_REASONING_ERROR: "prompt",
            FailureType.GENERATION_ERROR: "prompt",
            FailureType.TOOL_ERROR: "",
            FailureType.UNKNOWN_FAILURE: "",
        }
        return _map.get(self, "")


class AttributionSource(str, Enum):
    RULE = "rule"
    LLM = "llm"
    HYBRID = "hybrid"


# ══════════════════════════════════════════════════════════════════════════
# Pydantic models
# ══════════════════════════════════════════════════════════════════════════

class _BaseModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True, validate_assignment=True, extra="forbid")


class CandidateAttributionEntry(_BaseModel):
    type: str = ""      # e.g. "assessment_false_negative"
    confidence: float = 0.0
    signals: list[str] = Field(default_factory=list)


class CandidateAttribution(_BaseModel):
    question_id: str
    candidate_failure_types: list[CandidateAttributionEntry] = Field(default_factory=list)
    candidate_target_paths: list[str] = Field(default_factory=list)
    candidate_prompt_role: str = ""
    uncertainty_flags: list[str] = Field(default_factory=list)
    needs_llm_review: bool = True


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
    retrieved_evidence_ids: list[str] = Field(default_factory=list)
    gold_evidence_ids: list[str] = Field(default_factory=list)
    cited_evidence_ids: list[str] = Field(default_factory=list)
    react_rounds_used: int = 1
    stop_reason: str = ""
    proposed_patch_targets: list[str] = Field(default_factory=list)
    evolution_action: str = "generate_patch_candidate"
    # Structured data for LLM analysis
    retrieval_rounds: list[dict[str, Any]] = Field(default_factory=list)
    channel_stats: dict[str, int] = Field(default_factory=dict)


class SuccessPattern(_BaseModel):
    intent: str
    pattern: str          # descriptive name: "{query_structure}_{evidence_type}_{answer_structure}"
    frequency: int = 0
    avg_score: float = 0.0
    supporting_question_ids: list[str] = Field(default_factory=list)
    candidate_template_action: str = "distill_or_refine_template"
    query_structure: str = ""
    evidence_type: str = ""
    answer_structure: str = ""
    trigger_conditions: dict[str, Any] = Field(default_factory=dict)
    reusable_template: str = ""


class CoverageGapRecord(_BaseModel):
    question_id: str
    intent: str
    score: float
    question: str = ""
    gold_answer: str = ""
    missing_entities: list[str] = Field(default_factory=list)
    corpus_search_hint: str = ""
    source_paper: str = ""
    detected_at: str = ""
    status: str = "open"


class PartialSuccessRecord(_BaseModel):
    question_id: str
    intent: str
    score: float
    partial_success_reason: str = ""
    primary_failure_type: FailureType | None = None
    evolution_action: str = "uncertain_queue"


class AnalysisNote(_BaseModel):
    run_id: str
    generated_at: str = ""
    summary: str = ""
    findings: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)


class CaseLibrary(_BaseModel):
    failures: list[FailureRecord] = Field(default_factory=list)
    successes: list[SuccessPattern] = Field(default_factory=list)
    partial_successes: list[PartialSuccessRecord] = Field(default_factory=list)
    coverage_gaps: list[CoverageGapRecord] = Field(default_factory=list)
    analysis_notes: list[AnalysisNote] = Field(default_factory=list)


class TraceReport(_BaseModel):
    run_id: str
    generated_at: str = ""
    total_questions: int = 0
    average_score: float = 0.0
    by_intent: dict[str, dict[str, float]] = Field(default_factory=dict)
    failures: list[FailureRecord] = Field(default_factory=list)
    partial_successes: list[PartialSuccessRecord] = Field(default_factory=list)
    coverage_gaps: list[CoverageGapRecord] = Field(default_factory=list)
    success_patterns: list[SuccessPattern] = Field(default_factory=list)
    trace_field_audits: list[dict[str, Any]] = Field(default_factory=list)
    analysis_clusters: list[dict[str, Any]] = Field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════
# Rule-based candidate attribution (priority-ordered)
# ══════════════════════════════════════════════════════════════════════════

def _get_prompt_role(ft: FailureType) -> str:
    mapping = {
        FailureType.ENTITY_MISS: "query_rewrite",
        FailureType.ALIAS_MISS: "query_rewrite",
        FailureType.LOCAL_ID_MISS: "query_rewrite",
        FailureType.ROUTING_ERROR: "retrieval_routing",
        FailureType.PLANNER_ERROR: "query_rewrite",
        FailureType.GRAPH_EXPANSION_ERROR: "evidence_expansion",
        FailureType.ASSESSMENT_FALSE_NEG: "evidence_assessment",
        FailureType.TABLE_REASONING_ERROR: "evidence_assessment",
        FailureType.GENERATION_ERROR: "answer_generation",
        FailureType.TOOL_ERROR: "",
        FailureType.UNKNOWN_FAILURE: "",
    }
    return mapping.get(ft, "")


def _get_normalized_target(ft: FailureType) -> str:
    mapping = {
        FailureType.ENTITY_MISS: "strategy.query_rewrite",
        FailureType.ALIAS_MISS: "strategy.query_rewrite",
        FailureType.LOCAL_ID_MISS: "strategy.query_rewrite",
        FailureType.ROUTING_ERROR: "strategy.retrieval_routing",
        FailureType.PLANNER_ERROR: "strategy.query_rewrite",
        FailureType.GRAPH_EXPANSION_ERROR: "strategy.evidence_expansion",
        FailureType.ASSESSMENT_FALSE_NEG: "strategy.assessment",
        FailureType.TABLE_REASONING_ERROR: "strategy.assessment",
        FailureType.GENERATION_ERROR: "strategy.answer_generation",
        FailureType.TOOL_ERROR: "",
        FailureType.UNKNOWN_FAILURE: "",
    }
    return mapping.get(ft, "")


def _get_proposed_patch_targets(ft: FailureType) -> list[str]:
    mapping = {
        FailureType.ENTITY_MISS: ["strategy.query_rewrite", "strategy.retrieval_routing"],
        FailureType.ALIAS_MISS: ["strategy.query_rewrite", "strategy.retrieval_routing"],
        FailureType.LOCAL_ID_MISS: ["strategy.query_rewrite"],
        FailureType.ROUTING_ERROR: ["strategy.retrieval_routing"],
        FailureType.GRAPH_EXPANSION_ERROR: ["strategy.evidence_expansion"],
        FailureType.PLANNER_ERROR: ["strategy.query_rewrite"],
        FailureType.ASSESSMENT_FALSE_NEG: ["strategy.assessment"],
        FailureType.TABLE_REASONING_ERROR: ["strategy.assessment", "strategy.answer_generation"],
        FailureType.GENERATION_ERROR: ["strategy.answer_generation"],
        FailureType.TOOL_ERROR: [],
        FailureType.UNKNOWN_FAILURE: [],
    }
    return mapping.get(ft, [])


def _detect_assessment_false_neg(trace: StandardTrace, react_log: str, gold_answer: str) -> CandidateAttributionEntry | None:
    """Check if assessment incorrectly declared evidence insufficient."""
    answer = trace.answer.lower()
    _REFUSAL = ["insufficient", "not enough", "does not include", "cannot determine",
                "unable to", "not sufficient", "not found", "no data", "no evidence"]
    is_refusal = any(s in answer for s in _REFUSAL)
    if not is_refusal:
        return None
    if not trace.evidence_items:
        return None
    # Check react log for evidence with assessment false negative
    if "sufficient=false" in react_log.lower():
        new_counts = re.findall(r"(\d+) new", react_log)
        if new_counts and int(new_counts[0]) >= 3:
            return CandidateAttributionEntry(
                type="assessment_false_negative", confidence=0.85,
                signals=["assessment.sufficient=false", f"retrieved={new_counts[0]}", "answer_is_refusal"])
    return None


def _detect_routing_error(trace: StandardTrace, react_log: str, gold_answer: str) -> CandidateAttributionEntry | None:
    """Check if wrong retrieval channels were used."""
    answer = trace.answer.lower()
    _REFUSAL = ["insufficient", "not enough", "does not include", "cannot determine"]
    is_refusal = any(s in answer for s in _REFUSAL)
    if trace.score >= 0.3:
        return None
    new_counts = re.findall(r"(\d+) new", react_log)
    total_new = sum(int(n) for n in new_counts) if new_counts else 0
    if total_new < 3 and is_refusal:
        return CandidateAttributionEntry(type="routing_error", confidence=0.65,
                                         signals=[f"low_evidence={total_new}"])
    if is_refusal and "sufficient=false" in react_log.lower() and total_new >= 3:
        # Check if gold terms appear in evidence
        if gold_answer:
            gold_terms = set(re.findall(r'\b\w{4,}\b', gold_answer.lower()))
            common = {"product", "yield", "reaction", "solvent", "conditions", "table", "entry", "data"}
            specific = gold_terms - common
            if specific and not any(t in react_log.lower() for t in specific):
                return CandidateAttributionEntry(type="routing_error", confidence=0.70,
                                                 signals=["gold_terms_missing_from_evidence"])
    return None


def _detect_entity_miss(trace: StandardTrace, react_log: str, gold_answer: str = "") -> CandidateAttributionEntry | None:
    answer = trace.answer.lower()
    if "not found" in answer or "does not mention" in trace.feedback.lower():
        return CandidateAttributionEntry(type="entity_miss", confidence=0.65,
                                         signals=["entity_not_found"])
    return None


def _detect_local_id_miss(trace: StandardTrace, react_log: str, gold_answer: str = "") -> CandidateAttributionEntry | None:
    local_ids = re.findall(r"\b(\d+[a-z])\b", trace.question.lower())
    if not local_ids:
        return None
    answer = trace.answer.lower()
    if "not found" in answer:
        return CandidateAttributionEntry(type="local_id_miss", confidence=0.70,
                                         signals=[f"local_ids={local_ids}"])
    return None


def _detect_table_reasoning_error(trace: StandardTrace, react_log: str, gold_answer: str = "") -> CandidateAttributionEntry | None:
    q = trace.question.lower()
    table_signals = ["entry", "table", "row", "column", "yield", "solvent"]
    if not any(s in q for s in table_signals):
        return None
    answer = trace.answer.lower()
    _REFUSAL = ["insufficient", "not found", "no data"]
    if any(s in answer for s in _REFUSAL):
        return None
    if trace.score < 0.3 and ("table" in react_log.lower() or "entry" in react_log.lower()):
        return CandidateAttributionEntry(type="table_reasoning_error", confidence=0.65,
                                         signals=["table_evidence_present", "wrong_answer"])
    return None


def _detect_generation_error(trace: StandardTrace, react_log: str, gold_answer: str = "") -> CandidateAttributionEntry | None:
    if trace.score >= 0.5:
        return None
    if "sufficient=true" in react_log.lower():
        return CandidateAttributionEntry(type="generation_error", confidence=0.70,
                                         signals=["assessment.sufficient=true", "wrong_answer"])
    return None


def candidate_attribution(trace: StandardTrace, react_log: str = "",
                          gold_answer: str = "") -> CandidateAttribution:
    """Rule-based candidate attribution — priority-ordered detection.

    Runs each detector and collects candidates with confidence scores.
    LLM will verify in Stage 2. NOT a single conclusion.
    """
    qid = trace.question_id
    candidates: list[CandidateAttributionEntry] = []
    uncertainty: list[str] = []

    # Tool error check first (immediate return — not patchable)
    answer = trace.answer.lower()
    if any(s in answer for s in ["rdkit", "neo4j", "parse error", "could not parse"]):
        candidates.append(CandidateAttributionEntry(
            type="tool_error", confidence=0.80, signals=["tool_error_in_answer"]))
        return CandidateAttribution(
            question_id=qid, candidate_failure_types=candidates,
            candidate_target_paths=[], candidate_prompt_role="",
            uncertainty_flags=["tool_error_detected"], needs_llm_review=False)

    # Priority-ordered detectors
    detectors = [
        _detect_assessment_false_neg,
        _detect_routing_error,
        _detect_entity_miss,
        _detect_local_id_miss,
        _detect_table_reasoning_error,
        _detect_generation_error,
    ]
    for detector in detectors:
        result = detector(trace, react_log, gold_answer)
        if result:
            candidates.append(result)

    if not candidates:
        candidates.append(CandidateAttributionEntry(
            type="unknown_failure", confidence=0.30, signals=["no_rule_matched"]))
        uncertainty.append("no_rule_matched")

    if not gold_answer:
        uncertainty.append("missing_gold_answer")

    best_type = candidates[0].type
    ft = FailureType(best_type) if best_type in [e.value for e in FailureType] else FailureType.UNKNOWN_FAILURE

    return CandidateAttribution(
        question_id=qid,
        candidate_failure_types=candidates,
        candidate_target_paths=_get_proposed_patch_targets(ft),
        candidate_prompt_role=_get_prompt_role(ft),
        uncertainty_flags=uncertainty,
        needs_llm_review=ft.is_patchable and candidates[0].confidence < 0.85,
    )


# ══════════════════════════════════════════════════════════════════════════
# Analysis clusters (cross-skill)
# ══════════════════════════════════════════════════════════════════════════

def build_analysis_clusters(failures: list[FailureRecord]) -> list[dict[str, Any]]:
    """Group failures into analysis clusters for LLM analysis (cross-skill).

    Key = (failure_type, normalized_target_path, prompt_role)
    """
    groups: dict[tuple, dict[str, Any]] = {}

    for f in failures:
        if not f.primary_failure_type.is_patchable:
            continue
        if f.attribution_confidence < 0.5:
            continue

        key = (
            f.primary_failure_type.value,
            _get_normalized_target(f.primary_failure_type),
            _get_prompt_role(f.primary_failure_type),
        )

        if key not in groups:
            groups[key] = {
                "cluster_id": f"{key[0]}__{key[1]}__{key[2]}",
                "failure_type": key[0],
                "normalized_target_path": key[1],
                "prompt_role": key[2],
                "failures": [],
                "affected_skills": set(),
            }

        groups[key]["failures"].append(f)
        groups[key]["affected_skills"].add(f.skill_name)

    result: list[dict[str, Any]] = []
    for c in groups.values():
        c["affected_skills"] = sorted(c["affected_skills"])
        c["cluster_size"] = len(c["failures"])
        result.append(c)

    return result


# ══════════════════════════════════════════════════════════════════════════
# Success pattern mining
# ══════════════════════════════════════════════════════════════════════════

def mine_success_pattern(trace: StandardTrace) -> SuccessPattern:
    """Extract structural pattern from a successful query."""
    q_lower = trace.question.lower()
    answer_lower = trace.answer.lower()
    evidence_text = " ".join(
        ev.summary for rr in trace.retrieval_rounds
        for ev in rr.retrieved_evidence
    ).lower()

    # Query structure
    query_structure = "simple_lookup"
    if any(w in q_lower for w in ["compare", "versus", "vs", "higher", "lower", "best"]):
        query_structure = "comparison"
    elif any(w in q_lower for w in ["yield", "temperature", "solvent", "catalyst", "pressure"]):
        query_structure = "property_extraction"
    elif any(w in q_lower for w in ["which", "what compound", "what molecule", "identify"]):
        query_structure = "identification"
    elif any(w in q_lower for w in ["why", "mechanism", "reason", "explain"]):
        query_structure = "explanation"
    elif any(w in q_lower for w in ["synthesis", "procedure", "protocol", "how to"]):
        query_structure = "procedure_query"

    # Evidence type
    evidence_type = "text_evidence"
    if "table" in evidence_text or "entry" in evidence_text:
        evidence_type = "table_evidence"
    elif "reaction_event" in evidence_text or "reaction card" in evidence_text:
        evidence_type = "reaction_event"
    elif "molecule" in evidence_text:
        evidence_type = "entity_evidence"

    # Answer structure
    answer_structure = "narrative"
    if re.search(r'\d+%', answer_lower):
        answer_structure = "numeric_value"
    elif "," in trace.answer and len(trace.answer.split(",")) >= 3:
        answer_structure = "list"
    elif any(w in answer_lower for w in ["yes", "no", "true", "false"]):
        answer_structure = "boolean"

    pattern = f"{query_structure}_{evidence_type}_{answer_structure}"

    # Trigger conditions
    trigger: dict[str, Any] = {"intent": trace.intent}
    if query_structure == "comparison":
        trigger["query_signals"] = ["compare", "versus", "higher", "lower", "best"]
    elif query_structure == "property_extraction":
        trigger["query_signals"] = ["yield", "temperature", "solvent", "catalyst"]

    # Reusable template
    reusable = ""
    if query_structure == "comparison":
        reusable = ("For comparison queries: extract all relevant values from evidence, "
                     "identify the comparison dimension, and explicitly state which option "
                     "ranks highest/lowest with the supporting value.")
    elif query_structure == "property_extraction":
        reusable = ("For property extraction queries: locate the specific property in evidence, "
                     "extract the exact value with units, and cite the evidence source.")

    return SuccessPattern(
        intent=trace.intent,
        pattern=pattern,
        frequency=1,
        avg_score=trace.score,
        supporting_question_ids=[trace.question_id],
        query_structure=query_structure,
        evidence_type=evidence_type,
        answer_structure=answer_structure,
        trigger_conditions=trigger,
        reusable_template=reusable,
    )


# ══════════════════════════════════════════════════════════════════════════
# Coverage gap handling
# ══════════════════════════════════════════════════════════════════════════

def record_coverage_gap(outcome: OutcomeResult) -> CoverageGapRecord:
    """Create a coverage gap record for the backlog."""
    return CoverageGapRecord(
        question_id=outcome.trace_id,
        intent=outcome.metadata.get("intent", "unknown"),
        score=outcome.score,
        question=outcome.metadata.get("question", ""),
        gold_answer=outcome.metadata.get("gold_answer", ""),
        missing_entities=outcome.metadata.get("missing_entities", []),
        corpus_search_hint=outcome.metadata.get("corpus_search_hint", ""),
        source_paper=outcome.metadata.get("source_paper", ""),
        detected_at=datetime.now().strftime("%Y-%m-%d_%H%M"),
        status="open",
    )


# ══════════════════════════════════════════════════════════════════════════
# Main pipeline function
# ══════════════════════════════════════════════════════════════════════════

def generate_trace_report(
    eval_results_path: Path,
    react_logs_dir: Path,
    skill_configs: dict[str, dict] | None = None,
    run_id: str | None = None,
) -> TraceReport:
    """Main entry point for Stage 3. Generates TraceReport from eval results + react logs."""
    if not run_id:
        run_id = f"run_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"

    eval_data = json.loads(eval_results_path.read_text("utf-8"))
    results = eval_data.get("results", [])
    summary = eval_data.get("summary", {})

    report = TraceReport(
        run_id=run_id,
        generated_at=datetime.now().isoformat(),
        total_questions=len(results),
        average_score=summary.get("average_score", 0.0),
    )

    case_library = CaseLibrary()

    for r in results:
        trace = _build_trace_from_result(r, react_logs_dir)

        # Field audit
        audit = audit_trace_fields(trace)
        report.trace_field_audits.append(audit.model_dump(mode="json") if hasattr(audit, 'model_dump') else {})

        outcome = evaluate_outcome(trace)

        if outcome.outcome_type == OutcomeType.SUCCESS:
            sp = mine_success_pattern(trace)
            report.success_patterns.append(sp)
            case_library.successes.append(sp)

        elif outcome.outcome_type == OutcomeType.PARTIAL_SUCCESS:
            attr = candidate_attribution(trace, _load_react_log(r, react_logs_dir), trace.gold_answer)
            report.partial_successes.append(PartialSuccessRecord(
                question_id=r.get("id", ""),
                intent=r.get("intent", "unknown"),
                score=r.get("score", 0.0),
                partial_success_reason=attr.candidate_failure_types[0].type if attr.candidate_failure_types else "",
            ))

        elif outcome.outcome_type == OutcomeType.COVERAGE_GAP:
            cgr = record_coverage_gap(outcome)
            report.coverage_gaps.append(cgr)
            case_library.coverage_gaps.append(cgr)

        else:  # SYSTEM_FAILURE
            # Rule-based candidate attribution
            attr = candidate_attribution(trace, _load_react_log(r, react_logs_dir), trace.gold_answer)

            # Determine skill name
            intent = r.get("intent", "unknown")
            skill_name = intent
            skill_version = "1.0.0"

            # Tool error special handling
            if any(c.type == "tool_error" for c in attr.candidate_failure_types):
                report.failures.append(FailureRecord(
                    question_id=r.get("id", ""),
                    intent=intent,
                    skill_name=skill_name,
                    skill_version=skill_version,
                    score=r.get("score", 0.0),
                    primary_failure_type=FailureType.TOOL_ERROR,
                    attribution_confidence=0.80,
                    attribution_explanation="Tool error detected — not patchable",
                    evolution_action="record_only",
                ))
                continue

            # Build failure record
            best_candidate = attr.candidate_failure_types[0] if attr.candidate_failure_types else CandidateAttributionEntry(type="unknown_failure")
            ft = FailureType(best_candidate.type) if best_candidate.type in [e.value for e in FailureType] else FailureType.UNKNOWN_FAILURE

            fr = FailureRecord(
                question_id=r.get("id", ""),
                intent=intent,
                skill_name=skill_name,
                skill_version=skill_version,
                score=r.get("score", 0.0),
                primary_failure_type=ft,
                attribution_confidence=best_candidate.confidence,
                attribution_explanation=", ".join(best_candidate.signals),
                gold_answer=r.get("ground_truth", ""),
                predicted_answer=r.get("system_answer", "")[:200],
                retrieved_evidence_ids=r.get("evidence_ids", []),
                proposed_patch_targets=attr.candidate_target_paths,
                evolution_action="generate_patch_candidate" if ft.is_patchable else "record_only",
                retrieval_rounds=[_serialize_rr(rr) for rr in trace.retrieval_rounds],
            )
            report.failures.append(fr)
            case_library.failures.append(fr)

    # Build analysis clusters
    report.analysis_clusters = build_analysis_clusters(report.failures)

    # Build intent stats
    intent_scores: dict[str, list[float]] = {}
    for r in results:
        intent = r.get("intent", "unknown")
        intent_scores.setdefault(intent, []).append(r.get("score", 0.0))
    for intent, scores in intent_scores.items():
        report.by_intent[intent] = {
            "average_score": sum(scores) / len(scores) if scores else 0.0,
            "total": len(scores),
        }

    return report


# ══════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════

def _build_trace_from_result(r: dict, react_logs_dir: Path) -> StandardTrace:
    """Build StandardTrace from eval result dict (without full trace module)."""
    from .trace import standardize_trace as st

    question = r.get("question", "")
    q_hash = hashlib.md5(question.encode()).hexdigest()[:8] if question else "00000000"
    source_paper = r.get("source_paper", "").replace(".pdf", "")
    react_log = load_react_log(react_logs_dir, source_paper, q_hash)

    return st(r, react_log)


def _load_react_log(r: dict, react_logs_dir: Path) -> str:
    question = r.get("question", "")
    q_hash = hashlib.md5(question.encode()).hexdigest()[:8] if question else "00000000"
    source_paper = r.get("source_paper", "").replace(".pdf", "")
    return load_react_log(react_logs_dir, source_paper, q_hash)


def _serialize_rr(rr) -> dict:
    """Serialize a RetrievalRound to JSON-safe dict."""
    if hasattr(rr, '__dict__'):
        d = {}
        for k, v in rr.__dict__.items():
            if hasattr(v, '__dict__'):
                d[k] = v.__dict__
            elif isinstance(v, list):
                d[k] = [x.__dict__ if hasattr(x, '__dict__') else x for x in v]
            else:
                d[k] = v
        return d
    return dict(rr)


import hashlib
