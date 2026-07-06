"""Skill Evolution module for ChemEvoRAG.

Architecture-aligned modules:
  schemas.py      — A.D. Skill YAML / Prompt Registry schema definitions
  config.py       — Shared configuration (thresholds, limits)
  trace.py        — 1+2. Trace standardization + field audit
  evaluation.py   — 2. Outcome classification (success/partial/failure/coverage_gap)
  attribution.py  — 3+E. Failure attribution + success pattern mining + CaseLibrary
  mutation.py     — 4A. LLM-directed mutation (6 dimensions)
  distillation.py — 4B+C. Template distillation (add/rewrite/merge + cross-skill promotion)
  integration.py  — 5. Candidate integration + validator/schema adjustment
  validation.py   — 6. Schema + semantic + regression + composition validation
  apply.py        — 7. Writeback + snapshot
  rollback.py     — 7. Rollback
  metrics.py      — Cross-round evolution metrics tracking
  runner.py       — Pipeline orchestrator
"""

from .attribution import (
    generate_trace_report,
    TraceReport,
    FailureRecord,
    FailureType,
    SuccessPattern,
    CoverageGapRecord,
    CandidateAttribution,
    CaseLibrary,
    build_analysis_clusters,
)
from .trace import (
    standardize_trace,
    audit_trace_fields,
    StandardTrace,
    FieldAudit,
)
from .evaluation import (
    evaluate_outcome,
    OutcomeType,
    OutcomeResult,
)
from .config import EvolutionConfig, get_config, set_config
from .runner import EvolutionRunner, RunMode

__all__ = [
    # schemas
    "EvolutionConfig",
    "get_config",
    "set_config",
    # trace
    "standardize_trace",
    "audit_trace_fields",
    "StandardTrace",
    "FieldAudit",
    # evaluation
    "evaluate_outcome",
    "OutcomeType",
    "OutcomeResult",
    # attribution
    "generate_trace_report",
    "TraceReport",
    "FailureRecord",
    "FailureType",
    "SuccessPattern",
    "CoverageGapRecord",
    "CandidateAttribution",
    "CaseLibrary",
    "build_analysis_clusters",
    # runner
    "EvolutionRunner",
    "RunMode",
]
