"""Skill Evolution module for ChemEvoRAG.

Provides:
  - Attribution (§4): TraceReport generation and failure classification
  - PatchGenerator (§5): Patch schema and candidate generation
  - Validation (§5.5): Patch validation pipeline
"""

from .attribution import (
    AttributionSource,
    CoverageGapRecord,
    EnvironmentSnapshot,
    FailureRecord,
    FailureType,
    IntentStats,
    OutcomeType,
    PartialSuccessRecord,
    SuccessPattern,
    TraceReport,
    generate_trace_report,
    generate_trace_report_from_paths,
)

__all__ = [
    "AttributionSource",
    "CoverageGapRecord",
    "EnvironmentSnapshot",
    "FailureRecord",
    "FailureType",
    "IntentStats",
    "OutcomeType",
    "PartialSuccessRecord",
    "SuccessPattern",
    "TraceReport",
    "generate_trace_report",
    "generate_trace_report_from_paths",
]
