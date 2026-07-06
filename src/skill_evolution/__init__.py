"""Skill Evolution module for ChemEvoRAG.

Preserved base modules (validation / regression / writeback):
  schemas.py              — Skill YAML / Prompt Registry schema definitions
  types.py                — Shared type definitions (FailureType, FailureRecord, etc.)
  trace.py                — Trace standardization + field audit
  evaluation.py           — Outcome classification (success/partial/failure/coverage_gap)
  patch.py                — PatchSchema, PatchOperation, PatchStatus, patch generation
  validation.py           — Schema + semantic validation
  runtime_validation.py   — Regression sandbox validation (targeted + composition)
  regression.py           — RegressionRunner, RegressionComparison
  apply.py                — PatchApplier: write patches to YAML files
  rollback.py             — SnapshotManager: before/after snapshots + rollback

New modules (Reflexion-style evolution):
  memory.py               — Reflexion episodic memory: reflection generation + retrieval
  bootstrap.py            — DSPy-style bootstrap: extract few-shot from success traces
  prompt_evolver.py       — Prompt candidate generation from reflection + bootstrap
  simple_runner.py        — Minimal closed-loop evolution runner

Archived (do not use):
  archive/skill_evolution_v1/  — Old over-engineered pipeline (runner, mutation,
                                  distillation, integration, llm_analysis, etc.)
"""

# ── Shared types ────────────────────────────────────────────────────────────
from .types import (  # noqa: F401
    FailureType,
    FailureRecord,
    SuccessPattern,
    CoverageGapRecord,
    PartialSuccessRecord,
    AttributionSource,
)

# ── Trace + Evaluation ─────────────────────────────────────────────────────
from .trace import (  # noqa: F401
    standardize_trace,
    audit_trace_fields,
    StandardTrace,
    FieldAudit,
)

from .evaluation import (  # noqa: F401
    evaluate_outcome,
    OutcomeType,
    OutcomeResult,
)

# ── Patch ───────────────────────────────────────────────────────────────────
from .patch import (  # noqa: F401
    PatchSchema,
    PatchOperation,
    PatchStatus,
    PromptArtifact,
    FailureCluster,
    cluster_failures,
    generate_patches_for_cluster,
    generate_all_patches,
)

# ── Validation ─────────────────────────────────────────────────────────────
from .validation import (  # noqa: F401
    validate_patch,
    ValidationResult,
)

# ── Runtime validation ─────────────────────────────────────────────────────
from .runtime_validation import (  # noqa: F401
    validate_individual_patch,
    validate_composition,
    detect_conflicts,
    resolve_conflicts,
)

# ── Regression ──────────────────────────────────────────────────────────────
from .regression import (  # noqa: F401
    RegressionRunner,
    RegressionResult,
    RegressionComparison,
)

# ── Apply + Rollback ───────────────────────────────────────────────────────
from .apply import (  # noqa: F401
    PatchApplier,
    PatchApplyError,
)

from .rollback import (  # noqa: F401
    SnapshotManager,
    SnapshotError,
)

__all__ = [
    # Types (was attribution)
    "FailureType",
    "FailureRecord",
    "SuccessPattern",
    "CoverageGapRecord",
    "PartialSuccessRecord",
    "AttributionSource",
    # Trace
    "standardize_trace",
    "audit_trace_fields",
    "StandardTrace",
    "FieldAudit",
    # Evaluation
    "evaluate_outcome",
    "OutcomeType",
    "OutcomeResult",
    # Patch
    "PatchSchema",
    "PatchOperation",
    "PatchStatus",
    "PromptArtifact",
    "FailureCluster",
    "cluster_failures",
    "generate_patches_for_cluster",
    "generate_all_patches",
    # Validation
    "validate_patch",
    "ValidationResult",
    # Runtime validation
    "validate_individual_patch",
    "validate_composition",
    "detect_conflicts",
    "resolve_conflicts",
    # Regression
    "RegressionRunner",
    "RegressionResult",
    "RegressionComparison",
    # Apply + Rollback
    "PatchApplier",
    "PatchApplyError",
    "SnapshotManager",
    "SnapshotError",
]
