"""Shared configuration for Skill Evolution pipeline.

All configurable thresholds and settings live here.
No hardcoded values in other modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EvolutionConfig:
    """All configurable parameters for the Skill Evolution pipeline."""

    # ── Regression ─────────────────────────────────────────────
    regression_runs: int = 3
    """Number of parallel eval runs per patch (averaged). Configurable, not hardcoded."""

    regression_limit: int | None = None
    """Max questions per regression eval. None = full dataset."""

    regression_seed: int = 42
    """Random seed for question sampling when regression_limit is set."""

    workers: int = 1
    """Number of concurrent workers passed to eval_questions."""

    # ── LLM retry ──────────────────────────────────────────────
    max_retries: int = 3
    """Max LLM API retries per Stage 2/3 call with escalating temperature."""

    max_refine_iterations: int = 5
    """Max iterations for the refine loop (V2→V3→V4→...)."""

    # ── Patch limits ───────────────────────────────────────────
    max_patches_per_round: int = 10
    """Max patches selected per evolution round (enforced in integration.py)."""

    patch_confidence_threshold: float = 0.75
    """Minimum confidence for a patch to be considered valid."""

    # ── Regression thresholds ──────────────────────────────────
    min_score: float = 0.80
    """Minimum acceptable global average score after patching."""

    max_score_drop: float = 0.01
    """Maximum allowed score drop vs baseline."""

    min_targeted_improvement: float = 0.05
    """Minimum targeted improvement (after_score - original_score)."""

    max_failed_cases_increase: int = 1
    """Maximum allowed increase in failed cases (score < 0.3)."""

    # ── Mutation ───────────────────────────────────────────────
    mutation_enabled_dimensions: list[str] = field(default_factory=lambda: [
        "prompt", "routing", "planning", "fallback",
        "graph_expansion", "stop_condition",
    ])
    """Which mutation dimensions are active this round."""

    # ── Distillation ───────────────────────────────────────────
    min_pattern_frequency: int = 3
    """Minimum unique question count to distill a success pattern into a template."""

    min_pattern_score: float = 0.85
    """Minimum average score for a success pattern to be distilled."""

    template_merge_similarity: float = 0.8
    """Similarity threshold for merging overlapping templates."""

    cross_skill_promotion_min_frequency: int = 5
    """Minimum frequency in source skill before cross-skill promotion is considered."""

    cross_skill_similarity_threshold: float = 0.7
    """Minimum similarity between source and target skills for cross-skill promotion."""

    # ── Coverage Gap ───────────────────────────────────────────
    coverage_gap_backlog_dir: str = "data/evolution/backlog"
    """Directory for coverage gap records."""

    # ── Output ─────────────────────────────────────────────────
    stream_output: bool = True
    """Show eval progress during regression (tqdm + per-question scores)."""


# Singleton config — instantiated at pipeline startup, can be overridden by CLI
CONFIG = EvolutionConfig()


def get_config() -> EvolutionConfig:
    return CONFIG


def set_config(config: EvolutionConfig) -> None:
    global CONFIG
    CONFIG = config
