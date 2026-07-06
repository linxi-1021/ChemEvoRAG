"""Cross-round evolution metrics tracking. Writes to data/evolution/metrics.json.

Tracks: score_trend, failure_count_trend, patches_applied_total, resolved/ persistent failures.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class RoundMetrics:
    run_id: str
    before_score: float
    after_score: float | None = None   # None if validate-only (not applied)
    score_delta: float = 0.0
    applied_patches: list[str] = field(default_factory=list)
    rejected_patches: list[str] = field(default_factory=list)
    resolved_failures: list[str] = field(default_factory=list)
    new_failures: list[str] = field(default_factory=list)
    persistent_failures: list[str] = field(default_factory=list)


def write_metrics(
    metrics_path: Path,
    round_metrics: RoundMetrics,
    previous_failure_ids: list[str] | None = None,
) -> None:
    """Append this round's metrics to data/evolution/metrics.json.

    Also computes trend: score_trend, failure_count_trend.
    """
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing
    data: dict = {"rounds": [], "trend": {}}
    if metrics_path.exists():
        try:
            data = json.loads(metrics_path.read_text("utf-8"))
        except Exception:
            pass

    # Compute persistent/new/resolved failures
    if previous_failure_ids:
        current_failure_ids = set(round_metrics.applied_patches)
        previous_set = set(previous_failure_ids)
        round_metrics.resolved_failures = sorted(previous_set - set(round_metrics.applied_patches))
        round_metrics.new_failures = sorted(set(round_metrics.applied_patches) - previous_set)

    # Append
    data["rounds"].append({
        "run_id": round_metrics.run_id,
        "timestamp": datetime.now().isoformat(),
        "before_score": round_metrics.before_score,
        "after_score": round_metrics.after_score,
        "score_delta": round_metrics.score_delta,
        "applied_patches": round_metrics.applied_patches,
        "rejected_patches": round_metrics.rejected_patches,
        "resolved_failures": round_metrics.resolved_failures,
        "new_failures": round_metrics.new_failures,
        "persistent_failures": round_metrics.persistent_failures,
    })

    # Update trend
    scores = [r.get("before_score", 0) for r in data["rounds"]]
    failure_counts = [
        len(r.get("applied_patches", [])) + len(r.get("rejected_patches", []))
        for r in data["rounds"]
    ]
    data["trend"] = {
        "score_trend": scores,
        "failure_count_trend": failure_counts,
        "patches_applied_total": sum(len(r.get("applied_patches", [])) for r in data["rounds"]),
        "last_updated": datetime.now().isoformat(),
    }

    metrics_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
