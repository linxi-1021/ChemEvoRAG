"""
Shared utilities for 3-baseline RAG comparison experiment.
Loads questions from data/eval/, saves results in unified format.
"""
import json
import os
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = PROJECT_ROOT / "data" / "eval"
PDF_DIR = PROJECT_ROOT / "data" / "pdfs"
RESULTS_DIR = PROJECT_ROOT / "data" / "baseline_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_all_questions() -> list[dict[str, Any]]:
    """Load all questions from data/eval/*.json."""
    questions = []
    for fp in sorted(EVAL_DIR.glob("*_eval.json")):
        with open(fp, encoding="utf-8") as f:
            paper = json.load(f)
        paper_title = paper["paper_title"]
        paper_id = fp.stem.replace("_eval", "")
        for q in paper["questions"]:
            q["paper_title"] = paper_title
            q["paper_id"] = paper_id
            questions.append(q)
    return questions


def save_results(
    baseline_name: str,
    results: list[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Save baseline results to JSON."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"{baseline_name}_{timestamp}.json"
    output = {
        "baseline": baseline_name,
        "timestamp": timestamp,
        "total_questions": len(results),
        "metadata": metadata or {},
        "results": results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    return out_path


def load_env_config() -> dict[str, str]:
    """Load API configuration from project .env."""
    env_path = PROJECT_ROOT / ".env"
    config = {}
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    config[key.strip()] = value.strip()
    return config
