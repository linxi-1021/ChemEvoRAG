"""LLM-driven success pattern analysis. Refactored 4B — replaces empty template shells.

Part A (global): Analyze all successful traces → per-intent channel/evidence recommendations.
Part B (contrastive): Compare failed vs successful traces per cluster → targeted differences.

Output feeds into mutation.py rule generators as data-driven recommendations.
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from typing import Any


def _call_llm(system_prompt: str, user_prompt: str, temperature: float = 0.3) -> str | None:
    try:
        from openai import OpenAI
    except ImportError:
        return None
    api_key = os.environ.get("API_KEY")
    base_url = os.environ.get("BASE_URL")
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    if not api_key:
        return None
    client = OpenAI(api_key=api_key, base_url=base_url or None)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=2048,
        )
        return resp.choices[0].message.content
    except Exception as e:
        print(f"  [SuccessAnalysis] LLM call failed: {e}")
        return None


def _extract_json(text: str) -> dict | None:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


# ══════════════════════════════════════════════════════════════════════════
# Inline rule-based fallback (no LLM needed for basics)
# ══════════════════════════════════════════════════════════════════════════

def _rule_based_channel_stats(success_traces: list[dict]) -> dict[str, Any]:
    """Compute channel and evidence stats from successful traces without LLM."""
    by_intent: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "channel_counts": defaultdict(int),
        "evidence_type_counts": defaultdict(int),
        "total_rounds": 0,
        "trace_count": 0,
    })

    for trace in success_traces:
        intent = trace.get("intent", "unknown")
        stats = by_intent[intent]
        stats["trace_count"] += 1

        for rr in trace.get("retrieval_rounds", []):
            stats["total_rounds"] += 1
            for ev in rr.get("retrieved_evidence", []):
                if ev.get("contains_gold_answer"):
                    channel = ev.get("channel", "unknown")
                    etype = ev.get("evidence_type", "unknown")
                    stats["channel_counts"][channel] += 1
                    stats["evidence_type_counts"][etype] += 1

    # Build recommendations
    result: dict[str, Any] = {"by_intent": {}, "cross_skill_patterns": []}
    for intent, stats in by_intent.items():
        if stats["trace_count"] == 0:
            continue
        channel_rank = sorted(stats["channel_counts"].items(), key=lambda x: -x[1])
        evidence_rank = sorted(stats["evidence_type_counts"].items(), key=lambda x: -x[1])
        avg_rounds = stats["total_rounds"] / stats["trace_count"] if stats["trace_count"] else 1

        result["by_intent"][intent] = {
            "best_channels": [c for c, _ in channel_rank[:3]],
            "best_evidence_types": [e for e, _ in evidence_rank[:3]],
            "avg_rounds": round(avg_rounds, 1),
            "trace_count": stats["trace_count"],
        }

    return result


def _rule_based_contrastive(
    failure_traces: list[dict],
    success_traces: list[dict],
) -> dict[str, Any]:
    """Compare failure vs success traces, identify key differences."""
    if not success_traces:
        return {}

    # Aggregate
    def _agg(traces):
        ch = defaultdict(int)
        et = defaultdict(int)
        rounds = []
        for t in traces:
            for rr in t.get("retrieval_rounds", []):
                for ev in rr.get("retrieved_evidence", []):
                    ch[ev.get("channel", "?")] += 1
                    et[ev.get("evidence_type", "?")] += 1
            rounds.append(len(t.get("retrieval_rounds", [])))
        return ch, et, sum(rounds) / len(rounds) if rounds else 1

    f_ch, f_et, f_r = _agg(failure_traces)
    s_ch, s_et, s_r = _agg(success_traces)

    differences = []
    # Compare evidence types
    for etype in set(list(f_et.keys()) + list(s_et.keys())):
        f_cnt = f_et.get(etype, 0)
        s_cnt = s_et.get(etype, 0)
        if s_cnt > f_cnt * 1.5:
            differences.append({
                "dimension": "evidence_type",
                "value": etype,
                "success_count": s_cnt,
                "failure_count": f_cnt,
                "insight": f"Successful traces use '{etype}' {s_cnt}x vs failures {f_cnt}x",
            })

    # Compare channels
    for ch in set(list(f_ch.keys()) + list(s_ch.keys())):
        f_cnt = f_ch.get(ch, 0)
        s_cnt = s_ch.get(ch, 0)
        if s_cnt > f_cnt * 1.5:
            differences.append({
                "dimension": "channel",
                "value": ch,
                "success_count": s_cnt,
                "failure_count": f_cnt,
                "insight": f"Successful traces use channel '{ch}' {s_cnt}x vs failures {f_cnt}x",
            })

    return {
        "failure_avg_rounds": round(f_r, 1),
        "success_avg_rounds": round(s_r, 1),
        "key_differences": differences[:5],
    }


# ══════════════════════════════════════════════════════════════════════════
# LLM-enhanced analysis (when API available)
# ══════════════════════════════════════════════════════════════════════════

_GLOBAL_ANALYSIS_PROMPT = """You analyze successful RAG traces. Given per-intent channel and evidence stats,
recommend retrieval strategy improvements. Output JSON:

{
  "by_intent": {
    "property_query": {
      "recommended_primary_channels": ["lexical_search", "entity_search"],
      "recommended_boost_weights": {"table_block": 6.0, "reaction_event": 4.0},
      "recommended_top_k": {"lexical_search": 15, "entity_search": 10},
      "rationale": "Table data via lexical search dominates successful property lookups"
    }
  },
  "cross_skill_recommendations": [
    {"skills": ["property_query","reaction_condition_query"], "shared_channel": "lexical_search", "rationale": "..."}
  ]
}"""

_CONTRASTIVE_PROMPT = """You analyze why some RAG traces succeeded and others failed.
Given failure traces and successful traces for the same intent, identify key differences.
Output JSON:

{
  "root_cause_insight": "Failure traces relied on entity_search which missed table data; successes used lexical_search",
  "recommended_changes": {
    "retrieval_routing": {"should_change": true, "boost_table_block": true, "rationale": "..."},
    "query_rewrite": {"should_change": false}
  }
}"""


def analyze_success_patterns(
    success_traces: list[dict],
    failure_traces_by_cluster: dict[str, list[dict]] | None = None,
) -> dict[str, Any]:
    """Main entry point. Runs rule-based analysis, optionally enhances with LLM.

    Args:
        success_traces: List of successful trace dicts from TraceReport
        failure_traces_by_cluster: Optional {cluster_id: [failure traces]}

    Returns:
        {"global": {...}, "contrastive": {cluster_id: {...}}}
    """
    # Always run rule-based first (fast, no API needed)
    global_result = _rule_based_channel_stats(success_traces)
    contrastive_results: dict[str, Any] = {}

    if failure_traces_by_cluster:
        for cluster_id, failure_traces in failure_traces_by_cluster.items():
            # Find matching success traces by intent
            failure_intents = set(t.get("intent", "") for t in failure_traces)
            matching_success = [
                t for t in success_traces
                if t.get("intent", "") in failure_intents
            ]
            if matching_success:
                contrastive_results[cluster_id] = _rule_based_contrastive(
                    failure_traces, matching_success,
                )

    # Try LLM enhancement
    llm_result = _try_llm_enhancement(global_result, contrastive_results, success_traces)
    if llm_result:
        return llm_result

    return {
        "global": global_result,
        "contrastive": contrastive_results,
    }


def _try_llm_enhancement(
    global_result: dict,
    contrastive_results: dict,
    success_traces: list[dict],
) -> dict | None:
    """Try to enhance rule-based results with LLM analysis."""
    # Build a compact prompt for Part A
    lines = []
    lines.append("## Per-Intent Channel & Evidence Stats (from successful traces)")
    for intent, stats in global_result.get("by_intent", {}).items():
        lines.append(f"### {intent} ({stats.get('trace_count', 0)} traces)")
        lines.append(f"Best channels: {stats.get('best_channels', [])}")
        lines.append(f"Best evidence types: {stats.get('best_evidence_types', [])}")
        lines.append(f"Avg rounds: {stats.get('avg_rounds', 1)}")
        lines.append("")

    if not lines:
        return None

    response = _call_llm(_GLOBAL_ANALYSIS_PROMPT, "\n".join(lines), temperature=0.2)
    if not response:
        return None

    parsed = _extract_json(response)
    if not parsed:
        return None

    # Merge LLM recommendations with rule-based stats
    llm_by_intent = parsed.get("by_intent", {})
    for intent in global_result.get("by_intent", {}):
        if intent in llm_by_intent:
            global_result["by_intent"][intent].update(llm_by_intent[intent])

    return {
        "global": global_result,
        "contrastive": contrastive_results,
        "llm_recommendations": parsed.get("cross_skill_recommendations", []),
    }
