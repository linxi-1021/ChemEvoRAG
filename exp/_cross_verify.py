#!/usr/bin/env python
"""Cross-verify all baseline result files and ChemEvoRAG eval_results.json."""

import json
import os
import statistics
from collections import defaultdict

BASELINE_DIR = "D:/Desktop/evo/ChemEvoRAG_Phase1-main/data/baseline_results"
EVAL_DIR = "D:/Desktop/evo/ChemEvoRAG_Phase1-main/data/eval"

# ---- UTILS ----

def load_json(filepath):
    """Load JSON with multiple encoding fallbacks."""
    for enc in ['utf-8', 'utf-8-sig', 'latin-1', 'gbk', 'gb2312']:
        try:
            with open(filepath, 'r', encoding=enc) as f:
                return json.load(f)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    raise ValueError(f"Cannot decode: {filepath}")

def get_best_file(prefix):
    """Find the file with the most questions and highest avg score for a prefix."""
    files = [f for f in os.listdir(BASELINE_DIR)
             if f.startswith(prefix) and f.endswith('.json') and not f.startswith('~$')]
    if not files:
        return None, []

    scored = []
    for f in files:
        try:
            d = load_json(os.path.join(BASELINE_DIR, f))
            results = d.get("results", [])
            n = len(results)
            scores = [r.get("judge_score") for r in results
                      if r.get("judge_score") is not None]
            avg = statistics.mean(scores) if scores else 0.0
            scored.append((f, n, avg, d))
        except Exception as e:
            print(f"  [SKIP] {f}: {e}")
            continue

    if not scored:
        return None, []

    # Sort by question count desc, then avg score desc
    scored.sort(key=lambda x: (x[1], x[2]), reverse=True)
    best_file, best_n, best_avg, best_data = scored[0]
    return best_file, scored

# ---- ANALYSIS ----

def analyze_file(filename, data, prefix):
    """Analyze a single result file."""
    results = data.get("results", [])
    total_questions = data.get("total_questions", len(results))
    baseline = data.get("baseline", "unknown")

    issues = []
    questions = []

    for i, r in enumerate(results):
        q = r.get("question")
        sa = r.get("system_answer", r.get("answer", ""))
        gt = r.get("ground_truth", "")
        js = r.get("judge_score")
        intent = r.get("intent", "")

        if not q:
            issues.append(f"Q{i}: missing question_text")
        if sa is None or sa == "":
            issues.append(f"Q{i}: empty system_answer")
        if gt is None or gt == "":
            issues.append(f"Q{i}: empty ground_truth")
        if js is None:
            issues.append(f"Q{i}: missing judge_score")
        elif not isinstance(js, (int, float)):
            issues.append(f"Q{i}: judge_score is not numeric ({type(js).__name__}): {js}")
        elif js < 0.0 or js > 1.0:
            issues.append(f"Q{i}: judge_score out of range: {js}")

        questions.append({
            "idx": i,
            "intent": intent,
            "score": js if isinstance(js, (int, float)) else None,
            "has_empty_answer": (sa is None or sa == ""),
            "has_no_gt": (gt is None or gt == ""),
        })

    # Per-intent scores
    intent_scores = defaultdict(list)
    for q in questions:
        intent = q["intent"] or "unknown"
        if q["score"] is not None:
            intent_scores[intent].append(q["score"])

    # Overall
    all_scores = [q["score"] for q in questions if q["score"] is not None]
    empty_answers = sum(1 for q in questions if q["has_empty_answer"])
    no_gt = sum(1 for q in questions if q["has_no_gt"])

    return {
        "baseline": baseline,
        "total_questions_claimed": total_questions,
        "actual_question_count": len(questions),
        "scored_count": len(all_scores),
        "empty_answers": empty_answers,
        "no_ground_truth": no_gt,
        "min_score": min(all_scores) if all_scores else None,
        "max_score": max(all_scores) if all_scores else None,
        "avg_score": statistics.mean(all_scores) if all_scores else None,
        "intent_scores": {k: {
            "count": len(v),
            "avg": statistics.mean(v),
            "min": min(v),
            "max": max(v),
        } for k, v in sorted(intent_scores.items())},
        "issues": issues[:20],  # cap issues printed
        "total_issues": len(issues),
    }


# ---- MAIN ----

print("=" * 100)
print("CROSS-VERIFICATION OF ALL BASELINE RESULT FILES")
print("=" * 100)

# Define prefixes to analyze
prefixes = [
    ("direct", "Direct LLM"),
    ("langchain", "LangChain RAG"),
    ("paperqa2", "PaperQA2"),
    ("lightrag", "LightRAG"),
    ("llamaindex", "LlamaIndex"),
    ("haystack", "Haystack"),
    ("fastgraphrag", "FastGraphRAG"),
    ("velocirag", "VelociRAG"),
    ("r2r", "R2R"),
    ("agent_react", "ReAct Agent"),
    ("agent_codex", "Codex Sim Agent"),
    ("agent_claude_real_opus", "Claude Code Agent (opus)"),
    ("agent_codex_real_gpt-5_5", "Codex Real Agent (gpt-5)"),
]

results_summary = []

for prefix, label in prefixes:
    print(f"\n{'='*80}")
    print(f"[{label}] (prefix={prefix}_)")
    print(f"{'='*80}")

    best_file, all_files = get_best_file(prefix)
    if best_file is None:
        print(f"  NO VALID FILES FOUND!")
        results_summary.append((label, "NO FILE", 0, 0, 0, "N/A", {}))
        continue

    print(f"  All candidates (file, question_count, avg_score):")
    for fname, n, avg, _ in all_files:
        marker = " <-- BEST" if fname == best_file else ""
        print(f"    {fname}: {n} questions, avg={avg:.4f}{marker}")

    best_data = load_json(os.path.join(BASELINE_DIR, best_file))
    analysis = analyze_file(best_file, best_data, prefix)

    print(f"\n  SELECTED FILE: {best_file}")
    print(f"  Baseline field: {analysis['baseline']}")
    print(f"  Claimed total_questions: {analysis['total_questions_claimed']}")
    print(f"  Actual question count: {analysis['actual_question_count']}")
    print(f"  Scored count: {analysis['scored_count']}")
    print(f"  Empty answers: {analysis['empty_answers']}")
    print(f"  No ground_truth: {analysis['no_ground_truth']}")
    print(f"  Score range: [{analysis['min_score']}, {analysis['max_score']}]")
    print(f"  Overall avg: {analysis['avg_score']:.4f}" if analysis['avg_score'] is not None else "  Overall avg: N/A")

    print(f"\n  Per-intent scores:")
    for intent, stats in analysis['intent_scores'].items():
        print(f"    {intent}: n={stats['count']}, avg={stats['avg']:.4f}, range=[{stats['min']}, {stats['max']}]")

    if analysis['issues']:
        print(f"\n  ISSUES ({analysis['total_issues']} total, showing first 20):")
        for iss in analysis['issues']:
            print(f"    - {iss}")

    results_summary.append((
        label, best_file, analysis['actual_question_count'],
        analysis['empty_answers'], analysis['scored_count'],
        f"[{analysis['min_score']}, {analysis['max_score']}]" if analysis['min_score'] is not None else "N/A",
        analysis['intent_scores'],
        analysis['avg_score'],
    ))


# ---- ChemEvoRAG ----
print(f"\n{'='*80}")
print("[ChemEvoRAG] (eval_results.json)")
print(f"{'='*80}")

evo_path = os.path.join(EVAL_DIR, "eval_results.json")
evo_data = load_json(evo_path)
print(f"  File: eval_results.json")
print(f"  Top-level keys: {list(evo_data.keys())}")

# eval_results.json has a different structure - check it
# It might have per-question entries or aggregated results
if "results" in evo_data:
    evo_analysis = analyze_file("eval_results.json", evo_data, "chemevorag")
    print(f"  Question count: {evo_analysis['actual_question_count']}")
    print(f"  Scored count: {evo_analysis['scored_count']}")
    print(f"  Empty answers: {evo_analysis['empty_answers']}")
    print(f"  Score range: [{evo_analysis['min_score']}, {evo_analysis['max_score']}]")
    print(f"  Overall avg: {evo_analysis['avg_score']:.4f}" if evo_analysis['avg_score'] is not None else "  Overall avg: N/A")
    if evo_analysis['intent_scores']:
        print(f"  Per-intent scores:")
        for intent, stats in evo_analysis['intent_scores'].items():
            print(f"    {intent}: n={stats['count']}, avg={stats['avg']:.4f}")
    results_summary.append((
        "ChemEvoRAG", "eval_results.json", evo_analysis['actual_question_count'],
        evo_analysis['empty_answers'], evo_analysis['scored_count'],
        f"[{evo_analysis['min_score']}, {evo_analysis['max_score']}]" if evo_analysis['min_score'] is not None else "N/A",
        evo_analysis['intent_scores'],
        evo_analysis['avg_score'],
    ))
else:
    # Let's inspect the structure more carefully
    print(f"  Raw top-level keys: {list(evo_data.keys())}")
    # Print a sample
    sample_keys = list(evo_data.keys())
    if sample_keys:
        first_key = sample_keys[0]
        val = evo_data[first_key]
        print(f"  First entry key={first_key}, type={type(val).__name__}")
        if isinstance(val, dict):
            print(f"  First entry keys: {list(val.keys())[:10]}")
        elif isinstance(val, list):
            print(f"  First entry length: {len(val)}")
            if val:
                print(f"  First item: {val[0] if not isinstance(val[0], dict) else list(val[0].keys())[:10]}")


# ---- FINAL SUMMARY TABLE ----
print(f"\n\n{'='*120}")
print("FINAL SUMMARY TABLE")
print(f"{'='*120}")
print(f"{'Baseline':<30} {'File':<55} {'Q':>5} {'Empty':>6} {'Scored':>7} {'Avg':>8} {'Range':>18}")
print("-" * 130)

for (label, fname, nq, empty, scored, score_range, intent_scores, avg) in results_summary:
    avg_str = f"{avg:.4f}" if avg is not None else "N/A"
    print(f"{label:<30} {fname:<55} {nq:>5} {empty:>6} {scored:>7} {avg_str:>8} {score_range:>18}")

print(f"\n{'='*120}")
print("PER-INTENT COMPARISON TABLE")
print(f"{'='*120}")

# Collect all intents
all_intents = set()
for (label, fname, nq, empty, scored, score_range, intent_scores, avg) in results_summary:
    all_intents.update(intent_scores.keys())
all_intents = sorted(all_intents)

# Header
header = f"{'Baseline':<30}"
for intent in all_intents:
    header += f" {intent[:12]:>12}"
print(header)
print("-" * (30 + 14 * len(all_intents)))

for (label, fname, nq, empty, scored, score_range, intent_scores, avg) in results_summary:
    row = f"{label:<30}"
    for intent in all_intents:
        if intent in intent_scores:
            row += f" {intent_scores[intent]['avg']:>12.4f}"
        else:
            row += f" {'--':>12}"
    print(row)

print("\nDone.")
