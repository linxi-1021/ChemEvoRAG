#!/usr/bin/env python
"""Evaluate ChemEvoRAG against generated QA pairs from data/eval/all_questions.json.

For each question: run retrieval + LLM synthesis, compare answer to ground truth
using a second LLM call as judge.  Saves detailed results to data/eval/eval_results.json.

Supports concurrent evaluation with --workers.
"""

from __future__ import annotations

import argparse
import warnings
warnings.filterwarnings("ignore", message=".*resume_download.*", category=FutureWarning)
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

_dotenv = PROJECT_ROOT / ".env"
if _dotenv.exists():
    from dotenv import load_dotenv
    load_dotenv(_dotenv)

from solver.llm_solver import LLMChemSolver
from storage.local_store import LocalStore

def _apply_skill_prompts(skill_config: dict, prompt_registry: dict[str, str]) -> None:
    """DEPRECATED: Legacy monkey-patch kept for backward compatibility only.

    Use resolve_prompt_context() from solver.prompt_context instead.
    This function no longer modifies any module-level variables.
    """
    pass  # No-op — monkey-patching removed in favor of explicit PromptContext


def _load_skill_configs(skills_dir: Path) -> dict[str, dict]:
    """Load all Skill YAML files from config/skills/.

    Returns a dict mapping intent name → parsed YAML dict.
    Gracefully returns {} if the directory doesn't exist.
    """
    if not skills_dir.is_dir():
        return {}
    configs: dict[str, dict] = {}
    try:
        import yaml
    except ImportError:
        return {}
    for f in skills_dir.glob("*.yaml"):
        try:
            data = yaml.safe_load(f.read_text("utf-8"))
            intent = data.get("trigger", {}).get("intent", "")
            if intent:
                configs[intent] = data
        except Exception:
            pass
    return configs


def _load_prompt_registry(prompts_dir: Path) -> dict[str, str]:
    """Load all prompt YAML files from config/prompts/.

    Returns a dict mapping prompt_ref → content string.
    """
    if not prompts_dir.is_dir():
        return {}
    registry: dict[str, str] = {}
    try:
        import yaml
    except ImportError:
        return {}
    for f in prompts_dir.glob("*.yaml"):
        try:
            data = yaml.safe_load(f.read_text("utf-8"))
            ref = data.get("prompt_ref", "")
            content = data.get("content", "")
            if ref and content:
                registry[ref] = content
        except Exception:
            pass
    return registry


def select_questions(
    questions: list[dict],
    *,
    limit: int | None,
    seed: int = 42,
) -> list[dict]:
    """Select a random sample of questions, sorted by original order.

    - If limit is None or <= 0, return all questions.
    - If limit >= len(questions), return all questions.
    - Otherwise, randomly sample `limit` questions using the given seed for reproducibility.
    """
    if limit is None or limit <= 0:
        return questions
    if limit >= len(questions):
        return questions
    rng = random.Random(seed)
    indexed = list(enumerate(questions))
    sampled = rng.sample(indexed, k=limit)
    sampled.sort(key=lambda x: x[0])  # preserve original order
    return [q for _, q in sampled]


def build_sampling_metadata(
    limit: int | None,
    seed: int,
    selected_questions: list[dict],
) -> dict | None:
    """Build sampling metadata dict for eval_results.json.

    Returns None when no sampling was applied (limit is None).
    """
    if limit is None:
        return None
    return {
        "mode": "random",
        "limit": limit,
        "seed": seed,
        "selected_question_ids": [q["id"] for q in selected_questions],
    }


EVAL_DIR = PROJECT_ROOT / "data" / "eval"

JUDGE_PROMPT = """\
You are an evaluator. Compare the system's answer to the ground-truth answer.
Score from 0 to 1:
  1.0 = system answer is fully correct, contains all key facts from ground truth
  0.7 = mostly correct, minor omissions or imprecise wording
  0.5 = partially correct, some right facts but missing important details
  0.3 = vaguely related but factually wrong or very incomplete
  0.0 = completely wrong or unrelated

Also note whether key_entities from the ground truth appear in the system answer.

Return ONLY a JSON object:
{
  "score": 0.0,
  "key_entities_found": ["entity1"],
  "key_entities_missing": ["entity2"],
  "reasoning": "<one sentence explaining the score>"
}"""


def _judge_answer(system_answer: str, ground_truth: str, key_entities: list[str]) -> dict:
    key = os.environ.get("API_KEY")
    url = os.environ.get("BASE_URL")
    model = os.environ.get("LLM_MODEL", "gpt-5-mini")
    if not key:
        return {"score": 0.0, "key_entities_found": [], "key_entities_missing": key_entities, "reasoning": "no API key"}

    from openai import OpenAI
    client = OpenAI(api_key=key, base_url=url or None)

    entities_str = ", ".join(key_entities) if key_entities else "none"
    user_prompt = (
        f"SYSTEM ANSWER:\n{system_answer}\n\n"
        f"GROUND TRUTH:\n{ground_truth}\n\n"
        f"KEY ENTITIES TO CHECK: [{entities_str}]"
    )

    try:
        r = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": JUDGE_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            seed=42,
            max_tokens=int(os.environ.get("LLM_MAX_TOKENS", "16384")),
            extra_body={},
        )
        raw = r.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        return json.loads(raw)
    except Exception as exc:
        return {"score": 0.0, "key_entities_found": [], "key_entities_missing": key_entities, "reasoning": str(exc)}


def _save_results(
    results: list, scores: list, by_intent: dict,
    *,
    output_dir: Path | None = None,
    metadata: dict | None = None,
) -> None:
    """Save current results to eval_results.json (incremental, sorted)."""
    avg = sum(scores) / len(scores) if scores else 0
    out_dir = output_dir or EVAL_DIR

    # Sort results by paper number, then by question id
    def _sort_key(r):
        paper = r.get("source_paper", "")
        num = ""
        for ch in paper:
            if ch.isdigit():
                num += ch
            else:
                break
        paper_num = int(num) if num else 0
        qid = r.get("id", "")
        # Extract numeric from qid like "q1", "q_batch1_1"
        import re
        q_nums = [int(n) for n in re.findall(r"\d+", qid)]
        return (paper_num, q_nums[0] if q_nums else 0)

    sorted_results = sorted(results, key=_sort_key)

    out_path = out_dir / "eval_results.json"
    payload: dict = {
        "summary": {
            "total": len(scores),
            "average_score": avg,
            "by_intent": {k: sum(v) / len(v) for k, v in by_intent.items()},
        },
        "results": sorted_results,
    }
    if metadata:
        payload["metadata"] = metadata
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _eval_one(
    q: dict,
    store: LocalStore,
    solver,
    elementkg_client,
    use_react: bool,
    skill_config: dict | None = None,
    prompt_registry: dict[str, str] | None = None,
) -> dict:
    """Evaluate a single question. Thread-safe.

    Resolves PromptContext from skill_config + prompt_registry and passes
    it explicitly to the solver, so each question uses the prompts
    appropriate for its intent.
    """
    question = q["question"]
    ground_truth = q["ground_truth_answer"]
    key_entities = q.get("key_entities", [])
    intent = q.get("intent", "unknown")
    source = q.get("source_paper", "?")
    doc_id = source.replace(".pdf", "") if source else None

    # Resolve PromptContext from Skill YAML + Prompt Registry
    from solver.prompt_context import resolve_prompt_context
    prompt_context = resolve_prompt_context(
        skill_config or {}, prompt_registry or {},
    )

    try:
        if use_react:
            answer_obj = solver.answer(
                question, doc_ids=[doc_id] if doc_id else None,
                prompt_context=prompt_context,
            )
        else:
            from retrieval.router import RetrievalRouter
            router = RetrievalRouter(store, elementkg_client=elementkg_client)
            package = router.retrieve(question, doc_ids=[doc_id] if doc_id else None)
            answer_obj = solver.answer_from_package(
                package, prompt_context=prompt_context,
            )
        system_answer = answer_obj.answer
        evidence_ids = [s.evidence_id for s in answer_obj.supporting_evidence]
        confidence = answer_obj.confidence
    except Exception as exc:
        system_answer = f"[ERROR: {exc}]"
        evidence_ids = []
        confidence = 0.0

    judge = _judge_answer(system_answer, ground_truth, key_entities)
    score = judge.get("score", 0.0)

    return {
        "id": q["id"],
        "source_paper": source,
        "intent": intent,
        "question": question,
        "ground_truth": ground_truth,
        "system_answer": system_answer,
        "evidence_ids": evidence_ids,
        "confidence": confidence,
        "judge_score": score,
        "entities_found": judge.get("key_entities_found", []),
        "entities_missing": judge.get("key_entities_missing", []),
        "judge_reasoning": judge.get("reasoning", ""),
        "score": score,
        "intent_key": intent,
    }


import threading
_eval_write_lock = threading.Lock()


def _merge_react_logs(*, output_dir: Path | None = None) -> None:
    """Merge per-question react logs into sorted react_log.txt."""
    out_dir = output_dir or EVAL_DIR
    log_dir = out_dir / "react_logs"
    if not log_dir.is_dir():
        return
    import re
    logs = list(log_dir.glob("*.log"))
    if not logs:
        return

    # Sort by paper number (extract from filename like "1_abc12345.log")
    def _log_sort_key(p):
        name = p.stem.split("_")[0]
        num = ""
        for ch in name:
            if ch.isdigit(): num += ch
            else: break
        return int(num) if num else 0

    logs.sort(key=_log_sort_key)

    out_path = out_dir / "react_log.txt"
    with out_path.open("w", encoding="utf-8") as f:
        for p in logs:
            f.write(p.read_text(encoding="utf-8"))
            f.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="Number of questions to evaluate (randomly sampled). 0 or negative = all.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for --limit sampling (default: 42).")
    parser.add_argument("--paper", default=None, help="Only evaluate questions from a specific paper (e.g. '1').")
    parser.add_argument("--react", action="store_true", help="Use ReAct multi-round retrieval.")
    parser.add_argument("--workers", type=int, default=1, help="Number of concurrent workers (default: 1).")
    parser.add_argument("--skills", action="store_true", help="Load Skill YAML configs and prompt registry.")
    parser.add_argument("--skills-dir", type=Path, default=PROJECT_ROOT / "config" / "skills", help="Directory containing Skill YAML files (default: config/skills).")
    parser.add_argument("--prompts-dir", type=Path, default=PROJECT_ROOT / "config" / "prompts", help="Directory containing Prompt Registry YAML files (default: config/prompts).")
    parser.add_argument("--dataset", type=Path, default=EVAL_DIR / "all_questions.json", help="Path to questions JSON file (default: data/eval/all_questions.json).")
    parser.add_argument("--output-dir", type=Path, default=EVAL_DIR, help="Directory for eval output (default: data/eval).")
    args = parser.parse_args()

    questions_file = args.dataset
    if not questions_file.exists():
        print(f"Questions file not found: {questions_file}", file=sys.stderr)
        return 1

    eval_output_dir = args.output_dir
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    all_q = json.loads(questions_file.read_text("utf-8"))
    # Sort by paper number
    def _paper_sort_key(q):
        paper = q.get("source_paper", "")
        num = ""
        for ch in paper:
            if ch.isdigit():
                num += ch
            else:
                break
        return int(num) if num else 0
    all_q.sort(key=_paper_sort_key)

    if args.paper:
        all_q = [q for q in all_q if q.get("source_paper", "").startswith(args.paper)]

    # Apply limit with random sampling (seed-controlled for reproducibility)
    effective_limit: int | None = args.limit if args.limit and args.limit > 0 else None
    all_q = select_questions(all_q, limit=effective_limit, seed=args.seed)

    # Build sampling metadata
    sampling_meta = build_sampling_metadata(effective_limit, args.seed, all_q)
    eval_metadata: dict | None = None
    if sampling_meta:
        eval_metadata = {"sampling": sampling_meta}
    elif effective_limit is None:
        eval_metadata = {"sampling": {"mode": "full"}}

    workers = max(1, args.workers)
    print(f"Evaluating {len(all_q)} questions with {workers} worker(s)\n")

    # Skill configs (optional, activated by --skills)
    skill_configs: dict[str, dict] = {}
    prompt_registry: dict[str, str] = {}
    if args.skills:
        skill_configs = _load_skill_configs(args.skills_dir)
        prompt_registry = _load_prompt_registry(args.prompts_dir)
        print(f"Loaded {len(skill_configs)} skill configs, {len(prompt_registry)} prompt registry entries\n")

    store = LocalStore(base_dir=PROJECT_ROOT)

    # ElementKG client
    elementkg_client = None
    try:
        from normalization import Neo4jIdentityClient, Neo4jIdentityConfig
        _ekg_config = Neo4jIdentityConfig.from_yaml(PROJECT_ROOT / "config" / "neo4j.yaml")
        elementkg_client = Neo4jIdentityClient(_ekg_config)
    except Exception:
        pass

    if args.react:
        from solver import ReActChemSolver
        solver = ReActChemSolver(store, elementkg_client=elementkg_client)
    else:
        solver = LLMChemSolver()

    results: list[dict] = []
    scores: list[float] = []
    by_intent: dict[str, list[float]] = {}

    if workers > 1:
        # Concurrent evaluation
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _eval_one, q, store,
                    LLMChemSolver() if not args.react else ReActChemSolver(store, elementkg_client=elementkg_client),
                    elementkg_client, args.react,
                    skill_configs.get(q.get("intent", ""), {}),
                    prompt_registry,
                ): q
                for q in all_q
            }
            with tqdm(total=len(all_q), desc="Evaluating", unit="q") as pbar:
                for f in as_completed(futures):
                    result = f.result()
                    with _eval_write_lock:
                        results.append(result)
                        scores.append(result["score"])
                        by_intent.setdefault(result["intent_key"], []).append(result["score"])
                        _save_results(results, scores, by_intent, output_dir=eval_output_dir, metadata=eval_metadata)
                    tqdm.write(f"  {result['id']}: score={result['judge_score']:.1f} | {result['system_answer'][:80]}...")
                    pbar.update(1)
    else:
        # Sequential evaluation
        for q in tqdm(all_q, desc="Evaluating", unit="q"):
            result = _eval_one(
                q, store, solver, elementkg_client, args.react,
                skill_configs.get(q.get("intent", ""), {}),
                prompt_registry,
            )
            results.append(result)
            scores.append(result["score"])
            by_intent.setdefault(result["intent_key"], []).append(result["score"])
            tqdm.write(f"  {result['id']}: score={result['judge_score']:.1f} | {result['system_answer'][:80]}...")
            _save_results(results, scores, by_intent, output_dir=eval_output_dir, metadata=eval_metadata)

    # Final save
    _save_results(results, scores, by_intent, output_dir=eval_output_dir, metadata=eval_metadata)

    # Merge per-question react logs into sorted react_log.txt
    _merge_react_logs(output_dir=eval_output_dir)
    avg = sum(scores) / len(scores) if scores else 0
    print(f"\n{'='*50}")
    print(f"RESULTS: {len(scores)} questions")
    print(f"  Average score: {avg:.2f}")
    print(f"  By intent:")
    for intent, sc_list in sorted(by_intent.items()):
        print(f"    {intent}: {sum(sc_list)/len(sc_list):.2f} ({len(sc_list)} questions)")
    print(f"\nSaved to {eval_output_dir / 'eval_results.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
