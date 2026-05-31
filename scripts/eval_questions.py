#!/usr/bin/env python
"""Evaluate ChemEvoRAG against generated QA pairs from data/eval/all_questions.json.

For each question: run retrieval + LLM synthesis, compare answer to ground truth
using a second LLM call as judge.  Saves detailed results to data/eval/eval_results.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
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
            max_tokens=16384,
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="Only evaluate first N questions.")
    parser.add_argument("--paper", default=None, help="Only evaluate questions from a specific paper (e.g. '1').")
    args = parser.parse_args()

    questions_file = EVAL_DIR / "all_questions.json"
    if not questions_file.exists():
        print("all_questions.json not found. Run generate_eval_questions.py first.", file=sys.stderr)
        return 1

    all_q = json.loads(questions_file.read_text("utf-8"))
    # 按论文编号数字排序（1, 2, 3, ..., 10, 11, ...）
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
    if args.limit:
        all_q = all_q[:args.limit]

    print(f"Evaluating {len(all_q)} questions\n")

    store = LocalStore(base_dir=PROJECT_ROOT)
    solver = LLMChemSolver()

    results: list[dict] = []
    scores: list[float] = []
    by_intent: dict[str, list[float]] = {}

    for q in tqdm(all_q, desc="Evaluating", unit="q"):
        question = q["question"]
        ground_truth = q["ground_truth_answer"]
        key_entities = q.get("key_entities", [])
        intent = q.get("intent", "unknown")
        source = q.get("source_paper", "?")

        # Extract doc_id from source_paper
        doc_id = source.replace(".pdf", "") if source else None

        try:
            from retrieval.router import RetrievalRouter
            # 尝试创建 ElementKG 客户端
            elementkg_client = None
            try:
                from normalization import Neo4jIdentityClient, Neo4jIdentityConfig
                _ekg_config = Neo4jIdentityConfig.from_yaml(PROJECT_ROOT / "config" / "neo4j.yaml")
                elementkg_client = Neo4jIdentityClient(_ekg_config)
            except Exception:
                pass
            router = RetrievalRouter(store, elementkg_client=elementkg_client)
            package = router.retrieve(question, doc_ids=[doc_id] if doc_id else None)
            answer_obj = solver.answer_from_package(package)
            system_answer = answer_obj.answer
            evidence_ids = [s.evidence_id for s in answer_obj.supporting_evidence]
            confidence = answer_obj.confidence
        except Exception as exc:
            system_answer = f"[ERROR: {exc}]"
            evidence_ids = []
            confidence = 0.0

        # Judge
        judge = _judge_answer(system_answer, ground_truth, key_entities)
        score = judge.get("score", 0.0)

        scores.append(score)
        by_intent.setdefault(intent, []).append(score)

        result = {
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
        }
        results.append(result)
        tqdm.write(f"  {q['id']}: score={score:.1f} | {system_answer[:80]}...")

    # Summary
    avg = sum(scores) / len(scores) if scores else 0
    print(f"\n{'='*50}")
    print(f"RESULTS: {len(scores)} questions")
    print(f"  Average score: {avg:.2f}")
    print(f"  By intent:")
    for intent, sc_list in sorted(by_intent.items()):
        print(f"    {intent}: {sum(sc_list)/len(sc_list):.2f} ({len(sc_list)} questions)")

    # Save
    out_path = EVAL_DIR / "eval_results.json"
    out_path.write_text(json.dumps({
        "summary": {"total": len(scores), "average_score": avg, "by_intent": {k: sum(v)/len(v) for k, v in by_intent.items()}},
        "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
