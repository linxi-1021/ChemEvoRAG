#!/usr/bin/env python
"""Run a local ChemEvoRAG query over stored evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

# Load .env if present (API_KEY, etc.)
_dotenv_path = PROJECT_ROOT / ".env"
if _dotenv_path.exists():
    from dotenv import load_dotenv
    load_dotenv(_dotenv_path)

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from retrieval import RetrievalRouter  # noqa: E402
from solver import ChemRAGSolver, LLMChemSolver  # noqa: E402
from storage import LocalStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Question to answer from stored evidence.")
    parser.add_argument(
        "--doc-id", action="append", default=None,
        help="Restrict to a doc id. Can be repeated.",
    )
    parser.add_argument(
        "--top-k", type=int, default=8,
        help="Maximum candidate evidence count.",
    )
    parser.add_argument(
        "--base-dir", default=str(PROJECT_ROOT),
        help="Project base directory.",
    )
    parser.add_argument(
        "--use-llm", action="store_true",
        help="Synthesise answer with LLM. Requires API_KEY env var.",
    )
    parser.add_argument(
        "--react", action="store_true",
        help="Use ReAct multi-round retrieval (implies --use-llm).",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output full GroundedAnswer as JSON instead of plain text.",
    )
    args = parser.parse_args()

    store = LocalStore(base_dir=args.base_dir)

    # 尝试创建 ElementKG 客户端
    elementkg_client = None
    try:
        from normalization import Neo4jIdentityClient, Neo4jIdentityConfig
        config = Neo4jIdentityConfig.from_yaml(PROJECT_ROOT / "config" / "neo4j.yaml")
        elementkg_client = Neo4jIdentityClient(config)
    except Exception:
        pass

    if args.react:
        from solver import ReActChemSolver
        solver = ReActChemSolver(store, elementkg_client=elementkg_client)
        answer = solver.answer(args.query, doc_ids=args.doc_id, top_k=args.top_k)
    elif args.use_llm:
        solver = LLMChemSolver()
        package = RetrievalRouter(store, elementkg_client=elementkg_client).retrieve(
            args.query, doc_ids=args.doc_id, top_k=args.top_k
        )
        answer = solver.answer_from_package(package)
    else:
        solver = ChemRAGSolver(store)
        answer = solver.answer(args.query, doc_ids=args.doc_id, top_k=args.top_k)

    if args.json:
        print(json.dumps(answer.model_dump(mode="json"), ensure_ascii=False, indent=2))
    else:
        print(answer.answer)
        if answer.supporting_evidence:
            ids = [e.evidence_id for e in answer.supporting_evidence]
            print(f"\nEvidence: {', '.join(ids)}")
        print(f"Confidence: {answer.confidence}")
        if answer.uncertainty:
            print(f"Uncertainty: {answer.uncertainty}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

