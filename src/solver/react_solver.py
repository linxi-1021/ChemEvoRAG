"""ReAct-based multi-round retrieval solver for ChemEvoRAG.

Implements §5.4 Controlled ReAct Retrieval Loop:
  query → retrieve() → assess evidence → insufficient? → refine query → retrieve() → ...
"""

from __future__ import annotations

import json
import os
from typing import Any

from evidence import EvidencePackage, GroundedAnswer, SupportingEvidence
from retrieval import RetrievalRouter
from storage import LocalStore

from .llm_solver import LLMChemSolver
from .prompts import (
    ANSWER_GENERATION_EXAMPLES,
    ANSWER_GENERATION_SYSTEM,
    EVIDENCE_ASSESSMENT_SYSTEM,
)

import threading
_react_log_lock = threading.Lock()


class _AssessmentFailed(Exception):
    pass


_EVIDENCE_ASSESSMENT_SYSTEM = EVIDENCE_ASSESSMENT_SYSTEM


class ReActChemSolver:
    """Multi-round iterative retrieval solver.

    Flow:
      Round 1: retrieve(query) → assess evidence
        ├─ sufficient → generate answer
        └─ insufficient → refined_query → Round 2
      Round 2: retrieve(refined_query) → assess evidence
        ├─ sufficient → generate answer
        └─ insufficient → refined_query → Round 3
      Round 3: final answer with accumulated evidence
    """

    def __init__(
        self,
        store: LocalStore,
        *,
        elementkg_client: Any | None = None,
        max_rounds: int = 3,
        use_llm_intent: bool = True,
    ) -> None:
        self.store = store
        self.max_rounds = max_rounds
        self.router = RetrievalRouter(
            store,
            elementkg_client=elementkg_client,
            use_llm_intent=use_llm_intent,
        )
        self.llm_solver = LLMChemSolver()

    def answer(
        self,
        query: str,
        *,
        doc_ids: list[str] | None = None,
        top_k: int = 8,
        _retry: bool = True,
    ) -> GroundedAnswer:
        """ReAct loop: iterative retrieval with evidence assessment.

        If assessment LLM fails in any round, retries the entire process
        from scratch (up to 1 retry).
        """
        try:
            return self._answer_inner(query, doc_ids=doc_ids, top_k=top_k)
        except _AssessmentFailed:
            if _retry:
                import sys
                print(f"\n[ReAct] Assessment failed, retrying from scratch...", file=sys.stderr)
                return self.answer(query, doc_ids=doc_ids, top_k=top_k, _retry=False)

        # Last resort: single-pass with no assessment
        return self._answer_fallback(query, doc_ids=doc_ids, top_k=top_k)

    def _answer_inner(
        self,
        query: str,
        *,
        doc_ids: list[str] | None = None,
        top_k: int = 8,
    ) -> GroundedAnswer:
        """ReAct loop: iterative retrieval with evidence assessment."""
        import sys
        from pathlib import Path

        # Log file for ReAct process
        log_path = Path(__file__).resolve().parents[2] / "data" / "eval" / "react_log.txt"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_lines: list[str] = []

        def _log(msg: str) -> None:
            with _react_log_lock:
                print(msg, file=sys.stderr)
                log_lines.append(msg)

        accumulated_evidence: list = []
        accumulated_provenance = []
        retrieval_path: list[str] = []
        round_queries = [query]
        seen_evidence_ids: set[str] = set()

        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _log(f"\n{'='*60}")
        _log(f"[ReAct] Time: {timestamp}")
        _log(f"[ReAct] Original query: {query}")

        for round_num in range(self.max_rounds):
            current_query = round_queries[-1]
            _log(f"\n{'─'*50}")
            _log(f"[ReAct] Round {round_num + 1}/{self.max_rounds}")
            _log(f"[ReAct] Query: {current_query}")

            # Retrieve
            package = self.router.retrieve(
                current_query, doc_ids=doc_ids, top_k=top_k
            )

            # Accumulate new evidence (deduplicate)
            new_count = 0
            for c in package.candidate_evidence:
                if c.evidence_id not in seen_evidence_ids:
                    accumulated_evidence.append(c)
                    seen_evidence_ids.add(c.evidence_id)
                    new_count += 1
            accumulated_provenance.extend(package.provenance)
            retrieval_path.extend(package.retrieval_path)
            _log(f"[ReAct] Retrieved: {len(package.candidate_evidence)} candidates, {new_count} new (total: {len(accumulated_evidence)})")

            # Show top evidence (terminal: short, log: full)
            for i, ev in enumerate(package.candidate_evidence[:3]):
                summary = ev.summary or ""
                print(f"[ReAct]   [{i+1}] {ev.evidence_type} | {ev.evidence_id} | {summary[:100]}", file=sys.stderr)
                log_lines.append(f"[ReAct]   [{i+1}] {ev.evidence_type} | {ev.evidence_id} | {summary}")

            # Assess evidence sufficiency
            if round_num < self.max_rounds - 1:  # Don't assess on last round
                assessment = self._assess_evidence(query, accumulated_evidence, round_num)
                sufficient = assessment.get("sufficient", False)
                reason = assessment.get("reason", "")
                refined = assessment.get("refined_query")
                _log(f"[ReAct] Assessment: sufficient={sufficient}")
                _log(f"[ReAct] Reason: {reason}")
                if refined:
                    _log(f"[ReAct] Refined query: {refined}")

                # 评估 LLM 失败 → 抛异常触发从头重试
                if reason == "assessment failed":
                    _log(f"[ReAct] → Assessment failed, will retry from scratch.")
                    raise _AssessmentFailed()

                if sufficient:
                    _log(f"[ReAct] → Evidence sufficient, stopping.")
                    break

                if refined and refined not in round_queries:
                    round_queries.append(refined)
                else:
                    _log(f"[ReAct] → No refined query, stopping.")
                    break

        # Build final answer from accumulated evidence
        _log(f"\n{'─'*50}")
        _log(f"[ReAct] Finished: {len(round_queries)} rounds, {len(accumulated_evidence)} evidence items")
        _log(f"[ReAct] All queries: {round_queries}")

        merged_package = EvidencePackage(
            query=query,
            intent=package.intent,
            candidate_evidence=accumulated_evidence[:top_k * 2],
            provenance=accumulated_provenance,
            missing_slots=[] if accumulated_evidence else ["candidate_evidence"],
            retrieval_path=list(set(retrieval_path)),
        )

        answer = self.llm_solver.answer_from_package(merged_package)

        # Log final answer
        _log(f"\n[ReAct] Final answer: {answer.answer[:200]}")
        _log(f"[ReAct] Confidence: {answer.confidence}")
        if answer.uncertainty:
            _log(f"[ReAct] Uncertainty: {answer.uncertainty[:200]}")

        # Append log to file (not overwrite)
        with log_path.open("a", encoding="utf-8") as f:
            f.write("\n".join(log_lines) + "\n\n")

        # Annotate with ReAct metadata
        if len(round_queries) > 1:
            note = f"[ReAct: {len(round_queries)} rounds, queries: {round_queries}]"
            answer.uncertainty = (
                (answer.uncertainty + " " + note) if answer.uncertainty else note
            )

        return answer

    def _answer_fallback(
        self,
        query: str,
        *,
        doc_ids: list[str] | None = None,
        top_k: int = 8,
    ) -> GroundedAnswer:
        """Single-pass fallback when ReAct assessment consistently fails."""
        from retrieval import RetrievalRouter
        router = RetrievalRouter(self.store, elementkg_client=self.elementkg_client)
        package = router.retrieve(query, doc_ids=doc_ids, top_k=top_k)
        answer = self.llm_solver.answer_from_package(package)
        note = "[ReAct: assessment failed, used single-pass fallback]"
        answer.uncertainty = (answer.uncertainty + " " + note) if answer.uncertainty else note
        return answer

    def _assess_evidence(
        self, original_query: str, evidence: list, round_num: int
    ) -> dict:
        """LLM assesses whether current evidence is sufficient."""
        key = os.environ.get("API_KEY")
        if not key:
            return {"sufficient": False, "reason": "no API key"}

        try:
            from openai import OpenAI  # type: ignore
        except ImportError:
            return {"sufficient": False, "reason": "openai not installed"}

        client = OpenAI(api_key=key, base_url=os.environ.get("BASE_URL") or None)
        model = os.environ.get("LLM_MODEL", "gpt-4o-mini")

        # Build evidence summary
        ev_lines = []
        for i, ev in enumerate(evidence[:10]):
            summary = (ev.summary or ev.evidence_id)[:150]
            ev_lines.append(f"[{i+1}] {ev.evidence_type} | {ev.evidence_id} | {summary}")

        user_msg = (
            f"QUESTION: {original_query}\n\n"
            f"RETRIEVED EVIDENCE (round {round_num + 1}, {len(evidence)} items):\n"
            + "\n".join(ev_lines)
        )

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _EVIDENCE_ASSESSMENT_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=int(os.environ.get("LLM_MAX_TOKENS", "16384")),
                extra_body={},
                timeout=60,  # 60 second timeout
            )
            raw = (response.choices[0].message.content or "").strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
            result = json.loads(raw)
            if isinstance(result, dict):
                return result
        except Exception:
            pass

        # 重试一次
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _EVIDENCE_ASSESSMENT_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.1,
                max_tokens=int(os.environ.get("LLM_MAX_TOKENS", "16384")),
                extra_body={},
                timeout=60,
            )
            raw = (response.choices[0].message.content or "").strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
            result = json.loads(raw)
            if isinstance(result, dict):
                return result
        except Exception:
            pass

        return {"sufficient": False, "reason": "assessment failed"}
