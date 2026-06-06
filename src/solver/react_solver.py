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
from .prompt_context import PromptContext

import threading
_react_log_lock = threading.Lock()


class _AssessmentFailed(Exception):
    pass


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
        prompt_context: PromptContext | None = None,
    ) -> GroundedAnswer:
        """ReAct loop: iterative retrieval with evidence assessment.

        If assessment LLM fails in any round, retries the entire process
        from scratch (up to 1 retry).
        """
        try:
            return self._answer_inner(
                query, doc_ids=doc_ids, top_k=top_k,
                prompt_context=prompt_context,
            )
        except _AssessmentFailed:
            if _retry:
                import sys
                print(f"\n[ReAct] Assessment failed, retrying from scratch...", file=sys.stderr)
                return self.answer(
                    query, doc_ids=doc_ids, top_k=top_k, _retry=False,
                    prompt_context=prompt_context,
                )

        # Last resort: single-pass with no assessment
        return self._answer_fallback(
            query, doc_ids=doc_ids, top_k=top_k,
            prompt_context=prompt_context,
        )

    def _answer_inner(
        self,
        query: str,
        *,
        doc_ids: list[str] | None = None,
        top_k: int = 8,
        prompt_context: PromptContext | None = None,
    ) -> GroundedAnswer:
        """ReAct loop: iterative retrieval with evidence assessment."""
        import sys
        from pathlib import Path

        # Log file for ReAct process — one per question (named by doc_id + short query hash)
        log_dir = Path(__file__).resolve().parents[2] / "data" / "eval" / "react_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        import hashlib
        doc_tag = doc_ids[0] if doc_ids else "all"
        query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
        log_path = log_dir / f"{doc_tag}_{query_hash}.log"
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
                assessment = self._assess_evidence(
                    query, accumulated_evidence, round_num,
                    prompt_context=prompt_context,
                )
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
                    # Check if refined query is too similar to the previous one
                    if _query_too_similar(refined, round_queries[-1]):
                        _log(f"[ReAct] Refined query too similar to previous, diversifying...")
                        refined = _diversify_query(refined, query, round_num, accumulated_evidence)
                        _log(f"[ReAct] Diversified query: {refined}")
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

        answer = self.llm_solver.answer_from_package(
            merged_package, prompt_context=prompt_context,
        )

        # Log final answer
        _log(f"\n[ReAct] Final answer: {answer.answer[:200]}")
        _log(f"[ReAct] Confidence: {answer.confidence}")
        if answer.uncertainty:
            _log(f"[ReAct] Uncertainty: {answer.uncertainty[:200]}")

        # Write log to file (overwrite — one file per question)
        with log_path.open("w", encoding="utf-8") as f:
            f.write("\n".join(log_lines) + "\n")

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
        prompt_context: PromptContext | None = None,
    ) -> GroundedAnswer:
        """Single-pass fallback when ReAct assessment consistently fails."""
        from retrieval import RetrievalRouter
        router = RetrievalRouter(self.store, elementkg_client=self.elementkg_client)
        package = router.retrieve(query, doc_ids=doc_ids, top_k=top_k)
        answer = self.llm_solver.answer_from_package(
            package, prompt_context=prompt_context,
        )
        note = "[ReAct: assessment failed, used single-pass fallback]"
        answer.uncertainty = (answer.uncertainty + " " + note) if answer.uncertainty else note
        return answer

    def _assess_evidence(
        self, original_query: str, evidence: list, round_num: int,
        *,
        prompt_context: PromptContext | None = None,
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

        # Resolve assessment prompt: prompt_context override → default
        assessment_prompt = _resolve_assessment_prompt(prompt_context)

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
                    {"role": "system", "content": assessment_prompt},
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
                    {"role": "system", "content": assessment_prompt},
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


def _resolve_assessment_prompt(prompt_context: PromptContext | None) -> str:
    """Resolve the system prompt for evidence assessment.

    Priority: prompt_context override → default from prompts module.
    """
    if prompt_context and prompt_context.evidence_assessment_system:
        return prompt_context.evidence_assessment_system
    # Dynamic fallback: read from prompts module at call time, not import time
    import solver.prompts as _prompts_mod
    return _prompts_mod.EVIDENCE_ASSESSMENT_SYSTEM


def _query_too_similar(q1: str, q2: str) -> bool:
    """Check if two queries are too similar (>80% word overlap)."""
    words1 = set(q1.lower().split())
    words2 = set(q2.lower().split())
    if not words1 or not words2:
        return True
    overlap = len(words1 & words2) / max(len(words1), len(words2))
    return overlap > 0.8


def _diversify_query(
    refined: str,
    original: str,
    round_num: int,
    accumulated_evidence: list,
) -> str:
    """When refined query is too similar to the previous one, apply a
    rule-based diversification strategy to generate a more different query.

    Strategies (by round):
      Round 1 (→2): Try shorter keyword-only query
      Round 2 (→3): Try adding "experimental section" or "table" context
    """
    import re

    # Extract entity names from accumulated evidence summaries
    evidence_names = set()
    for ev in accumulated_evidence:
        summary = ev.summary or ""
        # Look for compound-like tokens (capitalized words, alphanumeric labels)
        for token in re.findall(r"\b[A-Za-z][A-Za-z0-9\-]{1,15}\b", summary):
            evidence_names.add(token)

    # Strategy 1: Extract the core entity name and search for it alone
    if round_num == 0:
        # First diversification: find the most specific entity in the query
        # and search for it with common experimental context terms
        words = original.split()
        # Find capitalized words (likely compound names/entities)
        entities = [w for w in words if w[0:1].isupper() and len(w) > 1]
        if entities:
            entity = entities[-1]  # Use the last entity (often the target)
            return f"{entity} experimental section characterization data"

    # Strategy 2: Add "table" or "scheme" context
    if round_num == 1:
        # Look for table/scheme mentions in evidence
        for ev in accumulated_evidence:
            summary = ev.summary or ""
            if "table" in summary.lower() or "entry" in summary.lower():
                return f"{original} table entry yields conditions"
            if "scheme" in summary.lower():
                return f"{original} scheme synthesis procedure"

    # Fallback: Return original with added diversity terms
    diversity_terms = ["experimental", "characterization", "synthesis procedure",
                       "conditions", "supporting information"]
    term = diversity_terms[round_num % len(diversity_terms)]
    # Remove the term if already present
    query_words = refined.split()
    filtered = [w for w in query_words if w.lower() != term.lower()]
    filtered.append(term)
    return " ".join(filtered)
