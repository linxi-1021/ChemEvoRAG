"""ReAct-based multi-round retrieval solver for ChemEvoRAG.

Implements §5.4 Controlled ReAct Retrieval Loop:
  query → retrieve() → assess evidence → insufficient? → refine query → retrieve() → ...
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

from tqdm import tqdm

from evidence import EvidencePackage, GroundedAnswer, SupportingEvidence
from retrieval import RetrievalRouter
from storage import LocalStore

from .llm_solver import LLMChemSolver
from .prompts import (
    ANSWER_GENERATION_EXAMPLES,
    ANSWER_GENERATION_SYSTEM,
)

_EVIDENCE_ASSESSMENT_SYSTEM = """\
You are a chemistry research assistant. Given a user's question and the \
retrieved evidence, determine if the evidence is sufficient to answer the question.

Return a JSON object:
{
  "sufficient": true/false,
  "reason": "<brief explanation>",
  "refined_query": "<a new search query to find missing information, or null>"
}

Rules:
- sufficient=true ONLY if the evidence contains specific, factual data that directly answers the question.
- If the evidence is partially helpful but missing key details, set sufficient=false and provide a refined_query that targets the missing information.
- refined_query should be a different search angle: use synonyms, expand abbreviations, try different keywords, or search for related entities.
- If the evidence is completely irrelevant, set sufficient=false and refined_query=null.
"""


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
    ) -> GroundedAnswer:
        """ReAct loop: iterative retrieval with evidence assessment."""
        accumulated_evidence: list = []
        accumulated_provenance = []
        retrieval_path: list[str] = []
        round_queries = [query]
        seen_evidence_ids: set[str] = set()

        for round_num in range(self.max_rounds):
            current_query = round_queries[-1]

            # Retrieve
            package = self.router.retrieve(
                current_query, doc_ids=doc_ids, top_k=top_k
            )

            # Accumulate new evidence (deduplicate)
            for c in package.candidate_evidence:
                if c.evidence_id not in seen_evidence_ids:
                    accumulated_evidence.append(c)
                    seen_evidence_ids.add(c.evidence_id)
            accumulated_provenance.extend(package.provenance)
            retrieval_path.extend(package.retrieval_path)

            # Assess evidence sufficiency
            if round_num < self.max_rounds - 1:  # Don't assess on last round
                assessment = self._assess_evidence(query, accumulated_evidence, round_num)

                if assessment.get("sufficient"):
                    break

                refined = assessment.get("refined_query")
                if refined and refined not in round_queries:
                    round_queries.append(refined)
                else:
                    break  # No new query to try

        # Build final answer from accumulated evidence
        merged_package = EvidencePackage(
            query=query,
            intent=package.intent,
            candidate_evidence=accumulated_evidence[:top_k * 2],
            provenance=accumulated_provenance,
            missing_slots=[] if accumulated_evidence else ["candidate_evidence"],
            retrieval_path=list(set(retrieval_path)),
        )

        answer = self.llm_solver.answer_from_package(merged_package)

        # Annotate with ReAct metadata
        if len(round_queries) > 1:
            note = f"[ReAct: {len(round_queries)} rounds, queries: {round_queries}]"
            answer.uncertainty = (
                (answer.uncertainty + " " + note) if answer.uncertainty else note
            )

        return answer

    def _assess_evidence(
        self, original_query: str, evidence: list, round_num: int
    ) -> dict:
        """LLM assesses whether current evidence is sufficient."""
        key = os.environ.get("API_KEY")
        if not key:
            return {"sufficient": len(evidence) > 0, "reason": "no LLM"}

        try:
            from openai import OpenAI  # type: ignore
        except ImportError:
            return {"sufficient": len(evidence) > 0, "reason": "no openai"}

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
                max_tokens=16384,
                extra_body={"thinking": {"type": "disabled"}},
            )
            raw = (response.choices[0].message.content or "").strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
            import sys
            tqdm.write(f"  [react] round {round_num + 1} raw: {repr(raw[:200])}", file=sys.stderr)
            result = json.loads(raw)
            if isinstance(result, dict):
                return result
        except Exception as e:
            import sys
            tqdm.write(f"  [react] assessment error: {type(e).__name__}: {e}", file=sys.stderr)

        return {"sufficient": len(evidence) > 0, "reason": "assessment failed"}
