"""LLM-based answer synthesis for ChemEvoRAG.

Wraps an OpenAI-compatible API to generate natural-language answers
from EvidencePackage objects.  Falls back to deterministic synthesis
when no API key is configured or the call fails.
"""

from __future__ import annotations

import json
import os
from typing import Any

from evidence import EvidencePackage, GroundedAnswer, SupportingEvidence

# Load .env for API_KEY if present.
from pathlib import Path as _Path
_dotenv = _Path(__file__).resolve().parents[2] / ".env"
if _dotenv.exists():
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(_dotenv)

from .prompts import (
    ANSWER_GENERATION_EXAMPLES,
    ANSWER_GENERATION_SYSTEM,
)


def _build_answer_prompt(package: EvidencePackage) -> str:
    """Build a prompt string from the EvidencePackage for answer generation."""

    lines: list[str] = []
    lines.append(f"QUESTION: {package.query}\n")
    lines.append(
        f"EVIDENCE (top {len(package.candidate_evidence)}):"
    )

    for i, ev in enumerate(package.candidate_evidence[:10]):
        summary = (ev.summary or ev.evidence_id)[:200]
        conf = ev.confidence or 0.0
        lines.append(
            f"[{i+1}] {ev.evidence_id} | {ev.evidence_type} | "
            f"{summary} | confidence: {conf:.2f}"
        )

    return "\n".join(lines)


def _call_openai(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> str | None:
    """Call OpenAI Chat Completions.  Returns the content string or None."""

    key = api_key or os.environ.get("API_KEY")
    url = base_url or os.environ.get("BASE_URL")
    if not key:
        return None

    try:
        from openai import OpenAI  # type: ignore
    except ImportError:
        return None

    client = OpenAI(api_key=key, base_url=url or None)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
    ]
    # Few-shot examples
    for ex_input, ex_output in ANSWER_GENERATION_EXAMPLES:
        messages.append({"role": "user", "content": ex_input})
        messages.append({"role": "assistant", "content": ex_output})
    messages.append({"role": "user", "content": user_prompt})

    try:
        response = client.chat.completions.create(
            model=model or os.environ.get("LLM_MODEL", "gpt-4o-mini"),
            messages=messages,
            temperature=0.1,
            max_tokens=int(os.environ.get("LLM_MAX_TOKENS", "16384")),
            extra_body={},
        )
        return response.choices[0].message.content
    except Exception:
        return None


class LLMChemSolver:
    """Answer chemistry questions with LLM synthesis, falling back to
    deterministic synthesis when the API is unavailable."""

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url

    def answer_from_package(self, package: EvidencePackage) -> GroundedAnswer:
        if not package.candidate_evidence:
            return GroundedAnswer(
                answer="No supported answer could be produced from the available evidence.",
                supporting_evidence=[],
                provenance=[],
                involved_entities=[],
                retrieval_path=package.retrieval_path,
                confidence=0.0,
                uncertainty="No candidate evidence was retrieved for the query.",
                raw_payload={"evidence_package": package.model_dump(mode="json")},
            )

        supporting = [
            SupportingEvidence(
                evidence_id=candidate.evidence_id,
                evidence_type=candidate.evidence_type,
                evidence=candidate.summary or "",
                source=candidate.source,
                confidence=candidate.confidence,
            )
            for candidate in package.candidate_evidence
        ]

        user_prompt = _build_answer_prompt(package)
        llm_output = _call_openai(
            ANSWER_GENERATION_SYSTEM,
            user_prompt,
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
        )

        if llm_output:
            try:
                parsed = json.loads(llm_output)
                answer_text = parsed.get("answer", llm_output)
                llm_conf = float(parsed.get("confidence", 0.5))
                uncertainty = parsed.get("uncertainty") or (
                    "Answer synthesised by LLM from retrieved evidence.  "
                    "LLM may hallucinate — verify against cited evidence."
                )
                return GroundedAnswer(
                    answer=answer_text,
                    supporting_evidence=supporting,
                    provenance=package.provenance,
                    involved_entities=[
                        candidate.evidence_id
                        for candidate in package.candidate_evidence
                        if candidate.evidence_type == "molecule"
                    ],
                    retrieval_path=package.retrieval_path,
                    confidence=round(llm_conf, 3),
                    uncertainty=uncertainty,
                    raw_payload={
                        "evidence_package": package.model_dump(mode="json"),
                        "llm_output_raw": llm_output,
                    },
                )
            except (json.JSONDecodeError, KeyError, ValueError):
                pass

        # ── fallback: deterministic ──────────────────────────────────
        from .roma_adapter import _synthesize_answer, _uncertainty

        confidence_values = [
            e.confidence for e in supporting if e.confidence is not None
        ]
        avg_confidence = (
            sum(confidence_values) / len(confidence_values)
            if confidence_values
            else 0.5
        )

        return GroundedAnswer(
            answer=_synthesize_answer(package),
            supporting_evidence=supporting,
            provenance=package.provenance,
            involved_entities=[
                c.evidence_id
                for c in package.candidate_evidence
                if c.evidence_type == "molecule"
            ],
            retrieval_path=package.retrieval_path,
            confidence=round(avg_confidence, 3),
            uncertainty=_uncertainty(package),
            raw_payload={
                "evidence_package": package.model_dump(mode="json"),
                "note": "LLM unavailable — used deterministic synthesis.",
            },
        )
