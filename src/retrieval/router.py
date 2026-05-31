"""Retrieval router — all-channel search with intent-based re-ranking."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from evidence import EvidencePackage
from storage import LocalStore

from .entity import entity_search
from .lexical import lexical_search
from .provenance import provenance_backtrack
from .reaction import reaction_event_search

# Load .env for API_KEY
_dotenv = Path(__file__).resolve().parents[2] / ".env"
if _dotenv.exists():
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(_dotenv)

_VALID_INTENTS = {
    "alias_resolution",
    "entity_lookup",
    "reaction_condition_query",
    "reaction_comparison",
    "property_query",
    "paper_local_question",
}

# Intent → (evidence_type boost, numeric_boost, table_boost)
_INTENT_BOOSTS: dict[str, tuple[str | None, bool, bool]] = {
    "alias_resolution": ("molecule", False, False),
    "entity_lookup": ("molecule", False, False),
    "reaction_condition_query": ("reaction_event", False, True),
    "reaction_comparison": ("reaction_event", True, True),
    "property_query": (None, True, True),
    "paper_local_question": (None, False, False),
}


class RetrievalRouter:
    """Search all channels, re-rank by intent, build EvidencePackage."""

    def __init__(
        self,
        store: LocalStore,
        *,
        default_top_k: int = 10,
        use_llm_intent: bool = True,
        elementkg_client: Any | None = None,
    ) -> None:
        self.store = store
        self.default_top_k = default_top_k
        self.use_llm_intent = use_llm_intent
        self.elementkg_client = elementkg_client

    def retrieve(
        self,
        query: str,
        *,
        doc_ids: list[str] | None = None,
        top_k: int | None = None,
    ) -> EvidencePackage:
        # Intent: LLM → keyword fallback
        if self.use_llm_intent:
            resolved_intent = _llm_infer_intent(query) or infer_intent(query)
        else:
            resolved_intent = infer_intent(query)

        limit = top_k or self.default_top_k

        # Step 1: Query Rewrite — 用 ElementKG 扩展实体名称
        expanded_queries = self._expand_entities(query)

        # Step 2: 全通道检索
        candidates = []
        for eq in expanded_queries:
            candidates.extend(
                entity_search(eq, self.store, doc_ids=doc_ids, top_k=limit)
            )
        candidates.extend(
            lexical_search(query, self.store, doc_ids=doc_ids, top_k=limit * 5)
        )
        candidates.extend(
            reaction_event_search(query, self.store, doc_ids=doc_ids, top_k=limit * 2)
        )
        # Dense retrieval — optional, fails gracefully if index/model missing
        try:
            from .dense import dense_search
            candidates.extend(
                dense_search(query, doc_ids=doc_ids, top_k=limit * 2)
            )
        except Exception:
            pass

        # Structure retrieval — optional, needs RDKit
        try:
            from .structure import structure_search
            candidates.extend(
                structure_search(
                    query, self.store,
                    elementkg_client=self.elementkg_client,
                    top_k=limit,
                )
            )
        except Exception:
            pass

        # Step 3: Entity Resolution — 用 ElementKG 填充分子信息
        if self.elementkg_client:
            candidates = self._resolve_entities(candidates)

        # Step 4: Intent-based re-ranking (before dedup/limit)
        candidates = _intent_rerank(candidates, resolved_intent, query)
        candidates = _dedupe_candidates(candidates)
        candidates = candidates[:limit]

        provenance = [
            source
            for candidate in candidates
            for source in [
                candidate.source
                or provenance_backtrack(
                    candidate.evidence_id, self.store, doc_ids=doc_ids
                )
            ]
            if source is not None
        ]

        retrieval_path = [
            "entity_retrieval", "lexical_retrieval",
            "reaction_retrieval", "dense_retrieval",
        ]

        return EvidencePackage(
            query=query,
            intent=resolved_intent,  # type: ignore[arg-type]
            candidate_evidence=candidates,
            provenance=provenance,
            missing_slots=[] if candidates else ["candidate_evidence"],
            retrieval_path=retrieval_path,
        )

    def _expand_entities(self, query: str) -> list[str]:
        """用 ElementKG 扩展查询中的实体名称。

        例如 "DCHA" → ["DCHA", "dicyclohexylamine", "N-cyclohexylcyclohexanamine"]
        """
        if not self.elementkg_client:
            return [query]

        # 提取查询中的潜在实体名称（首字母大写的词，或化学常见模式）
        words = re.findall(r"[A-Za-z0-9\-]+", query)
        expanded = [query]
        for word in words:
            if len(word) < 2:
                continue
            # 尝试用 ElementKG 查找别名
            try:
                candidates = self.elementkg_client.find_by_alias(word)
                for c in candidates:
                    if c.canonical_name and c.canonical_name.lower() != word.lower():
                        expanded.append(c.canonical_name)
                    for alias in c.aliases:
                        if alias and alias.lower() != word.lower():
                            expanded.append(alias)
            except Exception:
                pass
        return list(set(expanded))

    def _resolve_entities(self, candidates: list) -> list:
        """用 ElementKG 填充分子候选的别名/SMILES/InChIKey。"""
        resolved = []
        for c in candidates:
            if c.evidence_type == "molecule" and c.structured_slots:
                # 尝试用 InChIKey 或 SMILES 查找
                inchikey = c.structured_slots.get("inchi_key")
                smiles = c.structured_slots.get("canonical_smiles") or c.structured_slots.get("raw_smiles")

                identity = None
                if inchikey:
                    try:
                        results = self.elementkg_client.find_by_inchikey(inchikey)
                        if results:
                            identity = results[0]
                    except Exception:
                        pass
                if not identity and smiles:
                    try:
                        results = self.elementkg_client.find_by_smiles(smiles)
                        if results:
                            identity = results[0]
                    except Exception:
                        pass

                if identity:
                    # 用 ElementKG 信息丰富 summary
                    parts = [c.summary or ""]
                    if identity.canonical_name:
                        parts.append(identity.canonical_name)
                    parts.extend(identity.aliases)
                    c.summary = " | ".join(p for p in parts if p)
            resolved.append(c)
        return resolved


def _llm_infer_intent(query: str) -> str | None:
    """Call LLM to classify query intent. Returns intent string or None on failure."""
    from solver.prompts import QUERY_UNDERSTANDING_EXAMPLES, QUERY_UNDERSTANDING_SYSTEM

    key = os.environ.get("API_KEY")
    if not key:
        return None

    try:
        from openai import OpenAI  # type: ignore
    except ImportError:
        return None

    client = OpenAI(api_key=key, base_url=os.environ.get("BASE_URL") or None)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": QUERY_UNDERSTANDING_SYSTEM},
    ]
    for ex_input, ex_output in QUERY_UNDERSTANDING_EXAMPLES:
        messages.append({"role": "user", "content": ex_input})
        messages.append({"role": "assistant", "content": ex_output})
    messages.append({"role": "user", "content": query})

    try:
        response = client.chat.completions.create(
            model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
            messages=messages,
            temperature=0.0,
            max_tokens=256,
        )
        raw = (response.choices[0].message.content or "").strip()
        # Parse JSON response
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        parsed = json.loads(raw)
        intent = parsed.get("intent", "")
        if intent in _VALID_INTENTS:
            return intent
    except Exception:
        pass
    return None


def infer_intent(query: str) -> str:
    q = query.lower()

    # Alias resolution
    if any(
        kw in q
        for kw in (
            "alias", "refer to", "what is", "abbreviation", "stand for",
            "denote", "是什么", "指的是", "别名", "full name",
        )
    ):
        return "alias_resolution"

    # Entity lookup
    if any(
        kw in q
        for kw in (
            "structure", "smiles", "iupac", "molecular formula", "identity",
            "name of", "compound", "chemical name", "composition",
        )
    ):
        return "entity_lookup"

    # Reaction comparison — check before reaction_condition (more specific)
    if any(
        kw in q
        for kw in (
            "higher yield", "lower yield", "better yield", "compare",
            "versus", " vs ", "which.*higher", "which.*give",
        )
    ) or (
        re.search(r"which\b.*\b(yield|higher|better|gave)", q) is not None
    ):
        return "reaction_comparison"

    # Reaction condition
    if any(
        kw in q
        for kw in (
            "solvent", "catalyst", "temperature", "loading", "mol%",
            "condition", "procedure", "atmosphere", "balloon",
            "reaction time", "catalytic",
        )
    ):
        return "reaction_condition_query"

    # Property query
    if any(
        kw in q
        for kw in (
            "yield", "boiling point", "melting point", " mp ", " bp ",
            "isolated", "mass", "purity", "nmr yield", "percent",
        )
    ):
        return "property_query"

    return "paper_local_question"


def _intent_rerank(
    candidates: list, intent: str, query: str
) -> list:
    """Boost candidate scores based on intent and query features."""
    boost_type, numeric_boost, table_boost = _INTENT_BOOSTS.get(
        intent, (None, False, False)
    )

    # Extract numeric patterns from query for matching
    query_numbers = set(re.findall(r"\d+(?:\.\d+)?%?", query)) if numeric_boost else set()

    for c in candidates:
        bonus = 0.0

        # Type boost
        if boost_type and c.evidence_type == boost_type:
            bonus += 2.0

        # Table block boost — strong boost when query asks for comparisons/properties
        if table_boost and c.evidence_type == "document_block":
            slots = c.structured_slots or {}
            if slots.get("block_type") == "table":
                bonus += 5.0

        # Numeric matching boost
        if query_numbers:
            summary = (c.summary or "").lower()
            matched = sum(1 for n in query_numbers if n in summary)
            if matched:
                bonus += matched * 1.0

        c.confidence = min(1.0, (c.confidence or 0.0) + bonus / 10.0)

    candidates.sort(key=lambda c: c.confidence or 0.0, reverse=True)
    return candidates


def _dedupe_candidates(candidates):
    seen: set[tuple[str, str]] = set()
    result = []
    for candidate in candidates:
        key = (candidate.evidence_type, candidate.evidence_id)
        if key in seen:
            continue
        seen.add(key)
        result.append(candidate)
    return result
