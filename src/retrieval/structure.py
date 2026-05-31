"""Structure-based retrieval: substructure, similarity, functional group search."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from evidence import CandidateEvidence, SourceProvenance
from storage.local_store import LocalStore

# RDKit lazy import
_rdkit_available: bool | None = None

# Load .env
_dotenv = Path(__file__).resolve().parents[2] / ".env"
if _dotenv.exists():
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(_dotenv)


def _check_rdkit() -> bool:
    global _rdkit_available
    if _rdkit_available is None:
        try:
            from rdkit import Chem  # noqa: F401
            _rdkit_available = True
        except ImportError:
            _rdkit_available = False
    return _rdkit_available


_STRUCTURE_QUERY_SYSTEM = """\
You are a chemistry assistant. Analyze the user's query and determine if it is \
asking about chemical structure (substructure, similarity, or functional group).

Return a JSON object:
{
  "is_structure_query": true/false,
  "search_type": "similarity" | "substructure" | "functional_group" | null,
  "smiles_or_smarts": "<SMILES or SMARTS pattern, or null>"
}

Rules:
- is_structure_query=true ONLY if the query is specifically about finding molecules \
  by their chemical structure, substructure, similarity, or functional group.
- is_structure_query=false for general chemistry questions (reactions, conditions, \
  yields, names, properties, abbreviations, etc.)
- For similarity: return a valid SMILES string (e.g. "CCO" for ethanol)
- For substructure: return a valid SMARTS pattern (e.g. "c1ccccc1" for benzene ring)
- For functional_group: return a valid SMARTS pattern (e.g. "[OX2H]" for hydroxyl)
- Use standard RDKit SMARTS syntax.

Examples:
- "CCO" → {"is_structure_query": true, "search_type": "similarity", "smiles_or_smarts": "CCO"}
- "含有苯环的分子" → {"is_structure_query": true, "search_type": "substructure", "smiles_or_smarts": "c1ccccc1"}
- "酯基" → {"is_structure_query": true, "search_type": "functional_group", "smiles_or_smarts": "[CX3](=O)[OX2]"}
- "What solvent gave the highest yield?" → {"is_structure_query": false, "search_type": null, "smiles_or_smarts": null}
- "What is DCHA?" → {"is_structure_query": false, "search_type": null, "smiles_or_smarts": null}
- "Which reaction has higher yield?" → {"is_structure_query": false, "search_type": null, "smiles_or_smarts": null}
"""


def _llm_analyze_structure_query(query: str) -> dict | None:
    """用 LLM 判断查询是否与化学结构相关，如果是则返回 SMILES/SMARTS。"""
    key = os.environ.get("API_KEY")
    if not key:
        return None

    try:
        from openai import OpenAI  # type: ignore
    except ImportError:
        return None

    client = OpenAI(api_key=key, base_url=os.environ.get("BASE_URL") or None)
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _STRUCTURE_QUERY_SYSTEM},
                {"role": "user", "content": query},
            ],
            temperature=0.0,
            max_tokens=256,
        )
        raw = (response.choices[0].message.content or "").strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        result = json.loads(raw)
        if isinstance(result, dict) and "is_structure_query" in result:
            return result
    except Exception:
        pass
    return None


def structure_search(
    query: str,
    store: LocalStore,
    *,
    elementkg_client: Any | None = None,
    top_k: int = 10,
) -> list[CandidateEvidence]:
    """结构检索入口。

    用 LLM 判断查询是否与化学结构相关，如果是则执行结构检索。
    对自然语言查询（如 "What solvent gave the highest yield?"）直接返回空。

    Args:
        query: 用户查询
        store: 本地 evidence 存储
        elementkg_client: ElementKG 客户端（可选）
        top_k: 返回数量
    """
    if not _check_rdkit():
        return []

    from rdkit import Chem

    # 用 LLM 分析查询
    analysis = _llm_analyze_structure_query(query)

    if analysis is None or not analysis.get("is_structure_query"):
        return []

    search_type = analysis.get("search_type", "")
    smiles_or_smarts = analysis.get("smiles_or_smarts", "")

    if not search_type or not smiles_or_smarts:
        return []

    # 收集候选 SMILES
    candidates_smiles = _collect_smiles(store, elementkg_client)

    if search_type == "similarity":
        # 验证 SMILES
        mol = Chem.MolFromSmiles(smiles_or_smarts)
        if mol is None:
            return []
        return _similarity_search(smiles_or_smarts, candidates_smiles, top_k)
    elif search_type == "substructure":
        return _substructure_search(smiles_or_smarts, candidates_smiles, top_k)
    elif search_type == "functional_group":
        return _substructure_search(smiles_or_smarts, candidates_smiles, top_k)
    return []


def _collect_smiles(store: LocalStore, elementkg_client: Any | None) -> list[dict]:
    """从本地 evidence 和 ElementKG 收集 SMILES。"""
    results: list[dict] = []

    # 本地 evidence
    try:
        doc_ids = [
            p.stem.replace(".molecules", "")
            for p in store.evidence_dir.glob("*.molecules.json")
        ]
    except Exception:
        doc_ids = []

    for doc_id in doc_ids:
        try:
            mols = store.load_molecules(doc_id)
        except Exception:
            continue
        for m in mols:
            smi = m.canonical_smiles or m.raw_smiles
            if smi:
                results.append({
                    "smiles": smi,
                    "source_id": m.molecule_card_id,
                    "doc_id": doc_id,
                    "names": m.names + m.aliases,
                    "source": "evidence",
                })

    # ElementKG
    if elementkg_client:
        try:
            with elementkg_client._driver.session() as session:
                r = session.run(
                    "MATCH (m:Molecule) WHERE m.smiles IS NOT NULL "
                    "RETURN m.id, m.smiles, m.iupac_name LIMIT 5000"
                )
                for rec in r:
                    results.append({
                        "smiles": rec["m.smiles"],
                        "source_id": rec["m.id"],
                        "doc_id": "elementkg",
                        "names": [rec["m.iupac_name"]] if rec["m.iupac_name"] else [],
                        "source": "elementkg",
                    })
        except Exception:
            pass

    return results


def _similarity_search(
    query: str, candidates: list[dict], top_k: int
) -> list[CandidateEvidence]:
    """结构相似性搜索（Tanimoto + Morgan fingerprint）。"""
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem

    query_mol = Chem.MolFromSmiles(query)
    if query_mol is None:
        return []

    query_fp = AllChem.GetMorganFingerprintAsBitVect(query_mol, 2)

    scored: list[tuple[float, dict]] = []
    for c in candidates:
        mol = Chem.MolFromSmiles(c["smiles"])
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2)
        sim = DataStructs.TanimotoSimilarity(query_fp, fp)
        if sim > 0.1:
            scored.append((sim, c))

    scored.sort(key=lambda x: x[0], reverse=True)

    results: list[CandidateEvidence] = []
    for sim, c in scored[:top_k]:
        results.append(_to_candidate(c, confidence=sim, search_type="similarity"))
    return results


def _substructure_search(
    query: str, candidates: list[dict], top_k: int
) -> list[CandidateEvidence]:
    """子结构匹配搜索。"""
    from rdkit import Chem

    pattern = Chem.MolFromSmarts(query)
    if pattern is None:
        mol = Chem.MolFromSmiles(query)
        if mol is not None:
            pattern = Chem.MolFromSmarts(Chem.MolToSmiles(mol))
    if pattern is None:
        return []

    results: list[CandidateEvidence] = []
    for c in candidates:
        mol = Chem.MolFromSmiles(c["smiles"])
        if mol is None:
            continue
        if mol.HasSubstructMatch(pattern):
            results.append(_to_candidate(c, confidence=0.9, search_type="substructure"))
            if len(results) >= top_k:
                break
    return results


def _to_candidate(c: dict, *, confidence: float, search_type: str) -> CandidateEvidence:
    """将内部候选转为 CandidateEvidence。"""
    names = c.get("names", [])
    summary = " | ".join(n for n in [c["smiles"]] + names if n)
    return CandidateEvidence(
        evidence_id=c["source_id"],
        evidence_type="molecule",
        summary=summary,
        structured_slots={
            "doc_id": c["doc_id"],
            "smiles": c["smiles"],
            "names": names,
            "search_type": search_type,
            "source": c.get("source", "unknown"),
        },
        source=SourceProvenance(doc_id=c["doc_id"]),
        confidence=confidence,
    )
