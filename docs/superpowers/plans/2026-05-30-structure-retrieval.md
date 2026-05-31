# Structure Retrieval 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现结构检索通道，支持子结构匹配、结构相似性、官能团查询。

**Architecture:** 新建 `src/retrieval/structure.py`，用 RDKit 做分子指纹计算和相似度比较，搜索本地 evidence 和 ElementKG Neo4j。

**Tech Stack:** RDKit, Neo4j (via elementkg_client)

---

### Task 1: 创建结构检索模块

**Files:**
- Create: `src/retrieval/structure.py`

- [ ] **Step 1: 编写结构检索模块**

Create `src/retrieval/structure.py`:
```python
"""Structure-based retrieval: substructure, similarity, functional group search."""

from __future__ import annotations

from typing import Any

from evidence import CandidateEvidence, SourceProvenance
from storage.local_store import LocalStore

# RDKit lazy import
_rdkit_available: bool | None = None


def _check_rdkit() -> bool:
    global _rdkit_available
    if _rdkit_available is None:
        try:
            from rdkit import Chem  # noqa: F401
            _rdkit_available = True
        except ImportError:
            _rdkit_available = False
    return _rdkit_available


# 官能团 SMARTS 模式
FUNCTIONAL_GROUPS = {
    "hydroxyl": ("[OX2H]", ["hydroxyl", "alcohol", "oh"]),
    "carbonyl": ("[CX3]=[OX1]", ["carbonyl", "c=o"]),
    "carboxyl": ("[CX3](=O)[OX2H]", ["carboxyl", "cooh"]),
    "amine": ("[NX3;H2]", ["amine", "nh2"]),
    "ester": ("[CX3](=O)[OX2]", ["ester", "coo"]),
    "phenyl": ("c1ccccc1", ["phenyl", "benzene", "aromatic"]),
    "halogen": ("[F,Cl,Br,I]", ["halogen", "fluoro", "chloro", "bromo", "iodo"]),
}


def structure_search(
    query: str,
    store: LocalStore,
    *,
    elementkg_client: Any | None = None,
    search_type: str = "auto",
    top_k: int = 10,
) -> list[CandidateEvidence]:
    """结构检索入口。

    Args:
        query: SMILES 字符串、分子名称或官能团名称
        store: 本地 evidence 存储
        elementkg_client: ElementKG 客户端（可选）
        search_type: "substructure" | "similarity" | "functional_group" | "auto"
        top_k: 返回数量
    """
    if not _check_rdkit():
        return []

    from rdkit import Chem

    # 自动检测查询类型
    if search_type == "auto":
        search_type = _detect_search_type(query)

    # 收集候选 SMILES
    candidates_smiles = _collect_smiles(store, elementkg_client)

    if search_type == "similarity":
        return _similarity_search(query, candidates_smiles, top_k)
    elif search_type == "substructure":
        return _substructure_search(query, candidates_smiles, top_k)
    elif search_type == "functional_group":
        return _functional_group_search(query, candidates_smiles, top_k)
    return []


def _detect_search_type(query: str) -> str:
    """自动检测查询类型。"""
    from rdkit import Chem

    # 检查是否是官能团名称
    q_lower = query.lower().strip()
    for fg_name, (smarts, keywords) in FUNCTIONAL_GROUPS.items():
        if q_lower in keywords or q_lower == fg_name:
            return "functional_group"

    # 检查是否是有效 SMILES
    mol = Chem.MolFromSmiles(query)
    if mol is not None:
        return "similarity"

    # 检查是否是 SMARTS
    patt = Chem.MolFromSmarts(query)
    if patt is not None:
        return "substructure"

    # 默认尝试相似性搜索
    return "similarity"


def _collect_smiles(store: LocalStore, elementkg_client: Any | None) -> list[dict]:
    """从本地 evidence 和 ElementKG 收集 SMILES。"""
    results = []  # [{"smiles": ..., "source_id": ..., "doc_id": ...}]

    # 本地 evidence
    try:
        doc_ids = [p.stem.replace(".molecules", "") for p in store.evidence_dir.glob("*.molecules.json")]
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
                r = session.run("MATCH (m:Molecule) WHERE m.smiles IS NOT NULL RETURN m.id, m.smiles, m.iupac_name LIMIT 5000")
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


def _similarity_search(query: str, candidates: list[dict], top_k: int) -> list[CandidateEvidence]:
    """结构相似性搜索（Tanimoto + Morgan fingerprint）。"""
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem

    query_mol = Chem.MolFromSmiles(query)
    if query_mol is None:
        return []

    query_fp = AllChem.GetMorganFingerprintAsBitVect(query_mol, 2)

    scored = []
    for c in candidates:
        mol = Chem.MolFromSmiles(c["smiles"])
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2)
        sim = DataStructs.TanimotoSimilarity(query_fp, fp)
        if sim > 0.1:  # 最低阈值
            scored.append((sim, c))

    scored.sort(key=lambda x: x[0], reverse=True)

    results = []
    for sim, c in scored[:top_k]:
        results.append(_to_candidate(c, confidence=sim, search_type="similarity"))
    return results


def _substructure_search(query: str, candidates: list[dict], top_k: int) -> list[CandidateEvidence]:
    """子结构匹配搜索。"""
    from rdkit import Chem

    pattern = Chem.MolFromSmarts(query)
    if pattern is None:
        # 尝试从 SMILES 转换
        mol = Chem.MolFromSmiles(query)
        if mol is not None:
            pattern = Chem.MolFromSmarts(Chem.MolToSmiles(mol))
    if pattern is None:
        return []

    results = []
    for c in candidates:
        mol = Chem.MolFromSmiles(c["smiles"])
        if mol is None:
            continue
        if mol.HasSubstructMatch(pattern):
            results.append(_to_candidate(c, confidence=0.9, search_type="substructure"))
            if len(results) >= top_k:
                break
    return results


def _functional_group_search(query: str, candidates: list[dict], top_k: int) -> list[CandidateEvidence]:
    """官能团匹配搜索。"""
    from rdkit import Chem

    q_lower = query.lower().strip()
    smarts = None
    for fg_name, (pattern, keywords) in FUNCTIONAL_GROUPS.items():
        if q_lower in keywords or q_lower == fg_name:
            smarts = pattern
            break

    if smarts is None:
        return []

    pattern = Chem.MolFromSmarts(smarts)
    if pattern is None:
        return []

    results = []
    for c in candidates:
        mol = Chem.MolFromSmiles(c["smiles"])
        if mol is None:
            continue
        if mol.HasSubstructMatch(pattern):
            results.append(_to_candidate(c, confidence=0.85, search_type="functional_group"))
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
```

- [ ] **Step 2: 验证语法**

Run: `python -c "import py_compile; py_compile.compile('src/retrieval/structure.py', doraise=True)"`
Expected: 无输出（语法正确）

- [ ] **Step 3: Commit**

```bash
git add src/retrieval/structure.py
git commit -m "feat: add structure retrieval module (substructure, similarity, functional group)"
```

---

### Task 2: 集成到 Router

**Files:**
- Modify: `src/retrieval/router.py`
- Modify: `src/retrieval/__init__.py`

- [ ] **Step 1: 修改 router.py**

在 `retrieve()` 方法中，dense retrieval 之后添加：

```python
# 结构检索（可选，需要 RDKit）
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
```

- [ ] **Step 2: 更新 __init__.py**

添加导出：
```python
from .structure import structure_search
```

- [ ] **Step 3: 运行测试**

Run: `pytest tests/test_retrieval.py tests/test_solver.py -v -k "not roma_adapter"`
Expected: 全部通过

- [ ] **Step 4: Commit**

```bash
git add src/retrieval/router.py src/retrieval/__init__.py
git commit -m "feat: integrate structure retrieval into router"
```

---

### Task 3: 端到端验证

- [ ] **Step 1: 测试相似性搜索**

```powershell
python -c "
import sys; sys.path.insert(0, 'src')
from retrieval.structure import structure_search
from storage import LocalStore
store = LocalStore('.')
results = structure_search('CCO', store, search_type='similarity', top_k=5)
print(f'Found {len(results)} results')
for r in results:
    print(f'  {r.evidence_id}: {r.summary[:60]} (conf={r.confidence:.2f})')
"
```
Expected: 返回按相似度排序的分子列表

- [ ] **Step 2: 测试子结构搜索**

```powershell
python -c "
import sys; sys.path.insert(0, 'src')
from retrieval.structure import structure_search
from storage import LocalStore
store = LocalStore('.')
results = structure_search('c1ccccc1', store, search_type='substructure', top_k=5)
print(f'Found {len(results)} results')
for r in results:
    print(f'  {r.evidence_id}: {r.summary[:60]}')
"
```
Expected: 返回含苯环的分子

- [ ] **Step 3: 测试官能团搜索**

```powershell
python -c "
import sys; sys.path.insert(0, 'src')
from retrieval.structure import structure_search
from storage import LocalStore
store = LocalStore('.')
results = structure_search('hydroxyl', store, search_type='functional_group', top_k=5)
print(f'Found {len(results)} results')
for r in results:
    print(f'  {r.evidence_id}: {r.summary[:60]}')
"
```
Expected: 返回含羟基的分子

- [ ] **Step 4: Commit 最终状态**

```bash
git add -A
git commit -m "feat: structure retrieval complete — substructure, similarity, functional group"
git push origin master
```
