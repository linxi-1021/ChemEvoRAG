# Structure Retrieval 通道设计文档

> **版本**: 1.0
> **日期**: 2026-05-30
> **状态**: 待审批

---

## 1. 目标

实现结构检索通道（§5.5），支持子结构匹配、结构相似性、官能团匹配查询。

对应架构文档：§5.5 Multi-channel Retrieval — Structure Retrieval。

## 2. 支持的查询类型

| 查询类型 | RDKit 方法 | 示例查询 | 示例结果 |
|---|---|---|---|
| 子结构匹配 | `mol.HasSubstructPattern(query)` | "含有苯环的分子" | 所有含苯环的 MoleculeCard |
| 结构相似性 | Tanimoto + Morgan fingerprint | "与 ethanol 相似的分子" | 按相似度排序的分子列表 |
| 官能团匹配 | SMARTS 模式匹配 | "含有羟基的分子" | 所有含 -OH 的 MoleculeCard |

## 3. 数据源

同时搜索两个数据源：

1. **本地 evidence**：当前论文的 MoleculeCard（有 raw_smiles 的）
2. **ElementKG Neo4j**：34,057 个 Molecule 节点的 SMILES

合并结果，按相似度/匹配度排序。

## 4. 新建文件

`src/retrieval/structure.py`

### 4.1 接口设计

```python
def structure_search(
    query: str,
    store: LocalStore,
    *,
    elementkg_client=None,
    search_type: str = "similarity",
    top_k: int = 10,
) -> list[CandidateEvidence]:
    """结构检索入口。

    Args:
        query: SMILES 字符串或分子名称
        store: 本地 evidence 存储
        elementkg_client: ElementKG 客户端（可选）
        search_type: "substructure" | "similarity" | "functional_group"
        top_k: 返回数量
    """
```

### 4.2 查询类型检测

自动检测查询类型：
- 如果 query 是有效 SMILES → similarity search
- 如果 query 包含 SMARTS 模式 → substructure search
- 如果 query 是官能团名称（"hydroxyl", "carboxyl", "amine"）→ functional_group search

### 4.3 相似度计算

使用 RDKit Morgan fingerprint + Tanimoto 相似度：

```python
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

mol1 = Chem.MolFromSmiles("CCO")  # ethanol
mol2 = Chem.MolFromSmiles("CCCO")  # propanol

fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2)
fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2)
similarity = DataStructs.TanimotoSimilarity(fp1, fp2)  # 0.67
```

### 4.4 官能团 SMARTS 模式

| 官能团 | SMARTS | 名称关键词 |
|---|---|---|
| 羟基 | `[OX2H]` | hydroxyl, alcohol, OH |
| 羰基 | `[CX3]=[OX1]` | carbonyl, C=O |
| 羧基 | `[CX3](=O)[OX2H]` | carboxyl, COOH |
| 氨基 | `[NX3;H2]` | amine, NH2 |
| 酯基 | `[CX3](=O)[OX2]` | ester, COO |
| 苯环 | `c1ccccc1` | phenyl, benzene, aromatic |
| 卤素 | `[F,Cl,Br,I]` | halogen, fluoro, chloro |

## 5. 与 Router 集成

修改 `src/retrieval/router.py`：

```python
# 在 retrieve() 方法中，dense retrieval 之后
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
    pass  # RDKit 未安装时静默跳过
```

## 6. 文件变更清单

| 文件 | 操作 | 说明 |
|---|---|---|
| `src/retrieval/structure.py` | 新建 | 结构检索模块 |
| `src/retrieval/router.py` | 修改 | 添加 structure_search 调用 |
| `src/retrieval/__init__.py` | 修改 | 导出 structure_search |
| `tests/test_structure.py` | 新建 | 测试 |

## 7. 验证标准

1. `structure_search("CCO", store, search_type="similarity")` 返回按相似度排序的分子
2. `structure_search("c1ccccc1", store, search_type="substructure")` 返回含苯环的分子
3. `structure_search("hydroxyl", store, search_type="functional_group")` 返回含羟基的分子
4. 结合 ElementKG 搜索覆盖更广
5. RDKit 未安装时静默跳过，不影响其他检索通道

## 8. 与架构文档的对应关系

| 架构章节 | 设计覆盖 |
|---|---|
| §5.5 Structure Retrieval | ✅ 子结构、相似性、官能团匹配 |
| §5.5 Multi-channel Retrieval | ✅ 作为第 5 个检索通道 |
| §4.3 Alignment/Index Layer | ✅ 结构索引（RDKit fingerprint） |
