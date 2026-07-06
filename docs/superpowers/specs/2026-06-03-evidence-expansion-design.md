# Graph-based Evidence Expansion 设计文档

> **版本**: 1.0
> **日期**: 2026-06-03
> **状态**: 待审批

---

## 1. 目标

实现 §5.7 Graph-based Evidence Expansion — 基于证据内链扩展相关证据，提升多信息整合能力。

## 2. 设计

### 2.1 执行位置

`router.py` 的 `retrieve()` 中，检索后、排序前。

### 2.2 新建文件

`src/retrieval/expansion.py`

### 2.3 接口

```python
def expand_evidence(
    candidates: list[CandidateEvidence],
    store: LocalStore,
    top_n: int = 5,
) -> list[CandidateEvidence]:
    """基于内链扩展证据，返回扩展后的候选列表。"""
```

### 2.4 扩展策略

| 候选类型 | 内链字段 | 扩展目标 |
|---|---|---|
| molecule | source_mentions → block_id | 找到提及该分子的 DocumentBlock |
| reaction_event | supporting_block_ids | 找到支持该反应的 DocumentBlock |
| reaction_event | reactants/products.name | 找到对应的 MoleculeCard |
| document_block | prev_block_id / next_block_id | 相邻上下文的 block |
| document_block | section | 同一节的相邻 block |

## 3. 与 Router 集成

```python
# retrieve() 中，rerank 和 dedup 之后
candidates = _expand_evidence(candidates, self.store)
candidates = candidates[:limit]
```

## 4. 文件变更

| 文件 | 操作 | 说明 |
|---|---|---|
| `src/retrieval/expansion.py` | 新建 | 证据扩展模块 |
| `src/retrieval/router.py` | 修改 | 集成扩展调用 |
| `src/retrieval/__init__.py` | 修改 | 导出 |
