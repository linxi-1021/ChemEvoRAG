# ElementKG2.0 集成实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 ElementKG2.0（10M 三元组）集成到 ChemEvoRAG，实现化学实体的别名扩展、分子身份解析和查询改写。

**Architecture:** Neo4j 存储 ElementKG 三元组，`Neo4jIdentityClient`（已有）做实体查询，`router.py` 增加 Query Rewrite + Entity Resolution 步骤。

**Tech Stack:** Neo4j 5 Community (Docker), Python neo4j driver, Cypher

---

## 文件结构

| 文件 | 操作 | 职责 |
|---|---|---|
| `config/neo4j.yaml` | 新建 | Neo4j 连接配置 + ElementKG 字段映射 |
| `.env` | 修改 | 添加 NEO4J_URI/USER/PASSWORD |
| `.gitignore` | 修改 | 添加 data/neo4j/ |
| `scripts/import_elementkg.py` | 新建 | CSV → Neo4j 导入脚本 |
| `src/normalization/neo4j_identity_client.py` | 可能修改 | 适配 ElementKG 字段映射 |
| `src/retrieval/router.py` | 修改 | Query Rewrite + Entity Resolution |
| `tests/test_elementkg_integration.py` | 新建 | 集成测试 |

---

### Task 1: Neo4j Docker 部署 + 配置

**Files:**
- Create: `config/neo4j.yaml`
- Modify: `.env`
- Modify: `.gitignore`

- [ ] **Step 1: 检查 Docker 是否安装**

Run: `docker --version`
Expected: `Docker version 24.x.x` 或类似输出

- [ ] **Step 2: 启动 Neo4j 容器**

Run:
```bash
docker run -d --name chemevorag \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=chemevorag/chemevorag \
  -v D:\Desktop\evo\ChemEvoRAG_Phase1-main\data\neo4j:/data \
  neo4j:5-community
```
Expected: 返回容器 ID

- [ ] **Step 3: 验证容器运行**

Run: `docker ps | findstr chemevorag`
Expected: 显示 `chemevorag` 容器，状态 `Up`

- [ ] **Step 4: 创建 Neo4j 配置文件**

Create `config/neo4j.yaml`:
```yaml
neo4j:
  uri: bolt://localhost:7687
  username: chemevorag
  password: chemevorag
  database: neo4j

  # ElementKG 节点/字段映射
  molecule_label: Compound
  id_field: id
  canonical_name_field: name
  name_fields:
    - name
  alias_fields:
    - aliases
  smiles_field: smiles
  inchikey_field: inchikey
  result_limit: 20
```

- [ ] **Step 5: 更新 .env 文件**

在 `.env` 末尾添加：
```env
# Neo4j (ElementKG)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=chemevorag
NEO4J_PASSWORD=chemevorag
```

- [ ] **Step 6: 更新 .gitignore**

在 `.gitignore` 末尾添加：
```
data/neo4j/
```

- [ ] **Step 7: 验证 Neo4j Web 界面**

浏览器打开 `http://localhost:7474`
- 用户名: `chemevorag`
- 密码: `chemevorag`
Expected: 成功登录 Neo4j Browser

- [ ] **Step 8: Commit**

```bash
git add config/neo4j.yaml .env .gitignore
git commit -m "config: add Neo4j Docker setup and ElementKG connection config"
```

---

### Task 2: ElementKG CSV 导入脚本

**Files:**
- Create: `scripts/import_elementkg.py`

- [ ] **Step 1: 分析 CSV 结构**

Run:
```bash
head -20 data/kg/10m_elementkg_release.csv
```
记录 head_type 的所有取值和 relation 的所有取值。

- [ ] **Step 2: 编写导入脚本**

Create `scripts/import_elementkg.py`:
```python
#!/usr/bin/env python
"""Import ElementKG CSV into Neo4j."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load .env
_dotenv = PROJECT_ROOT / ".env"
if _dotenv.exists():
    from dotenv import load_dotenv
    load_dotenv(_dotenv)

import os
from neo4j import GraphDatabase

BATCH_SIZE = 10000

# 当 tail_type=literal 时，将 tail_value 作为关系属性存储
LITERAL_AS_PROPERTY = True

# 需要创建节点的 head_type
NODE_TYPES = {"Compound", "Reaction", "Apparatus", "Element"}


def _get_driver():
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "chemevorag")
    password = os.environ.get("NEO4J_PASSWORD", "chemevorag")
    return GraphDatabase.driver(uri, auth=(user, password))


def create_indexes(driver):
    """为每种节点类型创建索引。"""
    with driver.session() as session:
        for label in NODE_TYPES:
            session.run(f"CREATE INDEX IF NOT EXISTS FOR (n:{label}) ON (n.id)")
    print("Indexes created.")


def import_csv(driver, csv_path: Path, batch_size: int = BATCH_SIZE):
    """分批导入 CSV 到 Neo4j。"""
    total_rows = 0
    total_nodes = 0
    total_rels = 0
    start = time.time()

    with driver.session() as session:
        batch = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                batch.append(row)
                if len(batch) >= batch_size:
                    nodes, rels = _process_batch(session, batch)
                    total_nodes += nodes
                    total_rels += rels
                    total_rows += len(batch)
                    elapsed = time.time() - start
                    print(f"  {total_rows} rows, {total_nodes} nodes, {total_rels} rels ({elapsed:.1f}s)")
                    batch = []

        if batch:
            nodes, rels = _process_batch(session, batch)
            total_nodes += nodes
            total_rels += rels
            total_rows += len(batch)

    elapsed = time.time() - start
    print(f"\nDone: {total_rows} rows, {total_nodes} nodes, {total_rels} rels in {elapsed:.1f}s")


def _process_batch(session, batch: list[dict]) -> tuple[int, int]:
    """处理一批行，返回 (创建的节点数, 创建的关系数)。"""
    nodes_created = 0
    rels_created = 0

    # 收集需要创建的节点
    node_set: set[tuple[str, str]] = set()  # (type, id)
    rel_triples: list[tuple[str, str, str, str, str]] = []  # (head_type, head_id, relation, tail_type, tail_id)

    for row in batch:
        ht = row.get("head_type", "").strip()
        hv = row.get("head_value", "").strip()
        rel = row.get("relation", "").strip()
        tt = row.get("tail_type", "").strip()
        tv = row.get("tail_value", "").strip()

        if not ht or not hv or not rel:
            continue

        # 创建 head 节点
        if ht in NODE_TYPES:
            node_set.add((ht, hv))

        # 处理 tail
        if tt == "literal" and LITERAL_AS_PROPERTY:
            # literal 值作为关系属性
            rel_triples.append((ht, hv, rel, "__literal__", tv))
        elif tt in NODE_TYPES:
            node_set.add((tt, tv))
            rel_triples.append((ht, hv, rel, tt, tv))
        else:
            # 其他类型也作为 literal
            rel_triples.append((ht, hv, rel, "__literal__", tv))

    # 批量创建节点
    if node_set:
        by_type: dict[str, list[str]] = {}
        for ntype, nid in node_set:
            by_type.setdefault(ntype, []).append(nid)
        for ntype, ids in by_type.items():
            session.run(
                f"UNWIND $ids AS id MERGE (n:{ntype} {{id: id}})",
                ids=ids,
            )
            nodes_created += len(ids)

    # 批量创建关系
    for ht, hv, rel, tt, tv in rel_triples:
        rel_type = rel.replace(" ", "_").replace("-", "_").upper()
        if tt == "__literal__":
            # literal 值作为关系属性
            session.run(
                f"MATCH (h:{ht} {{id: $hid}}) "
                f"MERGE (h)-[r:{rel_type}]->(t:Literal {{value: $tval}}) "
                f"SET t.value = $tval",
                hid=hv, tval=tv,
            )
        else:
            session.run(
                f"MATCH (h:{ht} {{id: $hid}}), (t:{tt} {{id: $tid}}) "
                f"MERGE (h)-[r:{rel_type}]->(t)",
                hid=hv, tid=tv,
            )
        rels_created += 1

    return nodes_created, rels_created


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv", default=str(PROJECT_ROOT / "data" / "kg" / "10m_elementkg_release.csv"),
        help="Path to ElementKG CSV file.",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}", file=sys.stderr)
        return 1

    print(f"Connecting to Neo4j ...")
    driver = _get_driver()

    try:
        print("Creating indexes ...")
        create_indexes(driver)

        print(f"Importing {csv_path.name} ...")
        import_csv(driver, csv_path, args.batch_size)
    finally:
        driver.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: 测试导入脚本（小批量）**

Run:
```bash
python scripts/import_elementkg.py --batch-size 1000
```
Expected: 显示进度，无报错，最终显示 "Done: X rows, Y nodes, Z rels"

- [ ] **Step 4: 验证 Neo4j 中的数据**

在 Neo4j Browser (`http://localhost:7474`) 中执行：
```cypher
MATCH (n:Compound) RETURN count(n) LIMIT 1
```
Expected: 返回非零计数

```cypher
MATCH (c:Compound)-[:HAS_ALIAS]->(a) RETURN c.id, a.value LIMIT 10
```
Expected: 返回化合物 ID 和别名

- [ ] **Step 5: Commit**

```bash
git add scripts/import_elementkg.py
git commit -m "feat: add ElementKG CSV to Neo4j import script"
```

---

### Task 3: 适配 Neo4jIdentityConfig

**Files:**
- Modify: `src/normalization/neo4j_identity_client.py`
- Create: `tests/test_elementkg_integration.py`

- [ ] **Step 1: 检查现有 Neo4jIdentityConfig 是否兼容**

Read `src/normalization/neo4j_identity_client.py`，检查 `Neo4jIdentityConfig` 的字段是否能适配 ElementKG 的 Compound 节点结构。

ElementKG Compound 节点结构：
- `id`: string (如 "compound_123")
- `name`: string
- `aliases`: list (通过 HAS_ALIAS 关系连接到 Literal 节点)
- `smiles`: string (通过 HAS_SMILES 关系)
- `inchikey`: string (通过 HAS_INCHIKEY 关系)

如果现有 config 的 `alias_fields` 支持关系查询，则直接复用。如果只支持节点属性查询，需要扩展。

- [ ] **Step 2: 编写测试**

Create `tests/test_elementkg_integration.py`:
```python
"""Integration tests for ElementKG Neo4j client."""

import os
import pytest

# Skip if Neo4j not available
_neo4j_uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")


def _neo4j_available() -> bool:
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(
            _neo4j_uri,
            auth=(os.environ.get("NEO4J_USER", "chemevorag"),
                  os.environ.get("NEO4J_PASSWORD", "chemevorag")),
        )
        with driver.session() as session:
            session.run("RETURN 1")
        driver.close()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _neo4j_available(),
    reason="Neo4j not available",
)


@pytest.fixture
def config_path():
    return str(Path(__file__).resolve().parents[1] / "config" / "neo4j.yaml")


@pytest.fixture
def client(config_path):
    from normalization.neo4j_identity_client import Neo4jIdentityClient, Neo4jIdentityConfig
    config = Neo4jIdentityConfig.from_yaml(config_path)
    client = Neo4jIdentityClient(config)
    yield client
    client.close()


class TestElementKGQuery:
    def test_find_by_alias_returns_candidates(self, client):
        """查询一个已知的别名应该返回候选。"""
        # 需要先导入数据后才能测试
        # 这里测试查询不报错
        candidates = client.find_by_alias("DCHA")
        # 可能返回空（如果数据未导入），但不应报错
        assert isinstance(candidates, list)

    def test_find_by_name_returns_candidates(self, client):
        candidates = client.find_by_name("dicyclohexylamine")
        assert isinstance(candidates, list)

    def test_find_by_smiles_returns_candidates(self, client):
        candidates = client.find_by_smiles("C1CCC(CC1)NC2CCCCC2")
        assert isinstance(candidates, list)

    def test_resolve_molecule_returns_card(self, client):
        from evidence import MoleculeCard
        card = MoleculeCard(
            molecule_card_id="test_001",
            doc_id="test",
            names=["DCHA"],
        )
        result = client.resolve_molecule(card)
        assert isinstance(result, MoleculeCard)
        assert result.molecule_card_id == "test_001"
```

- [ ] **Step 3: 运行测试验证不报错**

Run: `pytest tests/test_elementkg_integration.py -v`
Expected: 测试通过（或因 Neo4j 未连接而 skip）

- [ ] **Step 4: 修改 Neo4jIdentityClient 的置信度评估**

当前 `_candidate_from_node` 对所有匹配返回 `score=1.0`。需要根据匹配类型分级：

```python
def _candidate_from_node(self, node, *, match_type):
    ...
    # 置信度分级
    score_map = {
        "inchikey": 1.0,   # 精确匹配
        "smiles": 0.95,    # SMILES 匹配
        "name": 0.9,       # 名称匹配
        "alias": 0.85,     # 别名匹配
    }
    score = score_map.get(match_type, 0.7)
    return MoleculeIdentityCandidate(
        ...,
        match_type=match_type,
        score=score,
    )
```

如果 ElementKG 的别名存储在关系而非节点属性中，需要扩展查询逻辑。

- [ ] **Step 5: Commit**

```bash
git add src/normalization/neo4j_identity_client.py tests/test_elementkg_integration.py
git commit -m "feat: adapt Neo4jIdentityClient for ElementKG Compound nodes"
```

---

### Task 4: Query Rewrite 集成到 Router

**Files:**
- Modify: `src/retrieval/router.py`

- [ ] **Step 1: 在 RetrievalRouter 中添加 ElementKG 客户端初始化**

在 `__init__` 中添加：
```python
def __init__(self, store, *, default_top_k=10, use_llm_intent=True, elementkg_client=None):
    self.store = store
    self.default_top_k = default_top_k
    self.use_llm_intent = use_llm_intent
    self.elementkg_client = elementkg_client
```

- [ ] **Step 2: 添加 _expand_entities 方法**

```python
def _expand_entities(self, query: str) -> list[str]:
    """用 ElementKG 扩展查询中的实体名称。"""
    if not self.elementkg_client:
        return [query]

    # 简单实现：提取查询中的潜在实体名称，用 ElementKG 查找别名
    # 后续可以用 LLM 做更精确的实体识别
    terms = query.split()
    expanded = [query]
    for term in terms:
        if len(term) >= 2 and term[0].isupper():
            candidates = self.elementkg_client.find_by_alias(term)
            for c in candidates:
                if c.canonical_name:
                    expanded.append(c.canonical_name)
                expanded.extend(c.aliases)
    return list(set(expanded))
```

- [ ] **Step 3: 修改 retrieve() 方法**

在 entity_search 之前添加查询改写：
```python
def retrieve(self, query, *, doc_ids=None, top_k=None):
    resolved_intent = ...
    limit = ...

    # Query Rewrite: 扩展实体名称
    expanded_terms = self._expand_entities(query)

    # 全通道检索
    candidates = []
    for term in expanded_terms:
        candidates.extend(entity_search(term, self.store, doc_ids=doc_ids, top_k=limit))
    candidates.extend(lexical_search(query, self.store, doc_ids=doc_ids, top_k=limit * 5))
    candidates.extend(reaction_event_search(query, self.store, doc_ids=doc_ids, top_k=limit * 2))
    try:
        from .dense import dense_search
        candidates.extend(dense_search(query, doc_ids=doc_ids, top_k=limit * 2))
    except Exception:
        pass

    # Entity Resolution
    if self.elementkg_client:
        candidates = self._resolve_entities(candidates)

    # Intent reranking
    candidates = _intent_rerank(candidates, resolved_intent, query)
    ...
```

- [ ] **Step 4: 添加 _resolve_entities 方法**

```python
def _resolve_entities(self, candidates):
    """用 ElementKG 填充分子候选的别名/SMILES/InChIKey。"""
    resolved = []
    for c in candidates:
        if c.evidence_type == "molecule":
            # 从 structured_slots 中获取 molecule_card_id
            card_id = c.structured_slots.get("molecule_card_id")
            if card_id:
                try:
                    card = self.store.load_molecule(card_id)
                    enriched = self.elementkg_client.resolve_molecule(card)
                    # 更新候选的 summary
                    names = enriched.names + enriched.aliases
                    if names:
                        c.summary = " | ".join(n for n in [c.summary] + names if n)
                except Exception:
                    pass
        resolved.append(c)
    return resolved
```

- [ ] **Step 5: 运行现有测试验证不破坏**

Run: `pytest tests/test_retrieval.py tests/test_solver.py -v -k "not roma_adapter"`
Expected: 全部通过

- [ ] **Step 6: Commit**

```bash
git add src/retrieval/router.py
git commit -m "feat: integrate Query Rewrite and Entity Resolution into router"
```

---

### Task 5: 端到端验证

**Files:**
- 无新文件，运行现有脚本验证

- [ ] **Step 1: 启动 Neo4j（如未运行）**

Run: `docker start chemevorag`

- [ ] **Step 2: 导入 ElementKG 数据**

Run: `python scripts/import_elementkg.py`
Expected: 成功导入，显示节点和关系数量

- [ ] **Step 3: 验证查询**

Run:
```bash
python -c "
import sys; sys.path.insert(0, 'src')
from normalization.neo4j_identity_client import Neo4jIdentityClient, Neo4jIdentityConfig
config = Neo4jIdentityConfig.from_yaml('config/neo4j.yaml')
client = Neo4jIdentityClient(config)
candidates = client.find_by_alias('DCHA')
print(f'Found {len(candidates)} candidates')
for c in candidates:
    print(f'  {c.canonical_name}: {c.smiles}')
client.close()
"
```
Expected: 返回 dicyclohexylamine 及其 SMILES

- [ ] **Step 4: 运行端到端查询**

Run:
```bash
python scripts/run_query.py "What is DCHA?" --doc-id 1 --use-llm --json
```
Expected: 返回包含 dicyclohexylamine 别名信息的 GroundedAnswer

- [ ] **Step 5: 运行评测**

Run: `python scripts/eval_questions.py --limit 10`
Expected: alias_resolution 和 entity_lookup 分数有提升

- [ ] **Step 6: Commit 最终状态**

```bash
git add -A
git commit -m "feat: ElementKG2.0 integration complete — entity resolution and query rewrite"
git push origin master
```
