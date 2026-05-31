# ElementKG2.0 集成设计文档

> **版本**: 1.1
> **日期**: 2026-05-30
> **状态**: 待审批

---

## 1. 目标

将 ElementKG2.0（10M 三元组）集成到 ChemEvoRAG 的检索流程中，实现：

- 化学实体的别名扩展（DCHA → dicyclohexylamine）
- 分子身份解析（name → SMILES/InChIKey/canonical_name）
- 查询改写（Query Rewrite）— 用扩展后的名称集合做检索，提高召回率
- 置信度评估 — 为候选别名设置置信度，避免过度扩展

对应架构文档：§4.1 ElementKG2.0 Core Graph、§4.3 Alignment/Index Layer、§5.2 Entity Resolution & Query Rewrite。

## 2. 技术选型

| 组件 | 选择 | 理由 |
|---|---|---|
| 图数据库 | Neo4j 5 Community | 项目已有 `neo4j_identity_client.py` 接口代码，Cypher 查询强大 |
| 部署方式 | Docker 本地部署 | 不依赖外部服务 |
| 数据来源 | `data/kg/10m_elementkg_release.csv`（858MB） | 千万级三元组 |
| 现有代码复用 | `src/normalization/neo4j_identity_client.py` | 已有 `Neo4jIdentityClient`，扩展而非重写 |

## 3. Neo4j 部署

### 3.1 Docker 命令

```bash
docker run -d --name chemevorag \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=chemevorag/chemevorag \
  -v D:\Desktop\evo\ChemEvoRAG_Phase1-main\data\neo4j:/data \
  neo4j:5-community
```

### 3.2 配置

- 容器名: `chemevorag`
- 用户名: `chemevorag`
- 密码: `chemevorag`
- Web 界面: `http://localhost:7474`
- Bolt 协议: `bolt://localhost:7687`
- 数据目录: `data/neo4j/`

### 3.3 .env 配置

```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=chemevorag
NEO4J_PASSWORD=chemevorag
```

### 3.4 .gitignore

```
data/neo4j/
```

## 4. 数据导入

### 4.1 CSV 格式

```csv
head_type,head_value,relation,tail_type,tail_value
Apparatus,apparatus_1,NAME_IS,literal,Reflux apparatus
Compound,compound_123,HAS_ALIAS,literal,DCHA
Compound,compound_123,HAS_SMILES,literal,C1CCC(CC1)NC2CCCCC2
```

### 4.2 导入脚本

新建 `scripts/import_elementkg.py`：

**功能**：
1. 分批读取 CSV（每批 10000 行）
2. 创建节点：`head_value` → `Node(type=head_type, id=head_value)`
3. 创建关系：`head -[relation]-> tail`（literal 值作为节点属性而非独立节点）
4. 创建索引：按 node type 和 id 建立索引

**节点类型**（从 CSV 的 head_type 推断）：
- `Compound` — 化合物
- `Reaction` — 反应
- `Apparatus` — 实验器材
- `Element` — 元素

**literal 处理**：当 `tail_type=literal` 时，不创建独立节点，而是将 `tail_value` 作为关系的属性存储。

**关系类型**（从 CSV 的 relation 推断）：
- `HAS_ALIAS` — 别名
- `HAS_SMILES` — SMILES
- `HAS_INCHIKEY` — InChIKey
- `HAS_IUPAC_NAME` — IUPAC 名称
- `NAME_IS` — 名称
- 其他 — 按原样创建

**运行命令**：
```bash
python scripts/import_elementkg.py
```

### 4.3 预估耗时

- CSV 解析: ~2-3 分钟
- Neo4j 导入: ~10-20 分钟（取决于磁盘速度）
- 索引创建: ~1-2 分钟
- 总计: ~15-25 分钟

## 5. Entity Resolution 服务

### 5.1 复用现有代码

扩展 `src/normalization/neo4j_identity_client.py` 中的 `Neo4jIdentityClient`，而不是新建类。

现有接口已包含：
- `find_by_name(name)` — 按名称查询
- `find_by_alias(alias)` — 按别名查询
- `find_by_smiles(smiles)` — 按 SMILES 查询
- `find_by_inchikey(inchikey)` — 按 InChIKey 查询
- `resolve_molecule(molecule)` — 解析分子身份

### 5.2 配置适配

`Neo4jIdentityConfig` 从 `config/neo4j.yaml` 读取配置：

```yaml
neo4j:
  uri: bolt://localhost:7687
  username: chemevorag
  password: chemevorag
  molecule_label: Compound
  id_field: id
  canonical_name_field: name
  name_fields: [name]
  alias_fields: [aliases]
  smiles_field: smiles
  inchikey_field: inchikey
```

### 5.3 查询流程

```
用户问 "DCHA 是什么"
  → entity_search 找到 MoleculeCard(names=["DCHA"])
  → Neo4jIdentityClient.find_by_alias("DCHA")
  → Cypher: MATCH (m:Compound) WHERE "DCHA" IN m.aliases RETURN m
  → 返回: MoleculeIdentityCandidate(canonical_name="dicyclohexylamine", smiles="...", inchikey="...")
  → resolve_molecule() 填充 MoleculeCard 的 aliases + raw_smiles + inchi_key
```

## 6. 查询改写（Query Rewrite）

### 6.1 目标

架构 §5.2 要求：将用户问题中的实体扩展为多种候选形式，用扩展后的名称集合做检索，提高召回率。

### 6.2 流程

```
用户问 "DCHA 的合成条件是什么"
  → LLM intent: reaction_condition_query, entities: ["DCHA"]
  → ElementKG 扩展: "DCHA" → ["DCHA", "dicyclohexylamine", "N-cyclohexylcyclohexanamine"]
  → 用扩展后的名称集合做 entity_search + lexical_search
  → 召回率提升
```

### 6.3 实现位置

在 `router.py` 的 `retrieve()` 中，entity_search 之前增加查询改写步骤：

```python
def retrieve(self, query, ...):
    ...
    # Query Rewrite: 扩展实体名称
    expanded_terms = self._expand_entities(query)

    # 用扩展后的名称做检索
    candidates = []
    for term in expanded_terms:
        candidates.extend(entity_search(term, self.store, ...))
    candidates.extend(lexical_search(query, self.store, ...))
    ...
```

## 7. 置信度评估

### 7.1 目标

架构 §5.2 要求：为候选别名设置置信度，避免过度扩展导致噪声引入。

### 7.2 策略

- **精确匹配**（SMILES/InChIKey 完全一致）: confidence = 1.0
- **名称匹配**（alias 完全一致）: confidence = 0.9
- **模糊匹配**（名称包含/前缀匹配）: confidence = 0.7
- **跨类型匹配**（通过别名链间接关联）: confidence = 0.5

在 `resolve_molecule()` 返回的 `MoleculeIdentityCandidate` 中已有 `score` 字段，直接使用。

## 8. 与检索流程集成

### 8.1 修改文件

- `src/retrieval/router.py` — 增加 Query Rewrite 和 Entity Resolution 步骤

### 8.2 完整流程

```python
def retrieve(self, query, ...):
    resolved_intent = ...

    # Step 1: Query Rewrite — 扩展实体名称
    expanded_terms = self._expand_entities(query)

    # Step 2: 全通道检索
    candidates = []
    for term in expanded_terms:
        candidates.extend(entity_search(term, self.store, ...))
    candidates.extend(lexical_search(query, self.store, ...))
    candidates.extend(reaction_event_search(query, self.store, ...))
    try:
        from .dense import dense_search
        candidates.extend(dense_search(query, ...))
    except Exception:
        pass

    # Step 3: Entity Resolution — 用 ElementKG 填充分子信息
    if self.elementkg_client:
        candidates = self._resolve_entities(candidates)

    # Step 4: Intent reranking
    candidates = _intent_rerank(candidates, resolved_intent, query)
    candidates = _dedupe_candidates(candidates)[:limit]
    ...
```

## 9. 文件变更清单

| 文件 | 操作 | 说明 |
|---|---|---|
| `scripts/import_elementkg.py` | 新建 | CSV → Neo4j 导入脚本 |
| `config/neo4j.yaml` | 新建 | Neo4j 连接配置（ElementKG 的 label/field 映射） |
| `.env` | 修改 | 添加 NEO4J_URI/USER/PASSWORD |
| `.gitignore` | 修改 | 添加 data/neo4j/ |
| `src/normalization/neo4j_identity_client.py` | 可能修改 | 适配 ElementKG 的字段映射（如需要） |
| `src/retrieval/router.py` | 修改 | 集成 Query Rewrite + Entity Resolution |
| `tests/test_elementkg_integration.py` | 新建 | 测试 |

## 10. 验证标准

1. `docker ps` 显示 `chemevorag` 容器运行中
2. `http://localhost:7474` 可访问 Neo4j Web 界面
3. `python scripts/import_elementkg.py` 成功导入数据
4. Neo4j 中可查询到化合物节点和别名关系
5. `Neo4jIdentityClient.find_by_alias("DCHA")` 返回正确的候选
6. `python scripts/run_query.py "What is DCHA?" --doc-id 1` 返回正确的别名信息
7. 评测分数提升（特别是 alias_resolution 和 entity_lookup）

## 11. 与架构文档的对应关系

| 架构章节 | 设计覆盖 |
|---|---|
| §2.1 ElementKG2.0 Core | ✅ read-only backbone，提供 alias/SMILES/InChIKey |
| §4.1 ElementKG2.0 Core Graph | ✅ Neo4j 部署，read-only |
| §4.3 Alignment/Index Layer | ✅ Entity Mapping（resolve_molecule） |
| §5.2 Entity Resolution | ✅ find_by_name/alias/smiles/inchikey |
| §5.2 Query Rewrite | ✅ _expand_entities() 扩展名称集合 |
| §5.2 置信度评估 | ✅ 基于匹配类型的 confidence 分级 |
| §5.6 Layered Fallback | ⚠️ 后续实现（ElementKG → Paper Evidence → Raw Corpus） |
