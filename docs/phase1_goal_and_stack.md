# ChemEvoRAG 第一阶段任务目标与项目组合说明

## 1. 第一阶段定位

第一阶段目标是搭建 ChemEvoRAG 的最小可用版本，也就是 MVP。

这一阶段不追求完整实现说明书中的全部能力，尤其暂时不做千万级 ElementKG、不做完整 Skill Evolution 自动优化、不做大规模生产部署。第一阶段重点是证明核心链路可行：

```text
化学 PDF
  -> 化学文献解析
  -> 证据对象构建
  -> 化学实体标准化
  -> 多通道检索
  -> 带证据和来源的回答
```

成功标准是：系统能够针对少量化学论文或专利 PDF，回答反应条件、产率、底物/产物、化合物别名、实验步骤来源等问题，并返回结构化证据和原文位置。

## 2. 第一阶段核心目标

### 2.1 输入目标

支持以下输入：

```text
1. 单篇化学论文 PDF
2. 多篇化学论文 PDF
3. Supporting Information PDF
4. 简单专利 PDF，作为可选测试项
```

第一阶段优先处理英文文献，中文别名和中英文跨语言实体对齐放到后续阶段增强。

### 2.2 解析目标

从 PDF 中解析出：

```text
1. 文本段落
2. 标题、摘要、章节
3. 表格
4. 图注
5. 实验步骤 procedure
6. 分子图候选区域
7. 反应图候选区域
8. PDF 页码和 bbox 来源位置
```

如果图像化学结构识别失败，也要保留失败状态和原图来源，避免静默丢失证据。

### 2.3 证据对象目标

第一阶段实现三类核心证据对象：

```text
1. MoleculeCard
2. ReactionEventCard
3. FactCard
```

暂时不追求完整 Paper-level Evidence Graph 的全部边类型，但需要至少保留：

```text
paper -> contains -> block
block -> mentions -> molecule
reaction_event -> has_reactant -> molecule
reaction_event -> has_product -> molecule
reaction_event -> has_condition -> condition
reaction_event -> derived_from -> block/table/figure
```

### 2.4 检索目标

第一阶段实现五类检索：

```text
1. Lexical Retrieval：关键词、compound ID、产率、温度、时间
2. Dense Retrieval：自然语言语义检索
3. Entity Retrieval：分子名称、SMILES、InChIKey、别名
4. Reaction Event Retrieval：按反应物、产物、条件、产率检索
5. Provenance Backtracking：回到原文 block/page/bbox 查证
```

结构相似搜索、子结构搜索、官能团搜索可以先设计接口，后续再增强。

### 2.5 回答目标

回答必须包含：

```text
1. answer：直接答案
2. supporting_evidence：支撑证据
3. provenance：来源位置，包含 doc_id、page、block_id、bbox
4. involved_entities：涉及的化学实体
5. retrieval_path：检索路径
6. confidence：置信度
7. uncertainty：不确定性或失败原因
```

## 3. 推荐项目组合

### 3.1 总体组合

推荐组合如下：

```text
OpenChemIE / ChemEagle
  -> ChemEvo Evidence Builder
  -> RDKit normalization
  -> PostgreSQL + Neo4j + Qdrant/pgvector
  -> LightRAG / RAG-Anything
  -> ROMA Solver
```

### 3.2 各组件职责

| 模块 | 首选项目/技术 | 职责 |
|---|---|---|
| PDF 与化学信息解析 | OpenChemIE | 从 PDF、图片、文本中抽取 molecule/reaction 信息 |
| 多 Agent 化学抽取增强 | ChemEagle | 作为反应图和复杂 PDF 抽取的增强模块 |
| 分子标准化 | RDKit | canonical SMILES、InChIKey、结构校验 |
| 名称解析 | PubChem / OPSIN，可选 | IUPAC/name 到结构表示的补充解析 |
| 关系图存储 | Neo4j | Evidence Graph、实体关系、多跳查询 |
| 结构化证据存储 | PostgreSQL | MoleculeCard、ReactionEventCard、FactCard、provenance |
| 向量检索 | Qdrant 或 pgvector | 段落、证据摘要、图注、procedure 语义检索 |
| 图谱 RAG | LightRAG | 第一阶段优先使用，轻量、适合文本+图谱检索 |
| 多模态 RAG | RAG-Anything，可选 | 后续增强表格、图片、公式、图文混合检索 |
| Agent Solver | ROMA | Query planning、retrieval routing、tool calling、answer synthesis |

## 4. 为什么这样组合

ROMA 不适合作为底层 RAG 或化学解析项目，因为它本质上是递归 Agent 编排框架，强项是任务拆解、工具调用、执行追踪和 prompt 优化。

ChemEvoRAG 的重心在化学文献结构化：

```text
PDF layout
chemical figure parsing
molecule recognition
reaction extraction
chemical identity alignment
evidence graph
retrieval routing
```

因此更合理的做法是：

```text
让 OpenChemIE / ChemEagle 负责“看懂化学文献”
让 RDKit / ElementKG 负责“认准化学实体”
让 LightRAG / RAG-Anything 负责“检索和图谱增强”
让 ROMA 负责“调度、规划和自进化”
```

## 5. 第一阶段 MVP 架构

```text
data/raw_pdfs
  ↓
PDF Ingestion
  ↓
OpenChemIE Parser
  ↓
Raw Extraction JSON
  ↓
Evidence Builder
  ↓
MoleculeCard / ReactionEventCard / FactCard
  ↓
RDKit Normalizer
  ↓
PostgreSQL + Neo4j + Vector Store
  ↓
Retrieval Router
  ↓
LightRAG / Custom Retrieval Tools
  ↓
ROMA ChemRAG Solver
  ↓
Grounded Answer JSON
```

## 6. 第一阶段目录建议

当前项目目录建议按以下方式组织：

```text
ChemEvoRAG_Phase1/
  README.md
  docs/
    phase1_goal_and_stack.md
    schemas.md
    api_plan.md
  config/
    parser.yaml
    storage.yaml
    solver.yaml
  data/
    raw_pdfs/
    parsed/
    evidence/
    indexes/
  src/
    ingestion/
    parsing/
    evidence/
    normalization/
    retrieval/
    solver/
    storage/
  scripts/
    ingest_pdf.py
    build_evidence.py
    run_query.py
```

## 7. 第一阶段任务拆分

### Task 1：项目骨架与配置

产出：

```text
1. 项目目录结构
2. parser/storage/solver 配置文件
3. 基础数据目录
4. 统一日志与运行入口
```

### Task 2：PDF 解析适配层

产出：

```text
1. OpenChemIE wrapper
2. ChemEagle wrapper，可选
3. PDF -> raw_extraction.json
4. 失败记录和 provenance metadata
```

### Task 3：证据 schema

产出：

```text
1. MoleculeCard schema
2. ReactionEventCard schema
3. FactCard schema
4. SourceProvenance schema
5. EvidencePackage schema
```

### Task 4：Evidence Builder

产出：

```text
1. raw extraction -> MoleculeCard
2. raw extraction -> ReactionEventCard
3. paragraph/table/figure -> FactCard
4. evidence completeness score
```

### Task 5：化学实体标准化

产出：

```text
1. RDKit canonical SMILES
2. InChIKey generation
3. invalid molecule detection
4. alias table 初版
```

### Task 6：存储层

产出：

```text
1. PostgreSQL 表结构
2. Neo4j 节点和边结构
3. Qdrant 或 pgvector 向量索引
4. evidence/provenance 查询接口
```

### Task 7：检索层

产出：

```text
1. lexical search
2. dense search
3. entity search
4. reaction event search
5. provenance backtracking
```

### Task 8：ROMA Solver 接入

产出：

```text
1. query understanding prompt
2. retrieval routing prompt
3. retrieval tools
4. answer generation prompt
5. grounded answer JSON
```

### Task 9：端到端样例

产出：

```text
1. 3-5 篇测试 PDF
2. 10-20 个测试问题
3. 每个问题的答案、证据、来源、置信度
4. 失败样例记录
```

## 8. 第一阶段暂不做的内容

以下内容放到第二阶段或第三阶段：

```text
1. 千万级 ElementKG
2. 完整中文化学别名体系
3. 大规模专利批处理
4. 完整自进化 Skill Evolution
5. 自动 GEPA 批量优化
6. 子结构检索和结构相似检索的高性能实现
7. 复杂多文献 synthesis route planning
8. 生产级 Web UI
```

## 9. 第一阶段验收标准

第一阶段完成时，应能运行如下流程：

```text
1. 放入一个 chemistry PDF
2. 执行 ingest
3. 生成 parsed JSON
4. 生成 evidence cards
5. 写入数据库和索引
6. 提问：
   - 某个 compound 的合成条件是什么？
   - 哪个 reaction yield 最高？
   - DCHA 指的是什么？
   - 某个 product 来自哪张表或哪页？
7. 返回结构化答案和来源位置
```

示例输出：

```json
{
  "answer": "Compound 7 was synthesized under ...",
  "supporting_evidence": [],
  "provenance": [
    {
      "doc_id": "paper_001",
      "page": 6,
      "block_id": "block_0032",
      "bbox": [120, 300, 800, 600]
    }
  ],
  "involved_entities": [],
  "retrieval_path": [
    "entity_retrieval",
    "reaction_event_retrieval",
    "provenance_backtracking"
  ],
  "confidence": 0.82,
  "uncertainty": "Yield value was extracted from table text and should be manually verified."
}
```

## 10. 推荐优先级

推荐先做：

```text
OpenChemIE wrapper
-> Evidence schema
-> RDKit normalization
-> PostgreSQL + Qdrant/pgvector
-> basic retrieval
-> ROMA Solver
```

Neo4j、ChemEagle、RAG-Anything 可以随后接入。这样第一阶段风险最低，也最容易快速跑通端到端链路。

