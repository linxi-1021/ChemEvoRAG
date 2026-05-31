# ChemEvoRAG Phase 1 实现步骤

## 0. 实现原则

第一阶段目标是先跑通端到端闭环，而不是一次性完成完整 ChemEvoRAG。

优先级如下：

```text
能解析 -> 能结构化 -> 能标准化 -> 能检索 -> 能回答 -> 能追溯
```

每一步都要保留中间产物，方便排错和评估。

## 1. 准备项目骨架

### 1.1 目录

当前已有基础目录：

```text
ChemEvoRAG_Phase1/
  config/
  data/
  docs/
  external/
  scripts/
  src/
```

下一步补齐：

```text
data/raw_pdfs/
data/parsed/
data/evidence/
data/indexes/
data/runs/

src/ingestion/
src/parsing/
src/evidence/
src/normalization/
src/storage/
src/retrieval/
src/solver/
src/evaluation/
```

### 1.2 配置文件

需要创建：

```text
config/parser.yaml
config/storage.yaml
config/retrieval.yaml
config/solver.yaml
```

建议第一版配置只保留最小字段：

```yaml
parser:
  primary: openchemie
  fallback: chemeagle

storage:
  mode: local
  evidence_dir: data/evidence
  parsed_dir: data/parsed

retrieval:
  lexical_top_k: 20
  dense_top_k: 10
  evidence_top_k: 8

solver:
  engine: roma
  require_provenance: true
  require_uncertainty: true
```

## 2. 建立数据 schema

先写 `docs/schemas.md`，再在 `src/evidence/schemas.py` 中实现 Pydantic 模型。

### 2.1 必须实现的 schema

```text
SourceProvenance
DocumentBlock
MoleculeCard
ReactionEventCard
FactCard
EvidencePackage
GroundedAnswer
```

### 2.2 第一版字段原则

每个对象必须包含：

```text
id
doc_id
source/provenance
confidence
raw_payload
errors
```

这样即使解析失败，也可以追踪失败来源。

## 3. PDF ingest 层

创建：

```text
src/ingestion/pdf_ingestor.py
scripts/ingest_pdf.py
```

职责：

```text
1. 接收 PDF 路径
2. 生成 doc_id
3. 复制或登记原始 PDF
4. 调用 parser adapter
5. 输出 data/parsed/{doc_id}.json
```

第一版命令目标：

```bash
python scripts/ingest_pdf.py data/raw_pdfs/example.pdf
```

预期输出：

```text
data/parsed/example.raw_extraction.json
```

## 4. OpenChemIE adapter

创建：

```text
src/parsing/openchemie_adapter.py
```

职责：

```text
1. 包装 external/OpenChemIE 的 PDF/figure/text 抽取能力
2. 将 OpenChemIE 原始输出转成统一 RawExtraction
3. 保留原始输出 raw_payload
4. 捕获异常并写入 errors
```

第一阶段优先抽取：

```text
1. text blocks
2. molecules from figures
3. reactions from figures
4. reactions from text
5. tables if available
```

如果 OpenChemIE 模型权重或系统依赖暂时缺失，adapter 也要能输出失败 JSON，而不是直接中断全流程。

## 5. Evidence Builder

创建：

```text
src/evidence/builder.py
```

职责：

```text
1. RawExtraction -> DocumentBlock
2. molecule candidates -> MoleculeCard
3. reaction candidates -> ReactionEventCard
4. claims/properties/procedures -> FactCard
5. 计算 evidence_completeness
```

第一版 `ReactionEventCard` 最少字段：

```text
reaction_event_id
doc_id
reactants
products
reagents
catalysts
solvents
temperature
time
yield
procedure_text
source
confidence
evidence_completeness
```

## 6. RDKit 标准化层

创建：

```text
src/normalization/rdkit_normalizer.py
```

职责：

```text
1. 校验 SMILES
2. 生成 canonical SMILES
3. 生成 InChIKey
4. 标记 invalid molecule
5. 保留 normalization_errors
```

第一版不强求 name-to-structure 全自动解析。名称解析可以先做 alias table：

```text
config/aliases.yaml
```

示例：

```yaml
DCHA:
  canonical_name: dicyclohexylamine
  smiles: C1CCC(CC1)NC2CCCCC2
```

## 7. 存储层

第一阶段建议先用文件存储跑通，再接数据库。

### 7.1 文件存储 MVP

创建：

```text
src/storage/local_store.py
```

输出：

```text
data/evidence/{doc_id}.molecules.json
data/evidence/{doc_id}.reactions.json
data/evidence/{doc_id}.facts.json
data/evidence/{doc_id}.blocks.json
```

### 7.2 数据库增强

文件存储跑通后，再接：

```text
PostgreSQL: structured evidence
Qdrant 或 pgvector: dense retrieval
Neo4j: graph retrieval
```

数据库不要一开始就阻塞主链路。

## 8. 检索层

创建：

```text
src/retrieval/lexical.py
src/retrieval/entity.py
src/retrieval/reaction.py
src/retrieval/provenance.py
src/retrieval/router.py
```

第一阶段支持五类检索：

```text
1. lexical_search(query)
2. dense_search(query)
3. entity_search(name_or_smiles_or_inchikey)
4. reaction_event_search(filters)
5. provenance_backtrack(evidence_id)
```

router 根据 intent 选择检索路径：

```text
alias_resolution -> entity_search
reaction_condition_query -> entity_search + reaction_event_search + provenance_backtrack
reaction_comparison -> reaction_event_search + lexical_search + provenance_backtrack
paper_local_question -> lexical_search + dense_search + provenance_backtrack
```

## 9. LightRAG 接入

创建：

```text
src/retrieval/lightrag_adapter.py
```

职责：

```text
1. 将 DocumentBlock / FactCard / ReactionEventCard summary 写入 LightRAG
2. 查询时返回候选 evidence id
3. 不直接让 LightRAG 生成最终答案
```

关键原则：

```text
LightRAG 负责召回；
ChemEvoRAG EvidencePackage 负责组织证据；
ROMA Solver 负责最终回答。
```

## 10. ROMA Solver 接入

创建：

```text
src/solver/roma_adapter.py
src/solver/prompts.py
scripts/run_query.py
```

ROMA 负责：

```text
1. query understanding
2. retrieval routing
3. 调用 retrieval tools
4. 组织 EvidencePackage
5. answer generation
6. uncertainty explanation
```

第一版查询命令：

```bash
python scripts/run_query.py "Which reaction has the highest yield?"
```

输出必须是 JSON：

```json
{
  "answer": "",
  "supporting_evidence": [],
  "provenance": [],
  "involved_entities": [],
  "retrieval_path": [],
  "confidence": 0.0,
  "uncertainty": ""
}
```

## 11. ChemEagle 增强

OpenChemIE 跑通后，再接 ChemEagle。

创建：

```text
src/parsing/chemeagle_adapter.py
```

使用场景：

```text
1. OpenChemIE 对反应图解析失败
2. 图片中存在复杂 reaction scheme
3. 需要多 Agent 交叉验证抽取结果
```

ChemEagle 输出也统一转成 RawExtraction，不能让上游格式泄漏到 Evidence Builder。

## 12. RAG-Anything 增强

RAG-Anything 暂时不放在第一条主链路里。

适合在第二阶段接入：

```text
1. 复杂 PDF 多模态混合解析
2. 表格、图片、公式、图文关系检索
3. 多模态 query
```

第一阶段仅保留源码和调研记录。

## 13. 测试样例

创建：

```text
data/raw_pdfs/
docs/evaluation_questions.md
```

建议第一批测试问题：

```text
1. What is compound 7?
2. Which reaction has the highest yield?
3. What solvent was used for the synthesis of compound X?
4. Which page supports this reaction condition?
5. Does DCHA refer to dicyclohexylamine?
```

每个问题至少记录：

```text
expected_answer
expected_evidence
expected_page
failure_notes
```

## 14. 推荐实施顺序

按以下顺序推进：

```text
Step 1: 创建 schema 文档和 Pydantic 模型
Step 2: 写 local_store 文件存储
Step 3: 写 OpenChemIE adapter
Step 4: 跑通单 PDF -> raw_extraction.json
Step 5: 写 Evidence Builder
Step 6: 接 RDKit 标准化
Step 7: 写 lexical/entity/reaction/provenance 检索
Step 8: 接 LightRAG 做增强召回
Step 9: 接 ROMA Solver 输出 GroundedAnswer
Step 10: 加 3-5 篇 PDF 和 10-20 个评测问题
Step 11: 接 ChemEagle 做 fallback
Step 12: 记录失败类型，为 GEPA/Skill Evolution 做准备
```

## 15. 第一阶段完成定义

第一阶段完成时，应该可以运行：

```bash
python scripts/ingest_pdf.py data/raw_pdfs/example.pdf
python scripts/build_evidence.py --doc-id example
python scripts/run_query.py "Which reaction has the highest yield?"
```

并得到包含以下字段的答案：

```text
answer
supporting_evidence
provenance
involved_entities
retrieval_path
confidence
uncertainty
```

这就是 ChemEvoRAG Phase 1 的可交付闭环。

