# ChemEvoRAG Phase 1 当前进度报告

生成日期：2026-05-16

## 1. 当前阶段结论

当前项目已经完成 Phase 1 MVP 的基础骨架，并打通了一个不依赖论文 PDF 的最小分子身份问答闭环：

```text
PubChem 分子数据
  -> MoleculeCard / FactCard 本地证据
  -> JSON 小型知识图谱
  -> 本地实体检索
  -> GroundedAnswer 结构化回答
```

当前最小测试问题：

```text
What is DCHA?
DCHA是什么？
```

系统能够从 PubChem seed 数据中回答：

```text
dicyclohexylamine is the best local molecule identity match.
SMILES: C1CCC(CC1)NC2CCCCC2.
InChIKey: XBPCUCUWBYBCDP-UHFFFAOYSA-N.
Source id: pubchem:7582.
```

## 2. 项目目录

当前工作目录：

```text
/root/ChemEvoRAG_Phase1
```

主要目录：

```text
ChemEvoRAG_Phase1/
  config/
  data/
    evidence/
    kg/
  docs/
  external/
  scripts/
  src/
    evidence/
    ingestion/
    normalization/
    parsing/
    retrieval/
    solver/
    storage/
  tests/
```

## 3. Conda 环境与依赖

已创建 Conda 环境：

```text
chemevorag
```

环境文件：

```text
environment.yml
```

当前依赖：

```yaml
name: chemevorag
channels:
  - defaults
dependencies:
  - python=3.10
  - pydantic
  - pyyaml
  - pytest
  - pip
  - pip:
      - neo4j
```

依赖用途：

| 包 | 用途 | 当前状态 |
|---|---|---|
| Python 3.10 | 项目运行环境 | 已使用 |
| pydantic | 证据对象、问答对象、schema 校验 | 已使用 |
| pyyaml | Neo4j 配置读取 | 已使用 |
| pytest | 单元测试 | 已使用 |
| neo4j | Neo4j 分子身份图谱查询接口 | 已加入依赖并测试 mock |
| urllib.request | PubChem PUG REST 下载，使用 Python 标准库 | 已使用 |
| json/pathlib/argparse | 本地文件、脚本和 CLI | 已使用 |

尚未正式接入的依赖：

| 依赖/项目 | 计划用途 | 当前状态 |
|---|---|---|
| RDKit | SMILES 校验、canonical SMILES、InChIKey 生成 | 未安装，暂未接入 |
| OpenChemIE | PDF/图文/化学信息解析 | 已拉取外部项目，adapter 已写，真实模型依赖未打通 |
| ChemEagle | 化学图像/反应解析增强 | 已拉取外部项目，暂未接入真实流程 |
| LightRAG | 图谱/文本检索增强 | 已写本地 adapter，未运行真实 LightRAG 服务 |
| RAG-Anything | 多模态 RAG 增强 | 已拉取外部项目，暂未接入 |
| ROMA | Agent solver / 任务规划 | 已写 adapter 边界，当前回答使用本地 deterministic solver |
| Neo4j 服务 | 真实分子图谱查询 | 接口已完成，连接配置需用户填写 |

## 4. 已拉取/准备的外部项目

外部项目位于：

```text
external/
```

已准备的组合方向：

| 项目 | 预期职责 | 当前状态 |
|---|---|---|
| OpenChemIE | 化学 PDF、图片、文本解析 | 已拉取，已有包装 adapter |
| ChemEagle | 复杂化学图像和反应抽取增强 | 已拉取，后续接入 |
| LightRAG | 图谱增强检索 | 已准备 adapter |
| RAG-Anything | 多模态 RAG 后续增强 | 已拉取，后续接入 |
| ROMA | Agent 编排和推理规划 | `/root/ROMA` 已存在，当前只做可用性边界 |

## 5. 已完成模块

### 5.1 Evidence schema

文件：

```text
src/evidence/schemas.py
src/evidence/__init__.py
docs/schemas.md
```

已实现的核心 Pydantic schema：

```text
SourceProvenance
DocumentBlock
RawExtractionItem
RawExtraction
MoleculeMention
MoleculeImageSource
MoleculeCard
MoleculeIdentityCandidate
ReactionParticipant
YieldValue
ReactionEventCard
FactCard
CandidateEvidence
EvidencePackage
SupportingEvidence
GroundedAnswer
```

已具备能力：

- 所有证据对象保留 `doc_id`
- 支持 `raw_payload`
- 支持 `errors`
- 支持 provenance/source 信息
- 支持 Molecule、Reaction、Fact、Answer 的结构化表达

### 5.2 Neo4j 分子身份查询接口

文件：

```text
src/normalization/neo4j_identity_client.py
config/neo4j.yaml.example
docs/neo4j_identity_interface.md
tests/test_neo4j_identity_client.py
```

已实现接口：

```python
find_by_name(name: str)
find_by_alias(alias: str)
find_by_smiles(smiles: str)
find_by_inchikey(inchi_key: str)
resolve_molecule(molecule: MoleculeCard)
close()
```

设计特点：

- 使用官方 `neo4j` Python driver
- 不写死用户图谱中的 label、字段名、连接信息
- 使用 YAML 配置读取：
  - `uri`
  - `username`
  - `password`
  - `database`
  - `molecule_label`
  - `name_fields`
  - `alias_fields`
  - `smiles_field`
  - `inchikey_field`
  - `id_field`
- Cypher 查询使用参数绑定
- 查询失败返回空列表
- 配置缺失或连接失败时抛出清晰异常

当前限制：

- 还没有连接你的真实 Neo4j 图谱
- 尚未执行真实 Neo4j 集成测试
- 当前 PubChem seed 测试使用的是本地 JSON 证据，不是 Neo4j 查询

### 5.3 LocalStore 本地 JSON 存储层

文件：

```text
src/storage/local_store.py
src/storage/__init__.py
tests/test_local_store.py
```

已实现接口：

```python
save_parsed(doc_id, payload)
load_parsed(doc_id)
save_blocks(doc_id, blocks)
load_blocks(doc_id)
save_molecules(doc_id, molecules)
load_molecules(doc_id)
save_reactions(doc_id, reactions)
load_reactions(doc_id)
save_facts(doc_id, facts)
load_facts(doc_id)
```

固定输出路径：

```text
data/parsed/{doc_id}.raw_extraction.json
data/evidence/{doc_id}.blocks.json
data/evidence/{doc_id}.molecules.json
data/evidence/{doc_id}.reactions.json
data/evidence/{doc_id}.facts.json
```

实现细节：

- 自动创建目录
- UTF-8 编码
- `indent=2`
- `ensure_ascii=False`
- Pydantic 对象使用 `model_dump(mode="json")`
- 缺失文件抛出 `FileNotFoundError`
- malformed JSON 保持原始 `json.JSONDecodeError`
- schema 不匹配时保留 Pydantic validation error

### 5.4 PDF ingestion 与 OpenChemIE adapter

文件：

```text
src/ingestion/pdf_ingestor.py
src/ingestion/__init__.py
src/parsing/openchemie_adapter.py
src/parsing/__init__.py
scripts/ingest_pdf.py
tests/test_pdf_ingestor.py
tests/test_openchemie_adapter.py
```

已实现能力：

- `scripts/ingest_pdf.py` 可接收 PDF 路径
- 生成 `doc_id`
- 调用 parser adapter
- 输出 `RawExtraction`
- OpenChemIE adapter 使用 lazy import
- 当 OpenChemIE 依赖或模型不可用时，不直接中断全流程，而是返回 failed extraction 和错误说明

当前限制：

- 尚未完成真实 PDF 解析模型依赖安装
- 尚未验证真实化学论文 PDF 的抽取质量
- 当前 PubChem seed 测试主动绕过 paper/PDF 流程

### 5.5 Evidence Builder

文件：

```text
src/evidence/builder.py
scripts/build_evidence.py
tests/test_evidence_builder.py
```

已实现能力：

- 将 `RawExtraction` 转换为证据对象
- 输出：
  - `DocumentBlock`
  - `MoleculeCard`
  - `ReactionEventCard`
  - `FactCard`
- 保留原始 payload 和错误信息
- 可将生成的证据保存到 LocalStore

当前限制：

- 真实抽取效果依赖上游 parser
- 分子标准化还没有接 RDKit
- 复杂 reaction/event 归一化仍需增强

### 5.6 本地检索层

文件：

```text
src/retrieval/common.py
src/retrieval/lexical.py
src/retrieval/entity.py
src/retrieval/reaction.py
src/retrieval/provenance.py
src/retrieval/router.py
src/retrieval/lightrag_adapter.py
src/retrieval/__init__.py
tests/test_retrieval.py
tests/test_lightrag_adapter.py
```

已实现检索通道：

| 检索类型 | 文件 | 当前能力 |
|---|---|---|
| Lexical retrieval | `lexical.py` | 对 blocks/facts/reactions 做关键词检索 |
| Entity retrieval | `entity.py` | 按名称、别名、SMILES、InChIKey、本地 ID 检索 MoleculeCard |
| Reaction retrieval | `reaction.py` | 按反应物、产物、条件、产率检索 ReactionEventCard |
| Provenance backtracking | `provenance.py` | 从证据 ID 回查来源 |
| Retrieval router | `router.py` | 根据问题意图选择检索通道 |
| LightRAG adapter | `lightrag_adapter.py` | 生成本地 JSONL manifest，可接外部 rag client |

中文意图识别已加入：

```text
是什么
指的是
别名
```

因此下面的问题会进入实体/别名检索：

```text
DCHA是什么？
```

### 5.7 Solver / 回答层

文件：

```text
src/solver/roma_adapter.py
src/solver/prompts.py
src/solver/__init__.py
scripts/run_query.py
tests/test_solver.py
```

已实现能力：

- `ChemRAGSolver` 使用本地 retrieval router 生成答案
- 输出标准 `GroundedAnswer`
- 包含：
  - `answer`
  - `supporting_evidence`
  - `provenance`
  - `involved_entities`
  - `retrieval_path`
  - `confidence`
  - `uncertainty`
  - `raw_payload`
- 对分子身份类问题进行了更直接的答案合成：
  - 名称
  - 别名
  - SMILES
  - InChIKey
  - PubChem/ElementKG ID

当前限制：

- 默认不调用真实 ROMA
- 当前是 deterministic local answer synthesis
- 没有 LLM 自然语言推理

## 6. PubChem seed 知识图谱测试

### 6.1 新增文件

```text
scripts/bootstrap_pubchem_seed.py
docs/pubchem_seed_test.md
tests/test_pubchem_seed.py
```

生成的数据文件：

```text
data/evidence/pubchem_seed.molecules.json
data/evidence/pubchem_seed.facts.json
data/kg/pubchem_seed_graph.json
```

### 6.2 数据来源

数据来自 PubChem PUG REST。

脚本使用 Python 标准库：

```python
urllib.request
urllib.parse
json
```

请求字段：

```text
CanonicalSMILES
IsomericSMILES
InChIKey
IUPACName
MolecularFormula
MolecularWeight
```

实际 PubChem 返回中也兼容：

```text
SMILES
ConnectivitySMILES
```

因此脚本会按以下优先级填充 SMILES：

```text
CanonicalSMILES
SMILES
ConnectivitySMILES
IsomericSMILES
```

### 6.3 Seed 分子

| 名称 | Alias | PubChem ID | SMILES | InChIKey |
|---|---|---|---|---|
| dicyclohexylamine | DCHA, di(cyclohexyl)amine | pubchem:7582 | `C1CCC(CC1)NC2CCCCC2` | `XBPCUCUWBYBCDP-UHFFFAOYSA-N` |
| ethanol | EtOH | pubchem:702 | `CCO` | `LFQSCWFLJHTTHZ-UHFFFAOYSA-N` |
| benzaldehyde | | pubchem:240 | `C1=CC=C(C=C1)C=O` | `HUMNYLRZRPPJDN-UHFFFAOYSA-N` |
| aniline | | pubchem:6115 | `C1=CC=C(C=C1)N` | `PAYRUJLWNCNPSJ-UHFFFAOYSA-N` |
| aspirin | acetylsalicylic acid | pubchem:2244 | `CC(=O)OC1=CC=CC=C1C(=O)O` | `BSYNRYMUTXBXSQ-UHFFFAOYSA-N` |

### 6.4 生成的 JSON 知识图谱

图谱文件：

```text
data/kg/pubchem_seed_graph.json
```

图谱结构：

```json
{
  "graph_id": "pubchem_seed",
  "source": "PubChem PUG REST",
  "nodes": [],
  "edges": []
}
```

节点类型：

```text
molecule
alias
identifier
```

边类型：

```text
alias_of
identifies
```

示例关系：

```text
alias:DCHA -> alias_of -> pubchem:7582
identifier:CID:7582 -> identifies -> pubchem:7582
```

当前生成规模：

```text
5 个 molecule seed
25 个 nodes
20 个 edges
```

### 6.5 运行 seed 构建

命令：

```bash
conda run -n chemevorag python scripts/bootstrap_pubchem_seed.py
```

成功输出示例：

```json
{
  "doc_id": "pubchem_seed",
  "molecule_count": 5,
  "fact_count": 5,
  "molecules_path": "/root/ChemEvoRAG_Phase1/data/evidence/pubchem_seed.molecules.json",
  "facts_path": "/root/ChemEvoRAG_Phase1/data/evidence/pubchem_seed.facts.json",
  "kg_path": "/root/ChemEvoRAG_Phase1/data/kg/pubchem_seed_graph.json",
  "test_question": "What is DCHA?"
}
```

### 6.6 运行测试问题

英文问题：

```bash
conda run -n chemevorag python scripts/run_query.py "What is DCHA?" --doc-id pubchem_seed
```

中文问题：

```bash
conda run -n chemevorag python scripts/run_query.py "DCHA是什么？" --doc-id pubchem_seed
```

核心返回：

```text
dicyclohexylamine is the best local molecule identity match.
Aliases: DCHA, di(cyclohexyl)amine, N-cyclohexylcyclohexanamine.
SMILES: C1CCC(CC1)NC2CCCCC2.
InChIKey: XBPCUCUWBYBCDP-UHFFFAOYSA-N.
Source id: pubchem:7582.
```

检索路径：

```text
entity_retrieval
provenance_backtracking
```

说明：

- 当前答案来自本地 `MoleculeCard`
- 证据来源是 `pubchem_seed`
- 没有调用论文解析
- 没有调用 LLM
- 没有调用真实 Neo4j

## 7. 当前测试状态

测试命令：

```bash
conda run -n chemevorag pytest
```

当前结果：

```text
46 passed
```

测试覆盖：

| 测试文件 | 覆盖内容 |
|---|---|
| `tests/test_schemas.py` | Evidence schema 基础校验 |
| `tests/test_neo4j_identity_client.py` | Neo4j 查询接口 mock 测试 |
| `tests/test_local_store.py` | LocalStore 保存/读取测试 |
| `tests/test_openchemie_adapter.py` | OpenChemIE adapter 失败/兼容路径 |
| `tests/test_pdf_ingestor.py` | PDF ingest 层 |
| `tests/test_evidence_builder.py` | RawExtraction 到 evidence 构建 |
| `tests/test_retrieval.py` | lexical/entity/reaction/provenance/router |
| `tests/test_lightrag_adapter.py` | LightRAG adapter 本地 manifest |
| `tests/test_solver.py` | solver 和 run_query CLI |
| `tests/test_pubchem_seed.py` | PubChem seed 构建、KG JSON、DCHA 问答 |

## 8. 当前已打通流程

### 8.1 已完全打通

```text
本地 schema 定义
  -> 本地 JSON 存储
  -> 本地证据检索
  -> 本地结构化回答
```

```text
PubChem 小分子下载
  -> MoleculeCard / FactCard
  -> JSON 小图谱
  -> DCHA 身份问答
```

### 8.2 部分打通

```text
PDF 输入
  -> OpenChemIE adapter
  -> RawExtraction
```

说明：接口和失败路径已打通，但真实 OpenChemIE 模型依赖未完成。

```text
Neo4j molecule identity client
  -> MoleculeIdentityCandidate
  -> MoleculeCard 更新
```

说明：代码和 mock 测试已完成，但需要用户填写真实配置并连接图谱。

```text
LightRAG adapter
  -> 本地 manifest
  -> 可接外部 rag_client
```

说明：本地索引 manifest 已可生成，真实 LightRAG 服务/库未接入。

```text
ROMA adapter
  -> 可检测 `/root/ROMA`
  -> 保留 solve 边界
```

说明：当前回答未使用真实 ROMA solver。

## 9. 还没有打通的问题

如果你的目标是完整 ChemEvoRAG，目前主要缺口如下：

| 缺口 | 影响 | 建议下一步 |
|---|---|---|
| 真实 PDF 解析未跑通 | 无法从论文自动抽取 molecule/reaction/fact | 安装并验证 OpenChemIE/ChemEagle 依赖和模型 |
| RDKit 未安装 | 无法本地生成 canonical SMILES/InChIKey | 在 conda env 中加入 RDKit，并实现 normalizer |
| PubChem seed 图谱未导入 Neo4j | 当前 KG 是 JSON，不支持 Neo4j 图查询 | 增加 JSON KG -> Neo4j import 脚本 |
| Neo4j 真实配置未填写 | 无法查询你的分子图谱 | 填写 `config/neo4j.yaml` 并运行集成测试 |
| LightRAG 未真实接入 | 当前检索是本地简化版 | 接入 LightRAG 或保持当前 router 先迭代 |
| ROMA 未真实参与规划 | 当前回答是 deterministic solver | 后续将 RetrievalRouter 暴露为 ROMA tool |
| 没有向量检索 | 自然语言语义召回弱 | 后续接 Qdrant/pgvector/LightRAG embedding |
| 没有论文 provenance | PubChem 测试没有 page/bbox | PDF parser 打通后补 provenance |

## 10. 推荐下一步

为了继续贴近你的原始思路，建议下一步按这个顺序推进：

```text
Step A: 将 PubChem seed JSON KG 导入 Neo4j
Step B: 用已写好的 Neo4jIdentityClient 查询这个测试图谱
Step C: 让 RetrievalRouter 支持 Neo4j identity retrieval channel
Step D: 安装 RDKit，补 MoleculeCard 标准化
Step E: 再回到 PDF/OpenChemIE，接真实论文解析
```

理由：

- 你已经有自己的 Neo4j 分子图谱
- 当前最大价值是先把“别名/SMILES/InChIKey -> 分子身份”查准
- PDF 解析复杂度较高，可以在分子身份链路稳定后再接入
- PubChem seed 可以作为 Neo4j 图谱接口的最小可重复测试集

## 11. 常用命令

运行全部测试：

```bash
conda run -n chemevorag pytest
```

重新生成 PubChem seed：

```bash
conda run -n chemevorag python scripts/bootstrap_pubchem_seed.py
```

查询 DCHA：

```bash
conda run -n chemevorag python scripts/run_query.py "What is DCHA?" --doc-id pubchem_seed
```

中文查询：

```bash
conda run -n chemevorag python scripts/run_query.py "DCHA是什么？" --doc-id pubchem_seed
```

运行 PDF ingest 接口：

```bash
conda run -n chemevorag python scripts/ingest_pdf.py data/raw_pdfs/example.pdf
```

从 parsed JSON 构建 evidence：

```bash
conda run -n chemevorag python scripts/build_evidence.py --doc-id example
```

## 12. 当前状态一句话总结

当前已经具备一个可测试、可复现的本地 ChemEvoRAG Phase 1 骨架；PubChem 分子身份小图谱测试已经跑通。下一步最合适的是把这个 seed 图谱导入 Neo4j，并让真实 Neo4j 查询成为 molecule normalization/retrieval 的一条正式通道。
