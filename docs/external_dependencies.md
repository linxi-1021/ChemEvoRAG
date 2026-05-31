# 外部项目依赖记录

本文件记录 ChemEvoRAG Phase 1 当前拉取的外部项目、用途和版本状态。

## 已拉取项目

| 项目 | 本地路径 | 当前提交 | 用途 | 阶段定位 |
|---|---|---|---|---|
| OpenChemIE | `external/OpenChemIE` | `d9b50bb Update README.md` | 化学 PDF、图像、文本中的 molecule/reaction 抽取 | 第一优先级 |
| ChemEagle | `external/ChemEagle` | `8100aff Update README.md` | 多 Agent 化学图像/反应/PDF 抽取增强 | 第二优先级 |
| LightRAG | `external/LightRAG` | `405525a Bump API version to 0292` | 轻量图谱增强 RAG、文本+图谱检索 | 第一阶段检索底座 |
| RAG-Anything | `external/RAG-Anything` | `146828f Merge pull request #275 from Abdeltoto/feat/docs-public-asset-urls` | 多模态 RAG 增强，处理复杂文档模态 | 后续增强 |
| ROMA | `/root/ROMA` | `a6e3bb4 Update paper link and citation in README.md` | 外层 Agent Solver、工具调度、trace、GEPA 优化 | Solver/Skill 层 |

## 本地体积

| 项目 | 大小 |
|---|---:|
| OpenChemIE | 5.3 MB |
| ChemEagle | 45 MB |
| LightRAG | 22 MB |
| RAG-Anything | 7.0 MB |

## 拉取方式

所有项目均采用浅克隆方式拉取。

```bash
git clone --depth 1 https://github.com/CrystalEye42/OpenChemIE.git external/OpenChemIE
git clone --depth 1 https://github.com/CYF2000127/ChemEagle.git external/ChemEagle
git clone --depth 1 https://github.com/HKUDS/LightRAG.git external/LightRAG
git clone --depth 1 --filter=blob:none --single-branch https://github.com/HKUDS/RAG-Anything.git external/RAG-Anything
```

`RAG-Anything` 第一次普通浅克隆时出现 GitHub HTTP2 传输中断，随后使用 `--filter=blob:none` 成功拉取。

## 第一阶段使用优先级

### 必须先用

```text
1. OpenChemIE
2. RDKit
3. PostgreSQL 或本地 JSON/SQLite 临时存储
4. Qdrant 或 pgvector
5. LightRAG
6. ROMA
```

### 后续接入

```text
1. ChemEagle
2. Neo4j
3. RAG-Anything
4. GEPA 自动优化
```

## 重要说明

`external/` 中的项目是上游依赖源码，不建议直接大改。

第一阶段应在 `src/` 中编写 wrapper 和 adapter：

```text
src/parsing/openchemie_adapter.py
src/parsing/chemeagle_adapter.py
src/retrieval/lightrag_adapter.py
src/solver/roma_adapter.py
```

这样可以保留上游项目的完整性，也方便后续更新或替换。

