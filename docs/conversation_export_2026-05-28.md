# ChemEvoRAG 项目会话导出

**导出时间**: 2026-05-28
**项目路径**: `D:\Desktop\evo\ChemEvoRAG_Phase1-main`
**Git Remote**: `git@github.com:linxi-1021/ChemEvoRAG.git`

---

## 一、项目概述

ChemEvoRAG Phase 1 是一个化学文献 RAG（检索增强生成）系统，用于从化学论文中提取信息并回答问题。

**当前完成度**: ~75%

---

## 二、已完成的工作

### ✅ 核心功能
- **RDKit 分子归一化** - 化合物名称标准化
- **PDF 解析** - 使用 OpenChemIE 提取化学信息
- **稠密检索** - Qdrant + SentenceTransformer 向量检索
- **LLM 答案合成** - 基于检索结果生成回答

### ✅ 近期修复
1. 批量评估脚本 + LLM-judge 评分
2. 统一 conda 环境名称为 `chemevorag`
3. 检索结果排序优化（置信度排序）
4. 实体查找和别名解析的词法搜索回退
5. 分子名称处理改进
6. 本地模型加载设置（HF_HUB_OFFLINE=1）
7. LLM 响应 JSON 控制字符清理

---

## 三、进行中的工作

### Gap #6: 多论文评估
- **状态**: 进行中
- **位置**: `data/pdfs/` 目录中的 PDF 文件
- **脚本**: `scripts/generate_eval_questions.py` 生成 QA 对
- **评估脚本**: `scripts/eval_questions.py` (已打开)

---

## 四、待完成工作

1. **PostgreSQL 存储** - 持久化数据存储
2. **Neo4j 导入** - 知识图谱构建
3. **配置文件完善** - 系统配置优化
4. **ChemEagle 包装器** - 集成接口

---

## 五、技术栈

### 环境
- **Conda 环境**: `chemevorag`
- **环境配置**: `environment.yml`
- **Python 包**: 已锁定版本

### LLM 配置 (.env, 已 gitignore)
```
API_KEY=sk-gr-72b1ca19eb0edfd7a7bab141b381b2b7880a6ce9
BASE_URL=https://endpoint.greatrouter.com
LLM_MODEL=gpt-5-mini
```

**注意事项**:
- 这是推理模型，需要较大的 max_tokens=16384
- `extra_body={"thinking": {"type": "disabled"}}` 在此模型上似乎无效
- 环境变量名称已去除 OpenAI 品牌标识

---

## 六、关键文件结构

```
ChemEvoRAG_Phase1-main/
├── CLAUDE.md                    # Claude 工作规范
├── config/                      # 配置文件
├── data/
│   └── pdfs/                    # 论文 PDF 文件
├── docs/
│   └── phase1_goal_and_stack.md # Phase 1 目标文档
├── scripts/
│   ├── run_query.py             # 查询脚本（--json, --use-llm）
│   ├── generate_eval_questions.py # 生成评估 QA 对
│   └── eval_questions.py        # 评估脚本
├── src/
│   └── solver/
│       └── llm_solver.py        # LLM 求解器
├── tests/                       # 测试文件
└── environment.yml              # Conda 环境配置
```

---

## 七、用户偏好设置

### 代码风格
- **不要过度强调 OpenAI 品牌** - 使用通用变量名 (API_KEY, BASE_URL, LLM_MODEL)
- **直接 PDF→LLM（多模态）** - 优先使用 PDF 直接输入 LLM 进行多模态处理
- **文件管理** - PDF 和数据文件放入 `data/` 子目录，按类型分类
- **修改代码前需确认** - 不要主动修改代码；先回答问题，等待执行指令
- **单步最短命令** - 终端命令使用单行格式，先 `cd ChemEvoRAG_Phase1-main`，再逐行命令

### Git 纪律
- 文件修改后：`git add` + `git commit` + `git push origin master`
- **提交信息禁止提及 Claude、Anthropic 或 AI 辅助**
- 作者/提交者必须是 `linxi-1021`
- 推送前报告变更内容

### 语言
- 中英文沟通均可

---

## 八、操作规范

### 核心规则

1. **不要修改代码除非明确要求**
   - 用户提问时，回答问题即可
   - 不要立即开始编辑文件

2. **不要绕过或降级功能**
   - 依赖不可用时，报告问题
   - 不要用 dummy/stub/简化回退替代

3. **不要修改项目计划或架构**
   - Phase 1 目标、架构、成功标准在官方文档中定义
   - 修改需要用户批准

4. **使用 superpowers skills**
   - 开始非平凡任务前检查是否有相关技能
   - 使用 brainstorming、systematic-debugging 等结构化思考

5. **代码变更后：报告 → 提交 → 推送**
   - 总结变更文件、变更内容、变更原因
   - 然后 git add/commit/push
   - 提交信息禁止引用 Claude/Anthropic/AI

---

## 九、已知问题

### layoutparser EfficientDet 模型
- **问题**: Dropbox 链接失效
- **状态**: 需要替代下载源
- **详情**: 见记忆文件 `layoutparser-effdet-models-missing.md`

---

## 十、快速恢复指南

### 新会话启动时
1. 告诉 Claude 项目路径：`D:\Desktop\evo\ChemEvoRAG_Phase1-main`
2. 提及本导出文件：`docs/conversation_export_2026-05-28.md`
3. 说明当前任务：多论文评估（eval_questions.py）

### 关键命令
```bash
# 进入项目目录
cd ChemEvoRAG_Phase1-main

# 激活环境
conda activate chemevorag

# 运行查询
python scripts/run_query.py "你的问题"

# 生成评估问题
python scripts/generate_eval_questions.py --extract

# 运行评估
python scripts/eval_questions.py
```

---

## 十一、项目目标（Phase 1）

详见 `docs/phase1_goal_and_stack.md` 和 `Phase1目标达成检查报告_2026-05-16.md`

**成功标准**:
- 从化学论文中准确提取化合物信息
- 支持多论文检索和问答
- LLM 驱动的答案合成
- 可扩展的 RAG 架构

---

## 十二、下一步建议

1. **完成评估流程** - 使用 `eval_questions.py` 评估当前系统
2. **修复已知问题** - layoutparser 模型下载源
3. **数据库集成** - PostgreSQL + Neo4j
4. **性能优化** - 检索准确率和响应时间

---

**导出完成** ✅

本文档包含了 ChemEvoRAG Phase 1 项目的完整上下文，可用于在新会话中快速恢复工作状态。
