# ChemEvoRAG Skill Evolution 方案设计

> **版本**: 2.1 — Prompt-only Reflexion + Bootstrap
> **日期**: 2026-07-06
> **背景**: 放弃 v2.0 的全栈进化方案，缩窄到 Phase 1 只做 prompt evolution。原因是当前 Skill YAML 中只有 `system_prompt_ref` 字段真正被运行时消费，改 retrieval/top_k/boost 会出现 "patch 看起来成功但运行时根本没生效" 的假闭环。

---

## 一、Skill Evolution 解决什么

**不是**做一个通用的自进化 Agent 框架。

**是**解决当前 118 题里反复出现、人工很难稳定调好的问题：
- 证据已经检索到了，但 **evidence assessment 误判为 insufficient**
- 表格、entry、yield、compound label 的读取规则不稳定
- 成功题里已经有好的回答模式，但没有被自动抽成 few-shot
- 每次手工改 prompt 容易过拟合一个样本

核心假设（来自 Reflexion, Shinn et al. NeurIPS 2023）：**自然语言反思文本 + 成功经验，可以直接注入 prompt 改进 LLM，不需要参数更新，不需要手写归因规则。**

---

## 二、Phase 1 流程（Prompt-only）

```
Round N:
  ┌─────────────────────────────────────────────┐
  │ 1. Eval     │ 跑全量 118 题，产出 score + trace  │
  │ 2. Reflect  │ LLM 对 score<0.8 生成反思文本        │
  │ 3. Memory   │ 反思存入 data/evolution/memory/     │
  │ 4. Bootstrap│ score≥0.9 抽 few-shot examples     │
  │ 5. Mutate   │ 每个 intent skill 生成 2 个 prompt 候选 │
  │ 6. Validate │ 失败题 + 同 skill holdout 验证       │
  │ 7. Apply    │ 写回 config/prompts/ + skill ref    │
  └─────────────────────────────────────────────┘
```

**只允许修改的字段：**

```yaml
# config/prompts/*.yaml — 完整 prompt 内容
# config/skills/{skill}.yaml:
  strategy.assessment.system_prompt_ref
  strategy.answer_generation.system_prompt_ref
```

**明确禁止修改（Phase 1）：**
- `strategy.retrieval_routing.*`（运行时未消费）
- `strategy.query_rewrite.*`（运行时未完全消费）
- `strategy.evidence_expansion.*`（运行时未消费）
- `templates.*`（运行时未消费）
- `trigger.*`、`dependencies.*`（身份字段）

**Phase 2 才有资格做的事情：**
- `answer_generation_examples` 接入运行时
- `max_evidence_items` 接入运行时
- 更精准的 targeted set 选择

**Phase 3 才有资格做的事情：**
- retrieval_routing / top_k / boost / query_rewrite 自动修改
- 前提：先把这些字段真正接入 RetrievalRouter

---

## 三、每一步的详细设计

### 3.1 Eval（评测）

```
输入: all_questions.json (118题)
输出: eval_results.json + react_logs/
命令: python scripts/eval_questions.py --react --skills
```

不做改动。eval_questions.py 保持现状，`--skills` 加载 Skill YAML + Prompt Registry。

### 3.2 Reflect（LLM 反思）

**来自 Reflexion**: $M_{sr}$（Self-Reflection model）给出自然语言反思。

```
输入: eval_results.json（取 score<0.8 的题）
处理:
  对每题，LLM 收到:
    问题 + system_answer + ground_truth + judge_reasoning
  输出 2-4 句反思:
    - 什么错了（阅读误差/过度保守/表格误读/幻觉）
    - prompt 应该具体怎么改
输出: Reflection[] → data/evolution/memory/{skill_name}.jsonl
```

**为什么不用手写归因规则**：Reflexion 论文的核心发现——自然语言反思比手写规则更有效，因为它能捕捉规则写不出来的微妙问题（比如 「assessment 在这个 case 明明有 yield 数据但说 insufficient，因为表格格式是管道符分隔的而非 Markdown」）。

### 3.3 Memory（情景记忆持久化）

**来自 Reflexion**: episodic buffer，反思文本持久化。

```
存储: data/evolution/memory/{skill_name}.jsonl
格式: {"round": N, "question_id": "q1", "score": 0.0, "reflection": "...", "timestamp": "..."}

加载: 下一轮 Mutate 时，读取最近 10 条此 skill 的反思，注入到 prompt 生成的 context 中
```

### 3.4 Bootstrap（成功样本抽取）

**来自 DSPy** (Khattab et al., NeurIPS 2023): BootstrapFewShot。

```
输入: eval_results.json（取 score≥0.9 的题）
处理:
  按 intent 分组
  抽取最多 3 对 (question + evidence_summary → correct_answer)
输出: 可注入到 ANSWER_GENERATION_SYSTEM 的 few-shot examples
```

**优势**: DSPy 论文证明，从成功 trace 自动抽取 few-shot 往往比 LLM 生成的 prompt 更有效，而且是确定性的（零额外 LLM 调用）。

### 3.5 Mutate（Prompt 候选生成）

```
输入:
  - 当前 prompt (V1)
  - 失败反思（最近 10 条）
  - 失败样本（最多 5 题）
  - Bootstrapped examples（最多 3 对）
  - skill_name + prompt_role (assessment / answer_generation)

LLM 提示:
  "你是优化一个化学RAG系统的{role} prompt。
   这是当前 V1... 这些是最近的失败反思... 这些是失败样本...
   请生成改进的 V2 prompt。保留仍然有效的部分，只改需要改的。"

输出:
  - 2 个候选 prompt (V2a, V2b)
  - temperature 0.0 / 0.3 分别生成，保证多样性
  - 每个候选包含: prompt_ref, content, change_summary
```

### 3.6 Validate（验证）

```
Targeted validation:
  评测集:
    - 目标 failure 题 (3-8 题)
    - 同 skill 成功题 (5-10 题)
  运行次数: 3 次取均值，temperature=0.0
  判定:
    - 目标 failure 的 avg score 提升 ≥ 0.1 OR 至少 50% 目标题从 <0.8 变为 ≥0.8
    - 成功题 avg score 不下降超过 0.03
    - 无新的 score<0.3 题

Full eval:
  只对通过 targeted validation 的候选跑一次全量 118 题
  判定:
    - overall score ≥ baseline - 0.01
    - 各 skill score ≥ baseline_skill - 0.02
```

### 3.7 Apply（写回）

```
1. Snapshot: 备份 config/prompts/ + config/skills/ 到 snapshot 目录
2. 写入 V2 prompt 文件到 config/prompts/{ref}.yaml
3. 修改 skill YAML 中的 system_prompt_ref 指向 V2
4. 更新 skill YAML 中的 changelog
5. Git commit: "evolution: {skill} {role} v2"
```

**关键**: 同时写入 prompt_content_update（新建 V2 文件）和 prompt_ref_update（更新 skill YAML 的 system_prompt_ref）。二者缺一则运行时不会生效。

---

## 四、与文献的对应关系

| 步骤 | 来源 | 为什么这么用 |
|------|------|-------------|
| **Eval** | 项目自身 | 118 题是环境，LLM judge 给出标量分数 |
| **Reflect** | Reflexion $M_{sr}$ | 自然语言反思替代手写归因规则，Reflexion 证明这比结构化规则更有效 |
| **Memory** | Reflexion episodic buffer | 持久化反思，下一轮作为额外 context 注入 prompt 生成 |
| **Bootstrap** | DSPy BootstrapFewShot | 确定性从成功 trace 抽取 few-shot，零额外 LLM 调用 |
| **Mutate** | EvoPrompt + Promptbreeder 简化 | 只取「每个 target 生成 2+ 多样候选」思想，不用完整遗传算法 |
| **Validate** | 项目自身的回归能力 | Targeted → Full 两级门控 |
| **Apply** | 项目自身 | Snapshot → Write → Git commit |

---

## 五、为什么不做这些

| 删除或延后的 | 原因 |
|-------------|------|
| 六级归因模型 | 不是不需要归因，而是 Reflexion 的自然语言反思比手写规则更好地捕捉微妙问题 |
| 复杂 failure clustering | 每个 intent skill 内部直接处理，不需要跨 skill 聚类 |
| 多类型 patch schema | Phase 1 只有 prompt patch |
| retrieval/top_k/boost 自动修改 | 运行时未消费，改了不生效 |
| composition validation | 每次只改 1 个 prompt，天然无冲突 |
| 自动冲突解决 | 不需要 |
| 复杂版本管理 | git commit 就是版本快照 |
| 长期无人值守自动写回 | Phase 1 每轮需要人工确认 |
| 遗传算法种群进化 | 118 题全量评估做种群不现实 |
| RL-based 方法 | 每次 action 成本 30-60min，trial-and-error 不可行 |
| TextGrad 文本梯度 | 检索不是可微操作 |

---

## 六、代码目录结构

```
src/skill_evolution/
  __init__.py              # 只导出保留模块 + 新模块
  types.py                 # FailureType, FailureRecord, SuccessPattern（从 attribution 提取）
  trace.py                 # 保留
  evaluation.py            # 保留
  patch.py                 # 保留（PatchSchema, PromptArtifact 等）
  apply.py                 # 保留（PatchApplier）
  validation.py            # 保留（静态验证）
  regression.py            # 保留
  runtime_validation.py    # 保留（sandbox 验证）
  rollback.py              # 保留
  schemas.py               # 保留
  memory.py                # 新增：Reflexion 记忆
  bootstrap.py             # 新增：DSPy Bootstrap
  prompt_evolver.py        # 新增：Prompt 候选生成
  simple_runner.py         # 新增：最小闭环 runner

scripts/
  evolve_prompt.py         # 新增：CLI 入口
  eval_questions.py        # 保留不变
  evolve.py                # 已归档

archive/skill_evolution_v1/
  # 旧复杂管线全部归档
  runner.py, mutation.py, distillation.py, integration.py,
  llm_analysis.py, success_analysis.py, metrics.py, config.py,
  attribution.py, evolve.py
```

---

## 七、验收标准

**Phase 1 MVP 通过标准**:
1. `python scripts/evolve_prompt.py --analyze-only` — 能正确分组 118 题的成功/失败
2. `python scripts/evolve_prompt.py --evolve --target-intent reaction_comparison` — 能生成 2+ 个 prompt 候选
3. 至少 1 个 prompt 候选通过 targeted validation
4. 通过 targeted 的候选通过 full eval（overall score 不低于 baseline - 0.01）
5. Writeback 后 `python scripts/eval_questions.py --react --skills` 能正常运行
6. Git snapshot 可回滚

**Phase 2 才验收的**:
- answer_generation_examples 自动注入
- max_evidence_items 优化

**Phase 3 才验收的**:
- retrieval / boost / query_rewrite 自动优化（前提是接入运行时）

---

## 八、文献引用

| 论文 | 来源 | 年份 | 本项目采用的设计 |
|------|------|------|----------------|
| **Reflexion** (Shinn et al.) | NeurIPS | 2023 | Actor→Evaluator→Self-Reflection 循环；反思文本持久化到 episodic memory；下一轮作为 context 注入 |
| **DSPy** (Khattab et al.) | NeurIPS/ICML | 2023/2024 | BootstrapFewShot：从成功 trace 确定性抽取 few-shot examples |
| **Voyager** (Wang et al.) | NeurIPS/TMLR | 2023/2024 | Skill 的向量索引 + 自动 curriculum（延后到 Phase 2） |
| **EvoPrompt** (Guo et al.) | 2024 | LLM 作为变异算子，生成 2+ 候选 |
| **Self-Refine** (Madaan et al.) | NeurIPS | 2023 | Prompt 生成后 self-critique 再 regression（延后到 Phase 2） |
