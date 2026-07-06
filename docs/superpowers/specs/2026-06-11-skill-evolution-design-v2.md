# Skill Evolution 完整设计文档 v2

## 架构图标号

- A. 身份与接口层 (frozen)
- B. 策略与规划层 (evolvable)
- C. 任务模板模块 (evolvable)
- D. 证据与校验模块 (evolvable)
- E. 案例与分析层

## 总体流程

```
Skill Storage (A/B/C/D/E)
  → [1] 轨迹采集 (trace.py)
  → [2] 结果评估 (evaluation.py)
  → [3] 失败归因 + 成功归纳 (attribution.py) → 存入 E
  → ┌ [4A] LLM 定向变异 (mutation.py)
  → └ [4B] 模板归纳 (distillation.py) → 存入 C
  → [5] 候选整合 + validator adjustment (integration.py)
  → [6] 验证 (validation.py: schema + semantic + regression + compose)
  → [7] 回写 (apply.py + rollback.py)
  → BACK TO Skill Storage ◄─ 闭环
```

## 模块结构

```
src/skill_evolution/
├── schemas.py          # A.D. Schema 定义 (SkillYAML, PromptEntry)
├── trace.py            # 1. 轨迹标准化 + 字段审计
├── evaluation.py       # 2. 结果评估 (success/partial/failure/coverage_gap)
├── attribution.py      # 3+E. 失败归因 + 成功归纳 + Coverage Gap 检测
├── mutation.py         # 4A. LLM 定向变异 (6 dimensions)
├── distillation.py     # 4B+C. 模板归纳 (add/rewrite/merge)
├── integration.py      # 5. 候选整合 + validator adjustment
├── validator_config.py # D. 证据校验规则
├── validation.py       # 6. schema + semantic + regression + composition
├── apply.py            # 7. 写回 + dry-run diff
├── rollback.py         # 7. 回滚
├── config.py           # 共享配置 (regression_runs, 阈值等)
├── metrics.py          # 跨轮进化指标追踪
└── runner.py           # 管线编排

scripts/
└── evolve.py           # CLI 入口
```

## 各模块详细设计

### 0. config.py（新增）

共享配置项：

```python
@dataclass
class EvolutionConfig:
    regression_runs: int = 3          # 可配置，不硬编码
    max_retries: int = 3
    regression_limit: int | None = None
    regression_seed: int = 42
    workers: int = 1
    max_patches_per_round: int = 2
    patch_confidence_threshold: float = 0.75
    min_score: float = 0.80
    max_score_drop: float = 0.01
    min_targeted_improvement: float = 0.05
    max_failed_cases_increase: int = 1
```

### 1. trace.py — 轨迹标准化 + 字段审计

**职责：** 只负责标准化和审计，不负责归因。

```python
def standardize_trace(eval_result: dict, react_log: str) -> StandardTrace:
    """将 eval_results.json + react_log 统一为标准 TraceRecord"""
    return StandardTrace(
        question_id=...,
        question=...,
        intent=...,
        entities=...,
        retrieval_rounds=[...],   # 结构化
        evidence_items=[...],
        answer=...,
        gold_answer=...,
        score=...,
        stop_reason=...,
        eval_metrics={...},       # 评价指标 (recall, precision, etc.)
        feedback=...,             # judge reasoning / error flags
        annotations=[...],        # 人工或自动标注
    )

def audit_trace_fields(trace: StandardTrace) -> FieldAudit:
    """检查关键字段是否完整"""
    return FieldAudit(
        question_id=trace.question_id,
        field_status={...},     # {field: "available"|"missing"}
        missing_fields=[...],
        trace_quality="complete"|"incomplete_trace"|"minimal",
    )
```

### 2. evaluation.py — 结果评估

**职责：** 判断 each question 的 outcome。

```python
class OutcomeType:
    SUCCESS = "success"           # score >= 0.8
    PARTIAL_SUCCESS = "partial"   # 0.3 <= score < 0.8
    SYSTEM_FAILURE = "failure"    # score < 0.3 (非 coverage_gap)
    COVERAGE_GAP = "coverage_gap"  # corpus doesn't contain answer

def evaluate_outcome(trace: StandardTrace) -> OutcomeResult:
    """判断结果 + 识别 coverage gap"""
    if trace.score >= 0.8:
        return OutcomeResult(OutcomeType.SUCCESS)
    if trace.score >= 0.3:
        return OutcomeResult(OutcomeType.PARTIAL_SUCCESS)
    if is_coverage_gap(trace):
        return OutcomeResult(OutcomeType.COVERAGE_GAP, action="record_to_backlog")
    return OutcomeResult(OutcomeType.SYSTEM_FAILURE, action="generate_patch")
```

**Coverage Gap 处理：**
- 不进入本轮 Skill 更新
- 写入 `data/evolution/backlog/coverage_gaps.json`
- 供后续 corpus/index 补全使用

### 3. attribution.py — 失败归因 + 成功归纳 + E.案例层

**职责：** 规则粗判 + LLM 复核 + 模式提取。

```
Failure (system_failure)
  → init_attribution(trace) → CandidateAttribution
  → build_analysis_clusters(failures) → AnalysisCluster (跨 skill)
  → llm_verify_attribution(cluster, traces) → VerifiedAttribution

Partial Success
  → mark_as_actionable_or_queue

Success
  → mine_success_pattern(trace) → SuccessPattern
  → 写入 E.案例层

# E.案例层结构
class CaseLibrary:
    failures: list[FailureRecord]           # 失败案例
    successes: list[SuccessPattern]          # 成功模式
    coverage_gaps: list[CoverageGapRecord]   # 覆盖缺口
    analysis_notes: list[AnalysisNote]       # 分析笔记
    # AnalysisNote = {round_id, summary, findings, recommendations}

Coverage Gap
  → record_to_backlog
  → 写入 corpus/index backlog
  → 不进入本轮的 patch generation
```

**规则输出格式（候选结论，非最终判定）：**

```json
{
  "question_id": "q3",
  "candidate_failure_types": [
    {"type": "assessment_false_negative", "confidence": 0.62, "signals": [...]}
  ],
  "candidate_target_paths": ["strategy.assessment"],
  "candidate_prompt_role": "evidence_assessment",
  "uncertainty_flags": [...],
  "needs_llm_review": true
}
```

**原则：先归因，再进化；失败驱动策略修正，成功驱动任务模板优化；默认不直接修改 ElementKG Core。**

### 4. mutation.py — 4A. LLM 定向变异

**核心：** 基于失败分析生成 6 个 dimension 的候选 patch。

**6 个 Mutation Dimension：**

```
prompt             → 改 evidence assessment/answer generation prompt
retrieval_routing  → 调整 primary_channels / top_k / boost_weights
query_rewrite      → 调整 fallback_terms / diversity_threshold / max_variants
evidence_expansion → 调整 max_expanded_blocks / expand_from_block_neighbors
answer_generation  → 改 answer prompt / max_evidence_items
stop_condition     → 调整 failure_labels / assessment thresholds
```

**每步过程：**

```
Stage 2 LLM 分析 cluster
  → 判断哪些 dimension 需要 mutate
  → 每个 dimension 输出 analysis + recommended changes

Stage 3 LLM prompt generation（仅 prompt/answer_generation 需要）
  → 基于 failure 实例生成 V2 prompt
  → 带 failure evidence 的针对性规则

rule-based（retrieval_routing / query_rewrite / evidence_expansion）
  → 基于 channel stats / query similarity / expansion gaps 生成参数 patch
```

**mutable_paths / frozen_paths 白名单校验：**

```python
def validate_dimension_mutable(dimension: str, skill_config: dict) -> bool:
    """检查该 dimension 的目标 path 是否在 skill 的 mutable_paths 中
    且不在 frozen_paths 中"""
    targets = MUTATION_DIMENSIONS[dimension].targets
    mutable = skill_config["evolution"]["mutable_paths"]
    frozen = skill_config["evolution"]["frozen_paths"]
    for t in targets:
        if t in frozen:
            return False
        if not any(t == m or t.startswith(m + ".") for m in mutable):
            return False
    return True
```

### 5. distillation.py — 4B. 模板归纳 + C.任务模板

**职责：** 从成功模式蒸馏模板，支持 add / rewrite / merge。

```
template_add:    pattern frequency >= 3 → 新模板
template_rewrite: 已有模板 applicable_when 太窄/太宽 → 调整 trigger
template_merge:   相似度 > 0.8 → 合并重叠模板
```

### 6. integration.py — 5. 候选整合

**职责：** 合并 4A + 4B 候选，解决冲突，排序，输出 selected。

```
Step 1：合并 mutation_patches + distillation_patches
Step 2：冲突检测（同 skill + 同 target_path）
Step 3：冲突解决（按 confidence × impact 排序，保留最优）
Step 4：依赖排序（prompt_content_update 必须在 prompt_ref_update 之前）
Step 5：Validator & schema adjustment
  ├── 调整 evidence sufficiency 阈值
  ├── 调整 citation accuracy 要求
  └── 调整 hallucination guard 规则
Step 6：输出 IntegrationResult
  ├── selected_patches
  ├── rejected_patches (含拒绝原因)
  └── validator_adjustments
```

### 7. validation.py — 6. 验证 + D.证据校验

**四层验证：**

```
Layer 1：Schema validation
  - 检查 patch 在 mutable_paths / frozen_paths 约束内
  - 检查 target_file 存在（update 时）/ parent_dir 存在（add 时）
  - 检查 prompt_ref 存在或有 dependency 创建

Layer 2：Semantic validation
  - prompt content 非空且 != V1
  - template 有 trigger_conditions
  - YAML 格式正确
  - 不产生循环依赖

Layer 3：Evidence & Validator Rules (D层)
  - 证据充分性规则是否保留
  - 引用准确性检查是否保留
  - 幻觉防护规则是否保留

Layer 4：Regression validation
  - 配置项 regression_runs（默认 3）
  - baseline = 原始 eval 结果（不重跑）
  - after = patch 后 eval（regression_runs 次并行 + 平均）
  - targeted_improvement = after_score - original_score
  - global score 不退化
  - Composition validation（多 patch）
```

### 8. apply.py — 7. 写回

**流程：**

```python
def apply_patches(selected, skill_configs, snapshot_mgr, mode="validate-only"):
    if mode == "validate-only":
        # 默认不写回
        return {"applied": 0, "reason": "validate-only"}

    if mode == "apply":
        # --dry-run diff 输出
        for patch in selected:
            print_diff(patch)  # before / after / reason

        # 创建 snapshot
        snapshot_mgr.snapshot_before()

        # 写回
        for patch in dependency_order(selected):
            if patch.prompt_artifact:
                write_prompt_file(patch)
            write_skill_yaml(patch)

        snapshot_mgr.snapshot_after()

        # Post-apply full eval
        result = run_full_eval()
        if result.degraded:
            snapshot_mgr.rollback()
            return {"decision": "rollback", ...}

        return {"decision": "keep", "applied": len(selected), ...}
```

**默认不修改 ElementKG Core；只允许把 coverage gap 写入 backlog。**

## CLI

```bash
python scripts/evolve.py --analyze-only     # 1-3
python scripts/evolve.py --dry-run          # 1-5, 含 diff 输出
python scripts/evolve.py --validate-only    # 1-6, 含 regression
python scripts/evolve.py --apply            # 1-7, 写回
```

## 配置项

```python
EvolutionConfig:
  regression_runs: int = 3        # 可配置
  regression_limit: int | None    # None = full eval
  regression_seed: int = 42
  workers: int = 1
  max_retries: int = 3            # LLM retry
  max_patches_per_round: int = 2
  patch_confidence_threshold: float = 0.75
```

## 修正清单

| # | 修正内容 | 状态 |
|---|---------|------|
| 1 | trace.py 只做标准化 + 审计，不做归因 | ✅ |
| 2 | 结果分类放入 evaluation.py | ✅ |
| 3 | Coverage Gap 写入 backlog，不进本轮更新 | ✅ |
| 4 | regression_runs 从配置读取，不硬编码 | ✅ |
| 5 | mutation.py 增加 mutable_paths/frozen_paths 白名单 | ✅ |
| 7 | apply.py 增加 dry-run diff 输出（before/after/reason） | ✅ |
| 8 | regression baseline 使用原始 eval 结果，不重跑 | ✅ |
| 10 | 默认不修改 ElementKG Core，只写入 backlog | ✅ |
| 11 | max_patches_per_round 在 integration.py 执行 top-N | ✅ |
| 12 | backlog/coverage_gaps.json 格式定义 | ✅ |
| 13 | Partial Success uncertain_queue 规则定义 | ✅ |
| 14 | Prompt 版本号递增规则 | ✅ |
| 15 | Mutation Dimension 与架构图术语对齐 | ✅ |
| 16 | FailureType 补上 TOOL_ERROR + EVIDENCE_EXPANSION_ERROR 改名为 graph_expansion_error | ✅ |
| 17 | Integration Step 5 CandidateType 枚举定义 | ✅ |
| 18 | graph_expansion 维度补上反应图/跨 section 扩展 | ✅ |
| 19 | distillation.py 增加跨 skill 成功经验升级 (cross-skill promotion) | ✅ |
| 21 | 跨轮进化指标追踪 metrics.json | ✅ |

## 遗漏补充（第 3 批）

### 遗漏 19：跨 skill 成功经验升级

```python
# distillation.py
def promote_to_cross_skill_rule(
    success_pattern: SuccessPattern,
    source_skill: str,
    compatible_skills: list[str],
) -> CrossSkillPromotion | None:
    """评估一次成功经验是否能升级为跨 skill 通用规则"""

    # Gate 1：源 skill 中足够成熟
    if success_pattern.frequency < 5:
        return None

    # Gate 2：目标 skill 与源 skill 结构相似
    for target in compatible_skills:
        similarity = compute_skill_similarity(source_skill, target)
        # similarity 基于：
        #   - shared evidence types
        #   - overlapping query structures
        #   - comparable output formats
        if similarity > 0.7:
            yield CrossSkillPromotion(
                source_skill=source_skill,
                target_skill=target,
                template=success_pattern.as_template(),
                rationale=f"Cross-skill promotion: pattern '{success_pattern.pattern}' "
                          f"from {source_skill} → {target} (similarity={similarity:.2f})",
                confidence=success_pattern.avg_score * similarity * 0.8,
            )

# 跨 skill promotion 生成的 template_rewrite patch
# 会被 integration.py 按正常流程收归，
# 但 confidence 被乘以 similarity 因子，优先级低于同 skill 内的蒸馏
```

### 补充模块：metrics.py — 跨轮进化追踪

**职责：** 每轮 evolution 结束时写入趋势数据。

```python
@dataclass
class RoundMetrics:
    run_id: str
    before_score: float       # 原始 eval 的 average_score
    after_score: float | None  # apply 后的 score（validate-only 时为 None）
    score_delta: float
    applied_patches: list[str]
    rejected_patches: list[str]
    resolved_failures: list[str]   # 本轮修好的 failure IDs
    new_failures: list[str]         # 本轮新出现的 failure IDs
    persistent_failures: list[str]  # 多轮仍未修复的 failure IDs

# 输出到 data/evolution/metrics.json
{
  "rounds": [
    {
      "run_id": "run_2026-06-11_000632",
      "before_score": 0.866,
      "after_score": 0.873,
      "score_delta": 0.007,
      "applied_patches": ["patch_1"],
      "rejected_patches": ["patch_2", "patch_3"],
      "resolved_failures": ["q_batch2_3"],
      "new_failures": [],
      "persistent_failures": ["q3"]
    }
  ],
  "trend": {
    "score_trend": [0.866, 0.873],
    "failure_count_trend": [5, 3],
    "patches_applied_total": 1,
    "last_updated": "2026-06-11T12:00:00"
  }
}
```

## 最终修正清单（共 20 项，覆盖 3 批次）

| 批次 | 编号 | 内容 |
|------|------|------|
| 1 | 1-10 | 基础修正 (trace/evaluation/coverage/regression/mutable/apply/baseline/elementkg) |
| 1 | 11-15 | 整合修正 (max_patches/backlog/partial/prompt_version/dimension_naming) |
| 2 | 16-18 | 深度修正 (tool_error/candidate_type/graph_expansion) |
| 3 | 19,21 | 跨 skill 经验升级 + 跨轮指标追踪 |

## 遗漏补充（第 2 批）

### 遗漏 16：FailureType 补全 + 对齐架构图

```python
class FailureType(str, Enum):
    ENTITY_MISS = "entity_miss"                     # 实体遗漏
    ALIAS_MISS = "alias_miss"                       # 别名未解析
    LOCAL_ID_MISS = "local_id_miss"                 # 局部 ID 丢失
    TOOL_ERROR = "tool_error"                       # ← 新增：RDKit/Neo4j/解析错误
    ROUTING_ERROR = "routing_error"                 # 路由错误
    PLANNER_ERROR = "planner_error"                 # 规划错误
    GRAPH_EXPANSION_ERROR = "graph_expansion_error" # ← 改名：图扩展错误
    ASSESSMENT_FALSE_NEG = "assessment_false_negative"
    TABLE_EXTRACTION_ERROR = "table_extraction_error"
    TABLE_REASONING_ERROR = "table_reasoning_error"
    GENERATION_ERROR = "generation_error"           # 答案生成错误
    UNKNOWN_FAILURE = "unknown_failure"

# TOOL_ERROR 处理规则：
#   - 不触发 patch generation
#   - 记录到 evolution_log.txt
#   - 如果同一 tool 连续失败 3 次，触发 alert
```

### 遗漏 17：Integration 候选类型枚举

```python
# integration.py 顶部定义
class CandidateType(str, Enum):
    PATCH = "patch"                              # 通用配置 patch
    PLANNER_PATCH = "planner_patch"              # 规划规则 patch (B层)
    TEMPLATE_ADD = "template_add"                # 模板新增 (C层)
    TEMPLATE_REWRITE = "template_rewrite"        # 模板改写 (C层)
    TEMPLATE_MERGE = "template_merge"            # 模板合并 (C层)
    VALIDATOR_ADJUSTMENT = "validator_adjustment" # 校验器调整 (D层)
    SCHEMA_ADJUSTMENT = "schema_adjustment"      # Schema 调整 (A层，受限)

# 后续每个候选更新都必须标注 candidate_type
# IntegrationResult 按 type 分组输出：
# {
#   "patch_candidates": [...],
#   "planner_patch_candidates": [...],
#   "template_candidates": [...],
#   "validator_adjustments": [...],
#   "schema_adjustments": [...]
# }
```

### 遗漏 18：graph_expansion 维度扩展

```yaml
# schemas.py EvidenceExpansion 扩展
class EvidenceExpansion(ChemEvoBaseModel):
    enabled: bool = True
    # 原有
    expand_from_block_neighbors: bool = True
    expand_from_molecule_mentions: bool = True
    max_expanded_blocks: int = 5

    # ← 新增
    expand_from_reaction_graph: bool = True
      # 从 ReactionEventCard 的 graph neighbor 扩展
    cross_section_linking: bool = False
      # 是否跨 section 链接 evidence (例如 Methods→Results→Tables)
    max_graph_neighbors: int = 10
      # reaction graph 扩展的最大邻居数
```

```yaml
# graph_expansion dimension 的 mutable_paths
mutable:
  - strategy.evidence_expansion.expand_from_block_neighbors
  - strategy.evidence_expansion.expand_from_molecule_mentions
  - strategy.evidence_expansion.max_expanded_blocks
  - strategy.evidence_expansion.expand_from_reaction_graph     # ← 新增
  - strategy.evidence_expansion.cross_section_linking           # ← 新增
  - strategy.evidence_expansion.max_graph_neighbors            # ← 新增
```

## 遗漏补充

### 遗漏 1：max_patches_per_round 执行位置

`integration.py` Step 3 之后执行：

```python
if len(resolved) > config.max_patches_per_round:
    resolved = sorted(resolved, key=lambda p: p.confidence, reverse=True)
    rejected_n = resolved[config.max_patches_per_round:]
    resolved = resolved[:config.max_patches_per_round]
    for r in rejected_n:
        r.rejection_reason = "rejected: exceeds max_patches_per_round"
```

### 遗漏 2：backlog/coverage_gaps.json 格式

```json
{
  "question_id": "q99",
  "question": "What is the melting point of compound X?",
  "gold_answer": "54-55 C",
  "missing_from_corpus": ["compound X characterization data"],
  "corpus_search_hint": "melting point compound X",
  "detected_at": "run_2026-06-11_000632",
  "source_paper": "paper_42.pdf",
  "status": "open"
}
```

### 遗漏 3：Partial Success 处理规则

```python
# attribution.py
def handle_partial_success(trace: StandardTrace) -> PartialAction:
    if trace.score >= 0.7:
        # 接近成功，可能只缺一个小改进
        return PartialAction("minor_patch_candidate")
    elif trace.score >= 0.5:
        # 中等水平，排入观察队列
        return PartialAction("uncertain_queue")
    else:
        # 接近失败，下一轮重点分析
        return PartialAction("review_next_round")

# uncertain_queue 存入 data/evolution/backlog/uncertain_queue.json
# 下一轮 evolution 时一起加载，连续 2 轮 partial → 升级为 failure 分析
```

### 遗漏 4：Prompt 版本号递增规则

```python
# 版本号规则
# 1. 读取当前 prompt 的 version 字段
# 2. 新版本号 = current_version + 1
# 3. V1(1.0.0) → V2(2.0.0)
# 4. 文件名: {prompt_ref_lower}_v{n}.yaml

# 覆盖规则：
# - 上一轮 V2 被 regression 拒绝 → 本轮可以覆盖 V2
# - 上一轮 V2 被 apply 接受 → 本轮生成 V3
# - 覆盖仅在沙箱内发生，不影响 config

# 示例：
# config/prompts/ 中只有 evidence_assessment_system.yaml (V1, version=1.0.0)
# mutation.py 生成: EVIDENCE_ASSESSMENT_SYSTEM_V2, version=2.0.0
# 沙箱 regression 通过 → apply 写回 config
# 下一轮: mutation.py 检测到 V2 已存在 → 生成 V3, version=3.0.0
```

### 遗漏 5：Mutation Dimension 与架构图术语对齐

修正后的 6 个 Dimension（按架构图术语）：

```
prompt             # 改 evidence assessment / answer generation prompt
routing            # 调整 primary_channels / top_k / boost_weights
planning           # 调整 query_rewrite / planner_rules / max_variants
fallback           # 调整 failure_labels / retry_strategy / fallback behavior
graph_expansion    # 调整 evidence_expansion / max_expanded_blocks
stop_condition     # 调整 assessment thresholds / max_rounds / early_stop
```

| 架构图术语 | Dimension 名 | 对应 YAML path | patch_type |
|-----------|-------------|---------------|------------|
| prompt | `prompt` | `strategy.assessment` / `strategy.answer_generation` | `prompt_content_update` + `prompt_ref_update` |
| 路由 | `routing` | `strategy.retrieval_routing` | `retrieval_strategy_update` |
| 规划 | `planning` | `strategy.query_rewrite` | `planner_patch` |
| fallback | `fallback` | `strategy.assessment.failure_labels` | `fallback_update` |
| graph expansion | `graph_expansion` | `strategy.evidence_expansion` | `evidence_expansion_update` |
| 停止条件 | `stop_condition` | `strategy.assessment` stop-related fields | `stop_condition_update` |
