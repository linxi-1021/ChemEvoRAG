# §6 ChemRAG Skill Evolution — 设计文档

> **日期**: 2026-06-05  
> **版本**: 1.1  
> **状态**: Draft  
> **关联架构**: docs/EVOChemRAG_Technical_Architecture.md §6

---

## 1. 概述

Skill Evolution 是 EVOChemRAG 的受控自进化机制。系统通过「执行 → 评估 → 归因 → 变异 → 验证 → 写回」闭环，持续优化每个 intent 对应的 Skill 配置。

**设计目标：**

- 每个 intent 类型对应一个 Skill YAML（可进化、可审计、可回滚）
- 失败驱动策略修正，成功驱动模板优化
- 所有变异必须经过回归验证才能写回
- 先归因，再进化；coverage gap 只记录不进化

**当前基础：**

- 118 道评测题，5 种 intent，平均分 0.89
- 丰富的 trace 数据（eval_results.json + react_logs/）
- 已有 ReAct 多轮检索、6 通道检索、Evidence Expansion 等完整链路

---

## 2. 系统组件

```text
┌─────────────────────────────────────────────────────┐
│              Skill YAML (config/skills/)             │
│  identity | strategy | templates | cases | evolution │
└────────────────────────┬────────────────────────────┘
                         │ load
                         ▼
┌─────────────────────────────────────────────────────┐
│              Execution & Trace Collection            │
│  eval_questions.py → react_logs/ + eval_results.json │
└────────────────────────┬────────────────────────────┘
                         │ traces
                         ▼
┌─────────────────────────────────────────────────────┐
│              Failure Attribution (§4)                │
│  trace_report.json: outcome + failure_type + action  │
└──────────┬─────────────────────────────┬────────────┘
           │ system_failure              │ success
           ▼                             ▼
┌──────────────────────┐    ┌─────────────────────────┐
│  LLM Patch Generator │    │  Template Distillation   │
│  (§5.1–5.3)          │    │  (§5.4)                  │
└──────────┬───────────┘    └────────────┬────────────┘
           │ candidate_patch             │ candidate_template
           └──────────┬──────────────────┘
                      ▼
┌─────────────────────────────────────────────────────┐
│              Validation Pipeline (§5.5)              │
│  schema check → individual regression → composition  │
└────────────────────────┬────────────────────────────┘
                         │ accepted patches
                         ▼
┌─────────────────────────────────────────────────────┐
│              Write-back & Changelog (§6.2/§6.4)      │
│  skill YAML ← patch   prompt registry ← new variant │
└────────────────────────┬────────────────────────────┘
                         │ re-run eval
                         ▼
┌─────────────────────────────────────────────────────┐
│              Evolution Runner (§6)                   │
│  round loop: compare → keep/rollback/stop            │
└─────────────────────────────────────────────────────┘
```

---

## 3. Skill YAML Schema

### 3.1 存储位置

```text
config/skills/property_query.yaml
config/skills/reaction_comparison.yaml
config/skills/entity_lookup.yaml
config/skills/alias_resolution.yaml
config/skills/reaction_condition_query.yaml
```

### 3.2 Pydantic 校验规则

以下约束必须在 Skill YAML 加载时由 Pydantic 模型校验：

1. `name` 必须与文件名一致（去掉 `.yaml` 后缀）
2. `trigger.intent` 必须与 `name` 一致
3. 所有 `system_prompt_ref` 必须在 Prompt Registry 中存在
4. 所有 `dependencies.indexes` 必须在检索层中注册
5. `failure_to_patch_mapping` 的 `target_paths` 可以是 `mutable_paths` 的父路径，但 `target_path + allowed_fields` 展开后的完整字段路径必须落在 `mutable_paths` 允许范围内
6. `frozen_paths` 和 `mutable_paths` 不得有交集
7. `allowed_fields` 必须是 `target_path` 下真实存在的字段
8. 当 `status=frozen` 时，`evolution.enabled` 必须为 `false`
9. `evaluation.regression_cases_ref` 文件必须存在

### 3.3 完整 Schema（property_query 示例）

```yaml
schema_version: "1.0"
name: property_query
skill_version: "1.0.0"
status: stable  # draft | experimental | stable | deprecated | frozen
description: "Handles compound-level property queries such as yield, melting point, boiling point, mass, and purity."

lifecycle:
  created_at: "2026-06-05"
  updated_at: "2026-06-05"
  owner: "EVOChemRAG"

trigger:
  intent: property_query
  confidence_threshold: 0.7
  aliases: [property_lookup, compound_property_query]
  required_signals: [property_term, compound_or_reaction_reference]
  negative_intents: [mechanism_query, synthesis_route_query]

scope:
  supported_properties:
    - isolated_yield
    - melting_point
    - boiling_point
    - mass
    - purity
  supported_units: ["%", "C", "mg", "g"]
  unsupported_properties: [biological_activity, computational_energy]
  domain_constraints: ["experimental chemistry literature", "compound-level property extraction"]

dependencies:
  prompts:
    - EVIDENCE_ASSESSMENT_SYSTEM
    - ANSWER_GENERATION_SYSTEM
  indexes:
    - entity_search
    - lexical_search
    - reaction_event_search
  validators:
    - numeric_extraction_validator
    - citation_validator
  core_modules:
    - ElementKG

interface:
  input_schema:
    required: [query, intent]
    optional: [resolved_entities, document_context, user_constraints]
  output_schema:
    required: [answer, evidence_items, confidence]
    optional: [extracted_values, uncertainty_notes, failure_reason]
  normalized_output:
    fields:
      compound_id: string
      property_name: string
      value: string
      unit: string
      evidence_id: string
      confidence: float

strategy:
  retrieval_routing:
    primary_channels: [entity_search, lexical_search, reaction_event_search]
    top_k:
      entity_search: 8
      lexical_search: 10
      reaction_event_search: 12
      final_rerank: 15
    boost_weights:
      table_block:
        weight: 5.0
        rationale: "Property values often appear in tables."
      numeric_match:
        weight: 1.5
        rationale: "Property queries require numeric evidence."
      molecule_evidence:
        weight: 2.0
        rationale: "Compound-level evidence should be prioritized."

  query_rewrite:
    enabled: true
    max_variants: 3
    diversity_threshold: 0.8
    preserve_original_query: true
    fallback_terms: [experimental, characterization, physical properties]

  evidence_expansion:
    enabled: true
    expand_from_block_neighbors: true
    expand_from_molecule_mentions: true
    max_expanded_blocks: 5

  assessment:
    system_prompt_ref: EVIDENCE_ASSESSMENT_SYSTEM
    require_numeric_extraction: true
    require_citation_check: true
    failure_labels:
      - insufficient_evidence
      - numeric_conflict
      - compound_mismatch
      - assessment_false_negative

  answer_generation:
    system_prompt_ref: ANSWER_GENERATION_SYSTEM
    max_evidence_items: 15
    require_grounded_answer: true
    uncertainty_policy: "state_uncertainty_when_evidence_is_partial"

templates:
  - name: yield_extraction
    description: "Extract isolated yield from experimental sections."
    applicable_when:
      property_name: [isolated_yield, yield]
      evidence_type: [experimental_block, reaction_table]
      required_signals: [compound_or_entry_id, numeric_percent_value]
    not_applicable_when:
      - "query asks for reaction condition without specifying yield"
      - "evidence has multiple conflicting yields without compound mapping"
    slots: [compound_id, yield_value, unit, evidence_id]
    examples:
      - question: "What isolated yield was reported for 2a?"
        evidence:
          text: "2a (1.2g, 95%)"
          evidence_type: experimental_block
        expected_answer:
          property: isolated_yield
          value: "95"
          unit: "%"
          evidence_required: true

  - name: melting_point_extraction
    description: "Extract melting point or boiling point from characterization data."
    applicable_when:
      property_name: [melting_point, boiling_point]
      evidence_type: [characterization_block, property_table]

evaluation:
  metrics:
    - numeric_exact_match
    - evidence_recall
    - citation_accuracy
    - answer_groundedness
  pass_criteria:
    numeric_exact_match: 0.9
    citation_accuracy: 0.95
    answer_groundedness: 0.95
  regression_cases_ref: "eval/property_query_regression.yaml"

evolution:
  enabled: true
  mutable_paths:
    - strategy.retrieval_routing.primary_channels
    - strategy.retrieval_routing.top_k
    - strategy.retrieval_routing.boost_weights
    - strategy.query_rewrite.max_variants
    - strategy.query_rewrite.diversity_threshold
    - strategy.query_rewrite.fallback_terms
    - strategy.evidence_expansion
    - strategy.assessment
    - strategy.answer_generation
    - templates
  frozen_paths:
    - name
    - trigger.intent
    - interface.input_schema
    - dependencies.core_modules
  mutation_policy:
    allow_prompt_mutation: true
    allow_policy_mutation: true
    allow_template_addition: true
    allow_template_merge: true
    require_regression_test: true
    require_human_approval_for: [interface, dependencies, trigger]
  failure_to_patch_mapping:
    entity_miss:
      target_paths: [strategy.query_rewrite, strategy.retrieval_routing]
      allowed_operations: [update]
      allowed_fields: [fallback_terms, diversity_threshold, boost_weights]
    alias_miss:
      target_paths: [strategy.query_rewrite, strategy.retrieval_routing]
      allowed_operations: [update]
      allowed_fields: [fallback_terms, primary_channels]
    local_id_miss:
      target_paths: [strategy.query_rewrite]
      allowed_operations: [update]
      allowed_fields: [fallback_terms, max_variants]
    routing_error:
      target_paths: [strategy.retrieval_routing]
      allowed_operations: [update]
      allowed_fields: [primary_channels, top_k, boost_weights]
    evidence_expansion_error:
      target_paths: [strategy.evidence_expansion, strategy.retrieval_routing]
      allowed_operations: [update]
      allowed_fields: [expand_from_block_neighbors, max_expanded_blocks, top_k]
    planner_error:
      target_paths: [strategy.query_rewrite]
      allowed_operations: [update]
      allowed_fields: [diversity_threshold]
    assessment_false_negative:
      target_paths: [strategy.assessment]
      allowed_operations: [update]
      allowed_fields: [system_prompt_ref, require_numeric_extraction, failure_labels]
    table_reasoning_error:
      target_paths: [strategy.assessment, strategy.answer_generation]
      allowed_operations: [update]
      allowed_fields: [system_prompt_ref, max_evidence_items]
    generation_error:
      target_paths: [strategy.answer_generation]
      allowed_operations: [update]
      allowed_fields: [system_prompt_ref, max_evidence_items]
    unknown_failure:
      target_paths: []
      allowed_operations: []

safety:
  elementkg_modification: forbidden
  max_patches_per_round: 2
  consecutive_fail_limit: 3  # 同一 target_path 连续 3 轮被 patch 且验证失败后暂停该路径进化
  patch_confidence_threshold: 0.75
  uncertain_patch_queue: true
  schema_validation:
    before_patch: true
    after_patch: true
    reject_on_schema_error: true
  semantic_invariants:
    - skill_name_matches_filename
    - trigger_intent_matches_skill_name
    - all_prompt_refs_exist
    - all_mutable_paths_exist
    - no_frozen_path_modified
  risk_policy:
    low:
      allow_auto_promote_after_regression: true
    medium:
      require_regression_pass: true
      require_composition_validation: true
    high:
      require_human_approval: true
  rollback:
    snapshot_before_apply: true
    rollback_on_score_drop: true
    rollback_scope: "last_promoted_patch_batch"

cases:
  representative:
    - query: "What melting point is reported for compound 2b?"
      intent: property_query
      expected_answer: "mp 179.9-181.2 C"
      outcome: success
  external_case_refs:
    success_cases: "cases/property_query_success.yaml"
    failure_cases: "cases/property_query_failures.yaml"
    regression_cases: "eval/property_query_regression.yaml"

provenance:
  created_from: [manual_seed]
  last_modified_by: skill_evolution_pipeline
  last_modified_reason: ""

changelog:
  - version: "1.0.0"
    round: "initial"
    change_type: "seed"
    reason: "Initial skill definition based on architecture doc."
    affected_paths: []
```

### 3.4 Prompt Registry Schema

Prompt 变体不直接存入 Skill YAML，而是独立存储。每个 prompt 文件有以下结构：

```yaml
prompt_ref: EVIDENCE_ASSESSMENT_SYSTEM_V2
version: "2.0.0"
status: experimental  # experimental | stable | deprecated
created_from: EVIDENCE_ASSESSMENT_SYSTEM
created_by: skill_evolution_pipeline
created_at: "2026-06-05"
compatible_skills:
  - property_query
  - reaction_condition_query
content: |
  ...（完整 prompt 文本）...
diff_summary: "Add numeric evidence sufficiency rule."
validation:
  passed_regression_runs:
    - run_2026_06_05_1530
hash: "sha256:..."
```

存储位置：

```text
config/prompts/
  evidence_assessment_system.yaml        # 版本 1
  evidence_assessment_system_v2.yaml     # Skill Evolution 生成的变体
  answer_generation_system.yaml
```

Skill YAML 通过 `system_prompt_ref` 引用。回滚时只需恢复 ref 或禁用变体。

---

## 4. Failure Attribution

### 4.1 枚举定义

```python
class OutcomeType(str, Enum):
    SUCCESS = "success"                 # judge_score >= 0.8
    PARTIAL_SUCCESS = "partial_success" # 0.3 <= score < 0.8
    SYSTEM_FAILURE = "system_failure"   # score < 0.3 且非 coverage_gap
    COVERAGE_GAP = "coverage_gap"       # 答案不在 corpus 中（经 oracle 验证）

class FailureType(str, Enum):
    ENTITY_MISS = "entity_miss"
    ALIAS_MISS = "alias_miss"
    LOCAL_ID_MISS = "local_id_miss"
    ROUTING_ERROR = "routing_error"
    EVIDENCE_EXPANSION_ERROR = "evidence_expansion_error"
    PLANNER_ERROR = "planner_error"
    ASSESSMENT_FALSE_NEG = "assessment_false_negative"
    TABLE_EXTRACTION_ERROR = "table_extraction_error"   # PDF parser / table extraction 问题 → 只记录
    TABLE_REASONING_ERROR = "table_reasoning_error"     # 表格正确但 LLM 理解错误 → 可进化
    GENERATION_ERROR = "generation_error"
    UNKNOWN_FAILURE = "unknown_failure"
```

`TABLE_EXTRACTION_ERROR` 属于 Builder/Parser 问题，只记录到 data quality report，不进入 Skill YAML patch。`TABLE_REASONING_ERROR` 才允许修改 assessment 或 answer_generation prompt。

### 4.2 Outcome 处理策略

| outcome_type | 处理方式 |
|---|---|
| success | 检查是否可蒸馏为模板 |
| partial_success | 运行轻量归因（见 §4.8） |
| system_failure | 运行完整归因，生成 candidate_patch |
| coverage_gap | 只记录，不进入进化 |

### 4.3 归因优先级（规则层）

```text
1. coverage_gap_check（多通道 oracle evidence 验证 + gold_evidence_ids 比对）
2. entity_miss
3. alias_miss
4. local_id_miss
5. routing_error
6. evidence_expansion_error
7. planner_error
8. table_extraction_error（只记录，不进进化）
9. table_reasoning_error
10. assessment_false_negative
11. generation_error
12. → fallback to LLM attribution
```

上游错误优先归因，避免下游污染。

### 4.4 多因果记录

每次失败记录 `primary_failure_type` + `contributing_failure_types`：

```json
{
  "primary_failure_type": "local_id_miss",
  "contributing_failure_types": ["routing_error", "generation_error"]
}
```

Patch 生成只根据 primary_failure_type 触发。

### 4.5 LLM 归因输出格式

```json
{
  "failure_type": "local_id_miss",
  "confidence": 0.82,
  "explanation": "The system failed to map compound 2b to the correct local identifier.",
  "supporting_trace_signals": [
    "question mentions 2b",
    "retrieved evidence contains 2a but not 2b",
    "answer cites block for 2a"
  ]
}
```

confidence >= 0.75 → 允许 patch；0.5-0.75 → uncertain queue；< 0.5 → manual review。

### 4.6 coverage_gap 严格判定

coverage_gap 必须同时满足以下全部条件：

1. judge_score < 0.3
2. retrieved evidence 中无答案
3. 多通道 oracle evidence search 中也无答案
4. 对 benchmark 样本，gold_evidence_ids 也不在 retrieved evidence 中

oracle evidence search 包括以下通道：

```yaml
oracle_search:
  channels:
    - lexical_full_corpus     # 全语料库关键词检索
    - dense_full_corpus       # 全语料库向量检索
    - entity_alias_search     # 实体别名检索
    - local_id_search         # 局部编号映射检索
    - fact_card_search        # FactCard 全文检索
    - reaction_event_search   # ReactionEventCard 检索
    - structure_search        # 结构相似检索
  require_gold_evidence_check: true
  allow_manual_override: true
```

### 4.7 trace_report.json 结构

```json
{
  "run_id": "run_2026-06-05_1530",
  "skill_snapshot": {
    "skill_dir": "config/skills",
    "schema_version": "1.0",
    "git_commit": "abc123"
  },
  "environment": {
    "model_name": "gpt-5.4",
    "model_version": "gpt-5.4-2026-05",
    "embedding_model": "all-MiniLM-L6-v2",
    "temperature": 0.1,
    "top_p": 1.0,
    "retriever_version": "6ch_v1",
    "index_version": "2026-06-05",
    "prompt_registry_version": "1.0",
    "corpus_snapshot_id": "23_papers_2026-06-05",
    "eval_dataset_version": "v1_118qs",
    "random_seed": 42
  },
  "total_questions": 118,
  "average_score": 0.89,
  "by_intent": {
    "property_query": {
      "average_score": 0.86,
      "success": 72,
      "partial_success": 8,
      "system_failure": 5,
      "coverage_gap": 2
    }
  },
  "failures": [
    {
      "question_id": "q5_1",
      "intent": "property_query",
      "skill_name": "property_query",
      "skill_version": "1.0.0",
      "score": 0.0,
      "outcome_type": "system_failure",
      "primary_failure_type": "assessment_false_negative",
      "contributing_failure_types": [],
      "attribution_source": "rule",
      "attribution_confidence": 0.95,
      "attribution_explanation": "Evidence contained the numeric yield, but assessment judged insufficient.",
      "evidence_contained_answer": true,
      "corpus_contained_answer": true,
      "gold_answer": "91%",
      "predicted_answer": "insufficient evidence",
      "retrieval_path": ["entity_search", "lexical_search"],
      "retrieved_evidence_ids": ["block_1_0122", "block_1_0036"],
      "gold_evidence_ids": ["block_1_0122"],
      "cited_evidence_ids": [],
      "react_rounds_used": 3,
      "stop_reason": "assessment_insufficient",
      "proposed_patch_targets": ["strategy.assessment"],
      "evolution_action": "generate_patch_candidate"
    }
  ],
  "coverage_gaps": [
    {
      "question_id": "q17_4",
      "intent": "property_query",
      "score": 0.0,
      "outcome_type": "coverage_gap",
      "gap_reason": "target property not present in retrieved evidence or corpus",
      "coverage_gap_verified_by": "oracle_search",
      "evolution_action": "record_only"
    }
  ],
  "partial_successes": [
    {
      "question_id": "q2_3",
      "intent": "property_query",
      "score": 0.62,
      "outcome_type": "partial_success",
      "partial_success_reason": "correct property value but weak citation",
      "primary_failure_type": "generation_error",
      "evolution_action": "minor_patch_candidate"
    }
  ],
  "success_patterns": [
    {
      "intent": "property_query",
      "pattern": "yield extraction from table",
      "frequency": 18,
      "avg_score": 0.95,
      "supporting_question_ids": ["q1_2", "q1_8", "q2_4"],
      "candidate_template_action": "distill_or_refine_template"
    }
  ]
}
```

### 4.8 Partial Success 处理策略

partial_success 不进入完整失败归因，但运行轻量归因判断行动方向：

```yaml
partial_success_policy:
  score_range: [0.3, 0.8)
  action_rules:
    correct_answer_wrong_citation:
      action: minor_patch_candidate
      target: strategy.answer_generation
    correct_evidence_incomplete_answer:
      action: template_refinement_candidate
    correct_value_wrong_unit:
      action: validator_patch_candidate
    correct_workflow_low_confidence:
      action: template_distillation_candidate
    ambiguous_judge_reason:
      action: uncertain_queue
```

---

## 5. Recipe Distillation & LLM Mutation

### 5.1 Patch Schema

```json
{
  "patch_id": "property_query_round1_patch1",
  "skill_name": "property_query",
  "skill_version": "1.0.0",
  "source_failure_ids": ["q5_1", "q8_3"],
  "primary_failure_type": "assessment_false_negative",
  "target_path": "strategy.assessment",
  "operation": "update",
  "current_value": {
    "system_prompt_ref": "EVIDENCE_ASSESSMENT_SYSTEM",
    "require_numeric_extraction": true
  },
  "proposed_value": {
    "system_prompt_ref": "EVIDENCE_ASSESSMENT_SYSTEM_V2",
    "require_numeric_extraction": true,
    "numeric_presence_override": true
  },
  "prompt_artifacts": [
    {
      "prompt_ref": "EVIDENCE_ASSESSMENT_SYSTEM_V2",
      "created_from": "EVIDENCE_ASSESSMENT_SYSTEM",
      "diff_summary": "Add instruction to treat explicitly present numeric values as sufficient.",
      "registry_path": "config/prompts/evidence_assessment_system_v2.yaml"
    }
  ],
  "rationale": "Evidence contained the target numeric value, but assessment judged insufficient.",
  "expected_improvement": "Reduce assessment false negatives for numeric property extraction.",
  "risk_level": "medium",
  "confidence": 0.82,
  "status": "candidate"
}
```

### 5.2 Patch 操作语义

| operation | 语义 | 约束 |
|---|---|---|
| add | 在 list 或 dict 中新增条目 | key 不得已存在 |
| update | 修改已有字段 | 字段必须在 allowed_fields 中 |
| delete | 仅允许 templates 或 fallback_terms | 禁止删除 required 字段 |
| merge | 合并模板 | 必须保留 not_applicable_when |
| replace | 整体替换（高风险） | 默认需人工审批 |

### 5.3 Patch Conflict Resolution

- 同一轮中，同一完整字段路径（target_path + allowed_field）只允许一个 patch 被选中
- 如果 target_path 相同但 allowed_field 不同（不冲突），可以进入 composition validation
- 多个 patch 修改同一字段时，按 patch_score 排序，只保留最高分
- 两个 patch 来自不同 failure cluster 但修改同一 `prompt_ref` → 进入 manual_review
- composition validation 前必须执行 conflict detection

### 5.4 Patch 生命周期

```text
generated → schema_validated → regression_validated → promoted → applied → monitored
                                  ↓
                              rejected → manual_review
                                  ↓
                              rolled_back
```

### 5.5 Failure Cluster 生成 Patch

不对每条失败单独生成 patch，而是先按 `skill_name + primary_failure_type + proposed_patch_targets` 聚类：

```text
failure_records
  ↓
cluster by skill / failure_type / target_path / trace pattern
  ↓
generate 1-3 candidate patches per cluster
  ↓
validate individually
  ↓
select top patches
```

### 5.6 Template Distillation

触发条件：同一 intent + 相似 query structure + 相似 evidence type + 相似 retrieval path，累计 >= 3 次 success（不要求连续出现）。

模板结构包含 `applicable_when`、`not_applicable_when`、`slots`、`examples`（见 §3.3 templates 部分）。

模板蒸馏作为 patch 进入统一验证流程。

### 5.7 Validation Pipeline

**5.7.1 Individual Validation**

```text
candidate_patch
  → schema + semantic invariant validation
  → apply to in-memory skill YAML
  → run regression_set
  → check intent_regression: score >= 0.80, max_score_drop <= 0.01
  → check targeted_failures: improvement >= +0.05 (absolute delta)
  → check high_severity: no new high-severity failure
```

**5.7.2 Composition Validation**

```text
all selected patches (max 2 per round)
  → conflict detection（同一 target_path 不冲突）
  → apply together to in-memory skill YAML
  → run regression_set + global smoke test
  → check: intent_regression max_score_drop <= 0.01, max_failed_cases_increase <= 1
  → check: global_guardrail max_score_drop <= 0.005
  → check: no new high_severity_failure
```

**5.7.3 Validation Metrics（三级）**

```yaml
intent_regression:
  min_score: 0.80
  max_score_drop: 0.01
  max_failed_cases_increase: 1

targeted_failures:
  min_improvement: 0.05  # absolute score delta
  require_any_improvement: true

global_guardrail:
  sample_set: "eval/global_smoke.yaml"
  max_score_drop: 0.005
```

### 5.8 Patch Ranking

```text
patch_score = targeted_improvement
            - regression_drop_penalty
            - risk_penalty
            + failure_frequency_bonus
            + attribution_confidence_bonus
            - modification_scope_penalty
```

在 `max_patches_per_round` 约束下选择最优 patch。

---

## 6. Evolution Runner

### 6.1 CLI

```bash
python scripts/evolve.py                    # 运行完整进化
python scripts/evolve.py --rounds 3         # 指定轮次
python scripts/evolve.py --dry-run          # 不写回
python scripts/evolve.py --analyze-only     # 只归因，不生成 patch
```

### 6.2 Round Loop

```text
Round N:
  1. Load skill YAML + prompt registry
  2. Record environment snapshot (model, embedding, index, dataset version, seed)
  3. Run eval_questions.py --react (collect trace)
  4. Generate trace_report.json (failure attribution)
  5. Separate outcomes: success / partial_success / system_failure / coverage_gap
  6. Cluster failures by skill + failure_type + target_path
  7. Generate candidate patches per cluster
  8. Template distillation for success patterns
  9. Schema + semantic invariant validation
  10. Individual regression validation
  11. Conflict detection among candidates
  12. Rank and select patches (max 2 per round)
  13. Composition validation
  14. Promote accepted patches
  15. Write back skill YAML + prompt registry
  16. Record changelog, snapshots, rollback plan
  17. Re-run eval with updated skills
  18. Compare: keep / rollback / manual_review
  19. Stop if: max_rounds reached OR no improvement for 2 consecutive rounds
```

### 6.3 Rollback Policy

```yaml
rollback_policy:
  metric: average_score
  min_drop_to_rollback: 0.01
  compare_against: previous_round
  checks:
    - global_average_score_stable  # 总分不下降超过 0.01
    - target_intent_no_significant_regression  # 目标 intent 回归集 score 不下降超过 0.01
    - targeted_failures_improved_or_stable  # 目标失败题不恶化
    - no_new_high_severity_failures  # 不出现新的高严重度失败
```

分数下降但目标失败题有提升 → 人工审查（不自动回滚）。

### 6.4 评测数据集划分

```yaml
datasets:
  evolution_train:
    source: data/eval/all_questions.json
    used_for: failure attribution and patch generation
  intent_regression:
    source: eval/{intent}_regression.yaml (每个 intent 的回归题集)
    used_for: patch individual and composition validation
  global_smoke:
    source: eval/global_smoke.yaml (跨 intent 的轻量抽样集)
    used_for: cross-intent guardrail in composition validation
  heldout_test:
    source: eval/heldout_test.yaml
    used_for: final report only
    never_used_for_patch_generation: true
```

`heldout_test` 严禁用于 patch 生成或验证，防止过拟合。

### 6.5 Directory Structure

```text
config/
  skills/
    property_query.yaml
    reaction_comparison.yaml
    entity_lookup.yaml
    alias_resolution.yaml
    reaction_condition_query.yaml
  prompts/
    evidence_assessment_system.yaml
    answer_generation_system.yaml

data/
  evolution/
    runs/
      run_2026-06-05_1530/
        trace_report.json
        failure_clusters.json
        candidate_patches.json
        validation_results.json
        selected_patches.json
        applied_patches.json
        rejected_patches.json
        rollback_plan.json
        skill_snapshot_before/
        skill_snapshot_after/
        prompt_snapshot_before/
        prompt_snapshot_after/
        evolution_log.txt
    patches/
      accepted/
      rejected/
      rolled_back/
    templates/
      distilled_candidates/
      accepted/
    reviews/
      uncertain_patch_queue.jsonl
```

---

## 7. 实施计划（按优先级）

### Phase 1: Skill YAML + Seed（MVP，必须完成）

- 定义 Skill YAML Pydantic 模型（含 §3.2 所有校验规则）
- 为 5 个 intent 创建 seed skill YAML（基于当前 prompts 和 eval 结果）
- 创建 Prompt Registry 结构（config/prompts/）及 Pydantic 模型
- 修改 eval_questions.py 支持读取 skill 配置

### Phase 2: Failure Attribution（MVP，必须完成）

- 实现 OutcomeType / FailureType 枚举
- 实现规则层归因（优先级顺序 + 多因果）
- 实现 LLM 层归因（confidence + supporting_trace_signals）
- 实现 trace_report.json 生成器（含 environment 字段）
- 实现 partial_success 轻量归因
- 对现有 eval_results.json + react_logs/ 运行归因，验证输出

### Phase 3: Patch Generator + Validation（原型，核心功能）

- 实现 Patch schema（Pydantic 模型 + 生命周期）
- 实现 failure cluster 聚类
- 实现 LLM patch generator（failure_type + mutable_paths + allowed_fields 约束）
- 实现 template distillation
- 实现 validation pipeline（schema + individual regression + conflict detection + composition）
- 实现 patch ranking / selection

### Phase 4: Evolution Runner（增强，完成自动闭环）

- 实现 evolve.py 主脚本（round loop + environment snapshot）
- 实现 rollback 机制（多指标判断，不只看平均分）
- 实现 dry-run / analyze-only 模式
- 端到端测试：运行 2 轮进化，验证分数提升

---

## 8. 安全约束总结

| 约束 | 说明 |
|---|--- |
| ElementKG 不可修改 | evolution 过程不触及核心知识图谱 |
| frozen_paths 严格执行 | name, intent, interface, core_modules 不可变异 |
| mutable_paths 与 failure_to_patch_mapping 对齐 | target_paths 必须属于 mutable_paths |
| field-level 白名单 | 每种 failure_type 只允许修改 allowed_fields |
| table_extraction_error 不进化 | 属于 parser 问题，只记录到 data quality report |
| max_patches_per_round: 2 | 避免过度变异 |
| patch_confidence >= 0.75 | 低于此值进入人工审查 |
| 所有 patch 必须通过回归验证 | risk_policy 不绕过 regression |
| composition validation | 多 patch 组合后必须重新验证 |
| conflict detection | 同一轮同一 target_path 只允许一个 patch |
| schema + semantic invariant check | 变更前后必须校验 |
| risk policy | 高风险 patch 需人工审批；低风险自动 promote 仍需先通过 regression |
| rollback checkpoint | 写回前保存快照，下降时可回滚 |
| consecutive_fail_limit: 3 | 同一 target_path 连续 3 轮 patch 失败后暂停 |
| heldout_test 隔离 | 最终报告数据不参与 patch 生成或验证 |
| environment snapshot | 每轮记录 model/embedding/index/version 保证可复现 |
