# Skill Evolution 完整设计文档

## 1. 目标

实现真正的 Skill Evolution，不只是 Prompt Evolution。系统能自动进化：

- **Prompt**（evidence assessment, answer generation）
- **Retrieval Routing**（primary_channels, top_k, boost_weights）
- **Query Rewrite**（fallback_terms, diversity_threshold, max_variants）
- **Evidence Expansion**（max_expanded_blocks, expand_from_block_neighbors）
- **Answer Generation**（prompt_ref, max_evidence_items）

## 2. 总体原则

```
规则粗判 → analysis_cluster → LLM failure analysis → FailureAnalysisReport
→ LLM 基于 V1 生成完整 V2 prompt → semantic validation
→ targeted regression → global regression → selected patches → apply / rollback
```

| 层级 | 作用 | 是否最终决策 |
|------|------|------------|
| 规则 | 低成本粗分类、候选聚类、发现明显问题 | 否 |
| LLM | 分析 root cause、修正规则、生成 prompt 改进 | 否 |
| Regression | 验证 patch 是否真的提升效果 | 是 |

## 3. Stage 0: Trace Field Audit

检查每个 failure 样本的信息完整性。

**必须字段：** question_id, question, gold_answer, predicted_answer, score, intent

**强烈需要：** retrieved_evidence_ids, react_log, evidence_contained_answer, assessment_result, assessment_reason

**输出：** TraceFieldAudit（field_status, missing_fields, trace_quality）

## 4. Stage 1: Rule-based Candidate Attribution

规则层输出候选结论，不是最终判断。

### Analysis Cluster（跨 skill）

```python
analysis_key = (failure_type, normalized_target_path, prompt_role)
```

允许相同 failure_type 不同 skill 的 failures 聚在一起分析。

### Candidate Attribution 输出

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

## 5. Stage 2: LLM Failure Analysis

### 输入格式

```json
{
  "cluster_info": {
    "failure_type": "assessment_false_negative",
    "normalized_target_path": "strategy.assessment",
    "prompt_role": "evidence_assessment",
    "affected_skills": ["reaction_condition_query", "reaction_comparison"]
  },
  "representative_failures": [
    {
      "question_id": "q3",
      "question": "...",
      "gold_answer": "...",
      "model_answer": "...",
      "score": 0.0,
      "outcome_type": "system_failure",
      "corpus_contained_answer": true,
      "evidence_contained_answer": true,
      "gold_evidence_ids": ["table_3_row_2"],
      "retrieved_evidence_ids": ["block_12", "table_3_row_2"],
      "cited_evidence_ids": [],
      "missing_slots": ["temperature", "yield"],
      "conflicts": [],
      "retrieval_rounds": [
        {
          "round": 1,
          "query": "...",
          "refined_query": "...",
          "retrieval_actions": ["lexical_search", "entity_search"],
          "retrieved_evidence": [
            {
              "evidence_id": "table_3_row_2",
              "evidence_type": "table_block",
              "channel": "lexical_search",
              "score": 0.82,
              "contains_gold_answer": true,
              "used_in_final_answer": false
            }
          ],
          "new_evidence_found": true,
          "skipped_due_to_similarity": false,
          "similarity_to_previous_query": 0.32
        }
      ],
      "assessment_result": "sufficient=false",
      "assessment_reason": "..."
    }
  ],
  "current_skill_configs": {
    "reaction_condition_query": {
      "skill_version": "1.0.0",
      "strategy": { ... },
      "evolution": { "mutable_paths": [...], "frozen_paths": [...], "failure_to_patch_mapping": {} }
    },
    "reaction_comparison": {
      "skill_version": "1.0.0",
      "strategy": { ... },
      "evolution": { "mutable_paths": [...], "frozen_paths": [...], "failure_to_patch_mapping": {} }
    },
    "entity_lookup": {
      "skill_version": "1.0.0",
      "strategy": { ... },
      "evolution": { "mutable_paths": [...], "frozen_paths": [...], "failure_to_patch_mapping": {} }
    }
  },
  "current_prompts": {
    "evidence_assessment": {
      "prompt_ref": "EVIDENCE_ASSESSMENT_SYSTEM",
      "content": "..."
    },
    "answer_generation": {
      "prompt_ref": "ANSWER_GENERATION_SYSTEM",
      "content": "..."
    }
  }
}
```

### 输出格式（FailureAnalysisReport）

```json
{
  "cluster_id": "assessment_false_negative__strategy.assessment__evidence_assessment",
  "primary_failure_type": "assessment_false_negative",
  "contributing_failure_types": ["evidence_expansion_error"],
  "confidence": 0.82,
  "coverage_gap": false,
  "explanation": "...",
  "supporting_trace_signals": [
    "gold answer appears in retrieved table_block",
    "assessment_result is sufficient=false"
  ],
  "common_root_causes": [
    {
      "name": "fragmented_evidence_not_combined",
      "description": "...",
      "supporting_failure_ids": ["q3", "q_batch2_3"],
      "confidence": 0.8
    }
  ],
  "subclusters": [
    {
      "name": "entity_specific_assessment_mismatch",
      "failure_ids": ["q_batch2_2"],
      "specific_issue": "...",
      "recommended_rule": "..."
    }
  ],
  "proposed_patch_targets": [
    "strategy.assessment",
    "strategy.evidence_expansion"
  ],
  "recommended_changes": {
    "prompt": {
      "should_change": true,
      "priority": "primary",
      "prompt_role": "evidence_assessment",
      "base_prompt_ref": "EVIDENCE_ASSESSMENT_SYSTEM",
      "change_summary": ["Add fragmented evidence aggregation rule"],
      "risk_level": "medium",
      "confidence": 0.82
    },
    "retrieval_routing": {
      "should_change": true,
      "priority": "secondary",
      "changes": [
        {
          "target_path": "strategy.retrieval_routing.boost_weights.table_block.weight",
          "operation": "update",
          "current_value": 4.0,
          "proposed_value": 6.0,
          "rationale": "...",
          "expected_improvement": "...",
          "source_failure_ids": ["q3"],
          "confidence": 0.76
        }
      ],
      "risk_level": "low"
    },
    "query_rewrite": {
      "should_change": false,
      "changes": [],
      "risk_level": "low"
    },
    "evidence_expansion": {
      "should_change": true,
      "priority": "secondary",
      "changes": [
        {
          "target_path": "strategy.evidence_expansion.max_expanded_blocks",
          "operation": "update",
          "current_value": 5,
          "proposed_value": 8,
          "rationale": "...",
          "expected_improvement": "...",
          "source_failure_ids": ["q3"],
          "confidence": 0.73
        }
      ],
      "risk_level": "low"
    },
    "answer_generation": {
      "should_change": false,
      "changes": [],
      "risk_level": "low"
    }
  },
  "should_generate_patch": true,
  "risk_level": "medium"
}
```

## 6. Stage 3: LLM Prompt Generation

当 `recommended_changes.prompt.should_change == true` 时，调用第二次 LLM。

### 输入

- FailureAnalysisReport
- V1 prompt content
- change_summary

### 输出（PromptArtifact）

```json
{
  "prompt_artifact": {
    "prompt_ref": "EVIDENCE_ASSESSMENT_SYSTEM_V2",
    "base_prompt_ref": "EVIDENCE_ASSESSMENT_SYSTEM",
    "prompt_role": "evidence_assessment",
    "version": "2.0",
    "prompt_filename": "evidence_assessment_system_v2.yaml",
    "content": "...完整 V2 prompt...",
    "change_summary": ["Add fragmented evidence aggregation rule"],
    "preserved_constraints": [
      "Evidence must support the answer",
      "Do not infer missing facts"
    ],
    "targeted_failure_ids": ["q3", "q_batch2_3", "q_batch2_2"],
    "validation_requirements": [
      "prompt_ref_unique",
      "content_non_empty",
      "grounding_constraints_preserved",
      "targeted_regression_improves",
      "global_regression_not_degraded"
    ]
  }
}
```

## 7. Stage 4: Patch 生成

### 7.1 Failure → Patch 决策规则

```
corpus_contained_answer = false
  → 不生成 skill patch，标记为 corpus_gap

corpus_contained_answer = true 且 evidence_contained_answer = false
  → 优先生成 retrieval_routing / query_rewrite / evidence_expansion patch

evidence_contained_answer = true 且 assessment_result = insufficient
  → 优先生成 evidence_assessment prompt patch

evidence_contained_answer = true 且 assessment_result = sufficient 但答案错误
  → 优先生成 answer_generation patch
```

### 7.2 统一 Patch Schema

```json
{
  "patch_id": "reaction_condition_query_round1_patch1",
  "patch_type": "retrieval_strategy_update",
  "skill_name": "reaction_condition_query",
  "target_file": "config/skills/reaction_condition_query.yaml",
  "target_path": "strategy.retrieval_routing.boost_weights.table_block.weight",
  "operation": "update",
  "old_value": 4.0,
  "new_value": 6.0,
  "rationale": "Table evidence often contains the answer but is underweighted.",
  "expected_improvement": "Improve evidence recall for table-supported questions.",
  "source_failure_ids": ["q3"],
  "targeted_failure_ids": ["q3"],
  "risk_level": "low",
  "confidence": 0.76,
  "dependencies": [],
  "validation_status": "pending",
  "prompt_artifact": null
}
```

prompt_ref_update patch 必须依赖 prompt_content_update：

```json
{
  "patch_id": "reaction_condition_query_prompt_ref_v2",
  "patch_type": "prompt_ref_update",
  "skill_name": "reaction_condition_query",
  "target_file": "config/skills/reaction_condition_query.yaml",
  "target_path": "strategy.assessment.system_prompt_ref",
  "operation": "update",
  "old_value": "EVIDENCE_ASSESSMENT_SYSTEM",
  "new_value": "EVIDENCE_ASSESSMENT_SYSTEM_V2",
  "rationale": "Point to V2 prompt with fragmented evidence aggregation rules.",
  "source_failure_ids": ["q3"],
  "targeted_failure_ids": ["q3"],
  "risk_level": "medium",
  "confidence": 0.82,
  "dependencies": ["evidence_assessment_system_v2_prompt_content"],
  "validation_status": "pending",
  "prompt_artifact": null
}
```

prompt_content_update patch 包含完整 prompt_artifact：

```json
{
  "patch_id": "reaction_condition_query_prompt_v2",
  "patch_type": "prompt_content_update",
  "skill_name": "reaction_condition_query",
  "target_file": "config/prompts/evidence_assessment_system_v2.yaml",
  "target_path": "content",
  "operation": "add",
  "old_value": null,
  "new_value": "...完整 V2 prompt...",
  "rationale": "Add fragmented evidence aggregation rule.",
  "expected_improvement": "Reduce false negatives in evidence assessment.",
  "source_failure_ids": ["q3", "q_batch2_3"],
  "targeted_failure_ids": ["q3", "q_batch2_3"],
  "risk_level": "medium",
  "confidence": 0.82,
  "dependencies": [],
  "validation_status": "pending",
  "prompt_artifact": {
    "prompt_ref": "EVIDENCE_ASSESSMENT_SYSTEM_V2",
    "base_prompt_ref": "EVIDENCE_ASSESSMENT_SYSTEM",
    "prompt_role": "evidence_assessment",
    "prompt_filename": "evidence_assessment_system_v2.yaml",
    "content": "...",
    "change_summary": ["Add fragmented evidence aggregation rule"],
    "preserved_constraints": ["Evidence must support the answer", "Do not infer missing facts"],
    "validation_requirements": ["prompt_ref_unique", "content_non_empty", "grounding_constraints_preserved"]
  }
}
```

### 7.3 Patch 类型

| patch_type | 说明 | target_path 示例 |
|-----------|------|-----------------|
| prompt_content_update | 新增/修改 prompt 文件内容 | `config/prompts/xxx.yaml` |
| prompt_ref_update | 修改 skill YAML 的 system_prompt_ref | `strategy.assessment.system_prompt_ref` |
| retrieval_strategy_update | 修改 retrieval_routing 参数 | `strategy.retrieval_routing.boost_weights.table_block.weight` |
| query_rewrite_update | 修改 query_rewrite 参数 | `strategy.query_rewrite.fallback_terms` |
| evidence_expansion_update | 修改 evidence_expansion 参数 | `strategy.evidence_expansion.max_expanded_blocks` |
| answer_generation_update | 修改 answer_generation 参数 | `strategy.answer_generation.max_evidence_items` |
| template_add | 添加新模板 | `templates` |

### 7.4 Patch 生成流程

```
recommended_changes.prompt.should_change == true
  → Stage 3 LLM 生成 V2 prompt
  → 生成 prompt_content_update patch（含 prompt_artifact）
  → 为每个 affected_skill 生成 prompt_ref_update patch

recommended_changes.retrieval_routing.should_change == true
  → 遍历 changes[]
  → 为每个 affected_skill 生成独立 retrieval_strategy_update patch

recommended_changes.query_rewrite.should_change == true
  → 遍历 changes[]
  → 为每个 affected_skill 生成独立 query_rewrite_update patch

recommended_changes.evidence_expansion.should_change == true
  → 遍历 changes[]
  → 为每个 affected_skill 生成独立 evidence_expansion_update patch

recommended_changes.answer_generation.should_change == true
  → 如果 prompt 改了：Stage 3 生成 V2 answer prompt
  → 生成 prompt_content_update patch（含 prompt_artifact）
  → 为每个 affected_skill 生成 prompt_ref_update patch（依赖 prompt_content_update）
  → 如果参数改了：遍历 changes[]，为每个 affected_skill 生成独立 answer_generation_update patch

prompt_ref_update 依赖规则：
  → prompt_ref_update.dependencies 必须包含对应 prompt_content_update 的 patch_id
  → 确保 prompt 文件先创建，skill YAML 再指向它
```

## 8. Stage 5: Semantic Validation

四层校验：

1. **schema_validate**：target_file 存在、target_path 存在、字段类型正确、值在有效范围内
2. **config_load_validate**：patch 应用后 YAML 可解析、Pydantic 模型可加载、不破坏现有字段
3. **prompt_registry_validate**：prompt_ref 存在（或 prompt_artifact 会创建它）、prompt content 非空、保留 grounding/citation/no-hallucination 约束
4. **runtime_smoke_test**：patch 应用后 skill 配置可加载、prompt 可解析、不触发 import 错误

检查清单：

1. target_file 存在性（按 operation 区分）：
   - operation = update / replace / merge：target_file 必须存在
   - operation = add：target_file 可以不存在，但 parent directory 必须存在，且 prompt_ref 不能与已有 prompt_ref 冲突
2. target_path 存在或允许新增
3. prompt_ref 存在（或 prompt_artifact 会创建它）
4. prompt content 非空
5. 保留了 V1 的关键安全约束
6. 有 targeted_failure_ids
7. 有 rationale
8. 字段值在有效范围内
9. 不修改 frozen_paths
10. 不修改 eval dataset
11. prompt 输出 schema 可被 parser 解析
12. 不删除 grounding / citation / no hallucination 约束

## 9. Stage 6-8: Regression

### 9.1 Targeted Regression 通过标准

- targeted_failure_ids 中至少 1 个样本分数提升
- targeted average score 必须提升
- 不允许 targeted 样本从正确变错误（score 从 >=0.8 降到 <0.3）
- hallucination 数量不增加
- eval 没有异常退出

### 9.2 Global Regression 通过标准

- global average score 不得下降超过 0.005（0.5%）
- previously passed cases（score >= 0.8）的通过率不得下降超过 1 个样本
- critical skills 不得出现新增 high-severity failure（score < 0.3）
- 无新增 RDKit / parsing / prompt registry 错误

### 9.3 Composition Validation 通过标准

- 单 patch 有效
- patch 组合后 global score 不低于单 patch 最优 score
- 如果组合退化，回退到单 patch 或重新排序组合
- 冲突 patch 保留收益最高、风险最低的

## 10. Stage 9: Apply + Snapshot + Rollback

### 10.1 Snapshot 结构

```
data/evolution/snapshots/<run_id>/
├── manifest.json          # patch 列表、应用顺序、时间戳
├── before/
│   ├── skills/            # 原始 skill YAML
│   └── prompts/           # 原始 prompt YAML
├── after/
│   ├── skills/            # patch 后的 skill YAML
│   └── prompts/           # patch 后的 prompt YAML
├── applied_patches.json   # 成功应用的 patch
└── rollback_manifest.json # 回滚清单
```

### 10.2 Apply 流程

1. 创建 snapshot（before）
2. 按 patch priority 排序应用
3. 每个 patch 应用后写入 applied_patches.json
4. 创建 snapshot（after）
5. 运行 full eval
6. 如果退化 → 自动 rollback
7. 写入 apply_report.json

### 10.3 Rollback 流程

1. 读取 rollback_manifest.json
2. 从 before/ 恢复 skills/ 和 prompts/
3. 删除本次新增的 prompt 文件
4. 验证恢复后的配置可加载
5. 写入 rollback_report.json

### 10.4 默认行为

- 默认 `--validate-only`，不写回
- `--apply` 显式写回
- `--skip-regression` 只能用于调试，禁止与 `--apply` 同时使用

## 11. 文件结构

```
src/skill_evolution/
├── __init__.py
├── attribution.py      # Stage 0 + Stage 1
├── llm_analysis.py     # Stage 2 + Stage 3 (LLM 调用)
├── patch.py            # Stage 4 (Patch 生成)
├── validation.py       # Stage 5 (Semantic Validation)
├── regression.py       # Stage 6-8 (Regression)
├── runtime_validation.py  # 沙箱验证
├── apply.py            # Stage 9 (Apply)
├── rollback.py         # Stage 9 (Rollback)
```

## 12. CLI

```bash
python scripts/evolve.py --analyze-only     # Stage 0 + 1
python scripts/evolve.py --dry-run          # Stage 0-5, 不 regression
python scripts/evolve.py --validate-only    # Stage 0-8, 不写回
python scripts/evolve.py --apply            # 全流程
```
