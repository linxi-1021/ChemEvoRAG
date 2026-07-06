# ReAct 多轮检索设计文档

> **版本**: 1.0
> **日期**: 2026-05-31
> **状态**: 待审批

---

## 1. 目标

实现 §5.4 Controlled ReAct Retrieval Loop — 基于 ReAct 框架的受控检索循环，根据中间结果动态调整检索策略，而非单次检索。

## 2. 当前架构

```
query → RetrievalRouter.retrieve() → EvidencePackage → LLMChemSolver.answer_from_package() → GroundedAnswer
```

单次检索，无法根据中间结果调整策略。

## 3. ReAct 架构

```
query → Round 1: retrieve() → LLM 评估证据
  ├─ 充分 → 生成回答
  └─ 不充分 → LLM 生成新查询 → Round 2: retrieve() → LLM 评估
       ├─ 充分 → 生成回答
       └─ 不充分 → Round 3（最多 N 轮）→ 最终回答
```

## 4. 设计

### 4.1 核心组件

新建 `src/solver/react_solver.py`：

```python
class ReActChemSolver:
    """多轮迭代检索的化学问答引擎。"""

    def __init__(self, store, *, elementkg_client=None, max_rounds=3):
        self.store = store
        self.elementkg_client = elementkg_client
        self.max_rounds = max_rounds
        self.router = RetrievalRouter(store, elementkg_client=elementkg_client)
        self.llm_solver = LLMChemSolver()

    def answer(self, query, *, doc_ids=None, top_k=8) -> GroundedAnswer:
        """ReAct 循环入口。"""
        accumulated_evidence = []
        retrieval_path = []
        round_queries = [query]

        for round_num in range(self.max_rounds):
            current_query = round_queries[-1]

            # 检索
            package = self.router.retrieve(current_query, doc_ids=doc_ids, top_k=top_k)
            accumulated_evidence.extend(package.candidate_evidence)
            retrieval_path.extend(package.retrieval_path)

            # LLM 评估证据是否充分
            assessment = self._assess_evidence(query, accumulated_evidence, round_num)

            if assessment["sufficient"]:
                # 证据充分 → 生成回答
                merged_package = self._build_package(query, accumulated_evidence, retrieval_path)
                return self.llm_solver.answer_from_package(merged_package)

            # 证据不充分 → 生成新查询
            if assessment.get("refined_query"):
                round_queries.append(assessment["refined_query"])
            else:
                break  # 无法生成新查询，停止

        # 所有轮次结束，用已有证据生成最佳回答
        merged_package = self._build_package(query, accumulated_evidence, retrieval_path)
        answer = self.llm_solver.answer_from_package(merged_package)
        answer.uncertainty = (answer.uncertainty or "") + f" [ReAct: {len(round_queries)} rounds]"
        return answer
```

### 4.2 证据评估（LLM）

```python
def _assess_evidence(self, query, evidence, round_num) -> dict:
    """LLM 评估当前证据是否足以回答问题。"""
    # 构建 prompt: 原始问题 + 当前证据列表
    # LLM 返回:
    # {"sufficient": true/false, "reason": "...", "refined_query": "..."}
    # refined_query: 针对缺失信息的新查询
```

### 4.3 查询优化策略

LLM 生成 refined_query 时的策略：
- **实体扩展**：原始查询 "DCHA 的合成条件" → 新查询 "dicyclohexylamine synthesis conditions"
- **通道切换**：entity_search 没找到 → 用 lexical_search 搜关键词
- **范围扩大**：限定 doc_id 没找到 → 搜索所有文档
- **具体化**：泛查询没找到 → 用更具体的关键词

### 4.4 停止条件

- 证据评估返回 `sufficient: true`
- 达到 max_rounds（默认 3）
- LLM 无法生成有意义的 refined_query
- 连续 2 轮检索结果完全相同

## 5. 与现有架构的关系

```
ReActChemSolver
  ├─ RetrievalRouter (不变) — 负责单次检索
  ├─ _assess_evidence() (新增) — LLM 评估证据
  └─ LLMChemSolver (复用) — 最终回答生成
```

不修改现有 router 和 solver，ReAct 是上层编排。

## 6. LLM 调用开销

| 场景 | LLM 调用次数 |
|---|---|
| 单轮（证据充分） | 1 评估 + 1 回答 = 2 次 |
| 两轮 | 2 评估 + 1 回答 = 3 次 |
| 三轮 | 3 评估 + 1 回答 = 4 次 |

相比当前单次检索（1 回答），最多增加 3 次 LLM 调用。

## 7. 文件变更清单

| 文件 | 操作 | 说明 |
|---|---|---|
| `src/solver/react_solver.py` | 新建 | ReAct 循环引擎 |
| `src/solver/prompts.py` | 修改 | 添加 evidence assessment prompt |
| `scripts/run_query.py` | 修改 | 添加 `--react` 参数 |
| `tests/test_react_solver.py` | 新建 | 测试 |

## 8. 验证标准

1. 单轮查询（证据充分）：行为与当前一致，只多 1 次 LLM 调用
2. 多轮查询（证据不足）：第 2 轮用 refined_query 检索，找到更多证据
3. 最大轮次限制：不超过 3 轮
4. 评测分数提升：特别是 reaction_comparison 和 property_query
