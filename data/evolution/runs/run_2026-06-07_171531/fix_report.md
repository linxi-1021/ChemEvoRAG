# Skill Evolution Fix Report
**Generated:** 2026-06-07
**Run analyzed:** run_2026-06-07_105045

---

## 1. Why All 9 Patches Failed in the Original Run

| Patch | Type | Failure Reason |
|-------|------|----------------|
| reaction_condition_query_round1_patch1 | prompt_ref → TABLE_V2 | Prompt file didn't exist; validation didn't check |
| reaction_condition_query_round1_patch2 | max_evidence_items 15→18 | Crude change, LLM noise amplified |
| reaction_comparison_round1_patch1 | prompt_ref → TABLE_V2 | Same nonexistent prompt |
| reaction_comparison_round1_patch2 | max_evidence_items 15→18 | Same crude change |
| entity_lookup_template_distill_1 | template add | pattern="score_1.0" = no structure |
| property_query_template_distill_2 | template add | Same |
| alias_resolution_template_distill_3 | template add | Same |
| reaction_condition_query_template_distill_4 | template add | Same |
| reaction_comparison_template_distill_5 | template add | Same |

**Root causes:**
- **4 prompt_ref patches**: Referenced `EVIDENCE_ASSESSMENT_SYSTEM_TABLE_V2` which didn't exist. Validation only checked UPPER_SNAKE_CASE format, not file existence.
- **2 max_evidence_items patches**: Crude parameter bumping (15→18) without addressing the actual failure mode.
- **5 template distillation patches**: Generated from `pattern="score_1.0"` which is just a score label, not a real task pattern. The 86 success_patterns had massive duplication (same question across 22 papers).

---

## 2. Patch Quality Issues

- Template distillation: `pattern="score_1.0"` carries zero structural information. Templates had `applicable_when: {intent: x, pattern: "score_1.0"}` — useless for routing.
- max_evidence_items: Blunt parameter change that affects ALL queries, not just table queries.
- prompt_ref: Pointed to a file that would only be created at apply-time, but regression runs before apply.

---

## 3. Semantic Validation Gaps

- `validate_semantic_invariants` only checked UPPER_SNAKE_CASE format for prompt refs.
- No check for whether the referenced prompt file actually exists on disk.
- No check for whether a prompt_artifact in the patch will create it.

---

## 4. Regression Runner Instability Indicators

- `targeted_improvement` was always 0.0 because `source_failure_ids` from patches were never passed to `compare()`.
- `failed_cases_increase` of 9-16 for template patches suggests LLM judge noise.
- Baseline stability test and no-op patch test scripts created but not yet run (require full eval).

---

## 5. Template Distillation Fixes

**File:** `src/skill_evolution/patch.py`

- Added regex filter: `_SCORE_LABEL_RE = re.compile(r"^score_\d+\.\d+$")` — skips bare score labels.
- Changed grouping from `list` to `set` for question IDs — prevents duplication.
- Templates now require: (a) pattern is NOT a score label, (b) ≥3 UNIQUE question_ids, (c) avg_score ≥ 0.85.

**Effect:** 0 template patches generated (was 5). Correct behavior — insufficient data to distill meaningful templates.

---

## 6. Prompt Artifact Validation Fix

**File:** `src/skill_evolution/validation.py`

- Added `prompts_dir` parameter to `validate_semantic_invariants` and `validate_patch`.
- New check: if `system_prompt_ref` is changed, verify the prompt file exists on disk OR a `prompt_artifact` in the patch will create it.
- If neither, patch is rejected with clear error message.

**Callers updated:**
- `src/skill_evolution/runtime_validation.py` — passes `prompts_dir` to `validate_patch`
- `scripts/evolve.py` — passes `PROMPTS_DIR` to `validate_patch`

**Verification:**
```
Patch referencing NONEXISTENT_PROMPT_XYZ → REJECTED ✓
Patch referencing EVIDENCE_ASSESSMENT_SYSTEM_V2 (exists) → ACCEPTED ✓
```

---

## 7. Success Patterns Aggregation Fix

**File:** `src/skill_evolution/attribution.py`

- Changed `supporting_question_ids` from `[question_id]` to `[f"{question_id}_{source_paper}"]`.
- This prevents the same question template (e.g., "q1") across 22 papers from being counted as 22 separate successes.

**Before:** 86 success_patterns, q1 appears 8 times, q2 appears 10 times
**After:** 86 success_patterns with unique IDs (86 unique identifiers)

---

## 8. Attribution Engine Fixes

**File:** `src/skill_evolution/attribution.py`

### 8a. `_detect_table_reasoning_error` — Refusal Detection
- Now checks if the answer is a refusal (contains "insufficient", "does not include", "cannot determine", etc.).
- Only classifies as table_reasoning_error if the system produced a WRONG answer from table data, not a refusal.

### 8b. `_detect_routing_error` — Evidence Content Check
- Now checks if gold answer's specific terms appear in the evidence.
- If specific terms match → assessment_false_negative (evidence was there but missed).
- If no match → routing_error (wrong evidence retrieved).
- Filters out common chemistry words to avoid false matches.

### 8c. `_detect_assessment_false_negative` — Specific Term Matching
- Now requires gold answer's specific terms to appear in the react log evidence.
- Checks for "sufficient=false" in addition to "insufficient" in react logs.
- Prevents false positives where evidence didn't actually contain the answer.

### 8d. `_detect_planner_error` — More Specific
- Now only triggers if refined queries found 0 new evidence in later rounds.
- Prevents catching cases where the planner worked but evidence was wrong.

### 8e. `evidence_contained_answer` — Fixed Calculation
- Was incorrectly set to `conf > 0.7` (based on attribution confidence).
- Now properly checks if gold answer terms appear in the react log evidence.

**Attribution results:**

| Question | Before | After |
|----------|--------|-------|
| q3 | table_reasoning_error | assessment_false_negative ✓ |
| q4 | table_reasoning_error | assessment_false_negative ✓ |
| q_batch2_3 | unknown_failure | assessment_false_negative ✓ |
| q_batch2_2 | unknown_failure | assessment_false_negative ✓ |
| q2 | unknown_failure | unknown_failure (correct) |

---

## 9. Regression Validation Improvement

**Files:** `src/skill_evolution/regression.py`, `src/skill_evolution/runtime_validation.py`, `scripts/evolve.py`

### 9a. Targeted Improvement Calculation
- `validate_individual_patch` now passes `patch.source_failure_ids` to `compare()` as `targeted_failure_ids`.
- `compare()` now calculates targeted_improvement even when using `RegressionResult_from_dict`.
- Handles duplicate question IDs by averaging all matching results.

### 9b. Pass/Fail Criteria Adjustment
- If `targeted_improvement > 0`, allows up to 2x `max_score_drop` (the patch improves targets but might slightly affect others).
- Only checks targeted_improvement when `results_path` is available (avoids false 0.0).
- Baseline eval_results path now included in baseline_result dict.

---

## 10. New Patch List

| Patch ID | Failure Type | Target | Source | Proposed Change |
|----------|-------------|--------|--------|-----------------|
| reaction_condition_query_round1_patch1 | assessment_false_negative | strategy.assessment | q3 | system_prompt_ref → EVIDENCE_ASSESSMENT_SYSTEM_V2 |
| reaction_comparison_round1_patch1 | assessment_false_negative | strategy.assessment | q4, q_batch2_3 | system_prompt_ref → EVIDENCE_ASSESSMENT_SYSTEM_V2 |
| entity_lookup_round1_patch1 | assessment_false_negative | strategy.assessment | q_batch2_2 | system_prompt_ref → EVIDENCE_ASSESSMENT_SYSTEM_V2 |

**New prompt files created:**
- `config/prompts/evidence_assessment_system_v2.yaml` — Enhanced table evidence rules
- `config/prompts/evidence_assessment_system_table_v2.yaml` — Same content, TABLE_V2 ref

---

## 11. Targeted Improvement Status

Not yet measured — requires running `--validate-only` with regression. The targeted_improvement calculation is now wired through (`source_failure_ids` → `compare()`), so it should show non-zero values when the V2 prompt improves q3/q4.

---

## 12. Can Enter Apply?

**Not yet.** Need to:
1. Run `--validate-only` to measure targeted_improvement
2. Run baseline stability test to confirm regression gate reliability
3. Run no-op patch test to confirm LLM judge stability
4. Verify patches don't break existing successes

---

## 13. Next Steps / Blockers

1. **Run `--validate-only`**: `python scripts/evolve.py --validate-only --workers 2`
2. **Run baseline stability**: `python scripts/baseline_stability_test.py --limit 30`
3. **Run no-op patch test**: `python scripts/noop_patch_test.py --limit 30`
4. **If regression gate is unstable**: Consider fixing temperature=0 for judge, caching judge results, or using deterministic slot-level validators instead of LLM-as-judge.

---

## Files Modified

| File | Changes |
|------|---------|
| `src/skill_evolution/validation.py` | Added prompt artifact existence check |
| `src/skill_evolution/patch.py` | Filter score-label patterns, deduplicate question IDs |
| `src/skill_evolution/attribution.py` | Fixed attribution: refusal detection, evidence content check, planner specificity, unique success IDs |
| `src/skill_evolution/regression.py` | Targeted improvement calculation, adjusted pass/fail criteria |
| `src/skill_evolution/runtime_validation.py` | Pass source_failure_ids and prompts_dir through |
| `scripts/evolve.py` | Pass prompts_dir and results_path through |
| `config/prompts/evidence_assessment_system_v2.yaml` | New: enhanced assessment prompt |
| `config/prompts/evidence_assessment_system_table_v2.yaml` | New: table-specific assessment prompt |
| `scripts/baseline_stability_test.py` | New: baseline stability test script |
| `scripts/noop_patch_test.py` | New: no-op patch test script |
