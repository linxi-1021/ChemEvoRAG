"""Run all 5 verification tests for Skill Evolution refactoring."""
import sys
import traceback
sys.path.insert(0, 'src')

results = []

def run_test(name, fn):
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print('='*60)
    try:
        fn()
        results.append((name, "PASSED", None))
        print(f"RESULT: PASSED")
    except Exception as e:
        results.append((name, "FAILED", str(e)))
        print(f"RESULT: FAILED - {e}")
        traceback.print_exc()

# ─── Test 1: Module imports ───
def test_imports():
    from skill_evolution.attribution import generate_trace_report
    print('attribution OK')
    from skill_evolution.patch import generate_all_patches
    print('patch OK')
    from skill_evolution.validation import validate_patch
    print('validation OK')
    from skill_evolution.regression import RegressionRunner
    print('regression OK')
    from skill_evolution.apply import PatchApplier
    print('apply OK')
    from skill_evolution.rollback import SnapshotManager
    print('rollback OK')
    from skill_evolution.runtime_validation import validate_individual_patch
    print('runtime_validation OK')

run_test("1: All skill_evolution module imports", test_imports)

# ─── Test 2: evolve.py --analyze-only ───
def test_analyze_only():
    import subprocess
    result = subprocess.run(
        [sys.executable, "scripts/evolve.py", "--analyze-only"],
        capture_output=True, text=True, cwd="."
    )
    print("STDOUT:", result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"Exit code {result.returncode}")
    # Check trace_report.json was generated
    from pathlib import Path
    tr = Path("trace_report.json")
    if tr.exists():
        print(f"trace_report.json generated ({tr.stat().st_size} bytes)")
    else:
        print("WARNING: trace_report.json not found (may need evaluation data)")

run_test("2: evolve.py --analyze-only", test_analyze_only)

# ─── Test 3: evolve.py --dry-run ───
def test_dry_run():
    import subprocess
    result = subprocess.run(
        [sys.executable, "scripts/evolve.py", "--dry-run"],
        capture_output=True, text=True, cwd="."
    )
    print("STDOUT:", result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"Exit code {result.returncode}")

run_test("3: evolve.py --dry-run", test_dry_run)

# ─── Test 4: Semantic validation rejects non-existent prompt_ref ───
def test_prompt_ref_rejection():
    from skill_evolution.validation import validate_patch
    from skill_evolution.patch import PatchSchema, PatchOperation, PatchStatus
    from skill_evolution.attribution import FailureType
    from pathlib import Path

    patch = PatchSchema(
        skill_name='reaction_condition_query',
        primary_failure_type=FailureType.ASSESSMENT_FALSE_NEG,
        target_path='strategy.assessment',
        operation=PatchOperation.UPDATE,
        proposed_value={'system_prompt_ref': 'NONEXISTENT_PROMPT_XYZ'},
        source_failure_ids=['q999'],
        rationale='Test patch with non-existent prompt ref',
    )
    config = {
        'name': 'reaction_condition_query',
        'trigger': {'intent': 'reaction_condition_query'},
        'evolution': {
            'mutable_paths': ['strategy.assessment'],
            'frozen_paths': ['name']
        }
    }
    result = validate_patch(patch, config, skill_filename='reaction_condition_query.yaml', prompts_dir=Path('config/prompts'))
    print(f'Valid: {result.valid}')
    print(f'Errors: {result.errors}')
    print(f'Rejection category: {result.rejection_category}')
    assert not result.valid, 'Should reject non-existent prompt_ref'
    print('TEST PASSED: Non-existent prompt_ref correctly rejected')

run_test("4: Semantic validation rejects non-existent prompt_ref", test_prompt_ref_rejection)

# ─── Test 5: Template distillation rejects score-based patterns ───
def test_score_pattern_rejection():
    from skill_evolution.patch import distill_templates
    from skill_evolution.attribution import SuccessPattern

    patterns = [
        SuccessPattern(
            intent='reaction_condition_query',
            pattern='score_1.0',
            frequency=5,
            avg_score=1.0,
            supporting_question_ids=['q1', 'q2', 'q3', 'q4', 'q5'],
        )
    ]
    patches = distill_templates(patterns, [])
    print(f'Patches generated: {len(patches)}')
    assert len(patches) == 0, f'Should reject score-based patterns, got {len(patches)} patches'
    print('TEST PASSED: score_1.0 pattern correctly rejected')

run_test("5: Template distillation rejects score-based patterns", test_score_pattern_rejection)

# ─── Summary ───
print(f"\n{'='*60}")
print("SUMMARY")
print('='*60)
for name, status, err in results:
    marker = "PASS" if status == "PASSED" else "FAIL"
    detail = f" ({err})" if err else ""
    print(f"  [{marker}] {name}{detail}")

passed = sum(1 for _, s, _ in results if s == "PASSED")
print(f"\n{passed}/{len(results)} tests passed")
