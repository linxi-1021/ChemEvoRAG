"""Test sandbox flow: verify V2 is created with auto-generated content."""
import sys, tempfile, yaml, shutil
sys.path.insert(0, 'src')
from pathlib import Path
from skill_evolution.patch import generate_all_patches
from skill_evolution.attribution import generate_trace_report
from skill_evolution.apply import PatchApplier

# Generate patches with auto-generated content
eval_path = Path('data/eval/eval_results.json')
logs_path = Path('data/eval/react_logs')
report = generate_trace_report(eval_path, logs_path)
skill_configs = {}
for f in Path('config/skills').glob('*.yaml'):
    data = yaml.safe_load(f.read_text('utf-8'))
    intent = data.get('trigger', {}).get('intent', '')
    if intent:
        skill_configs[intent] = data
patches = generate_all_patches(report, skill_configs)

# Find the patch with auto-generated content
target_patch = None
for p in patches:
    for a in p.prompt_artifacts:
        if a.content:
            target_patch = p
            break
    if target_patch:
        break

if not target_patch:
    print('No patch with auto-generated content found')
    sys.exit(1)

print(f'Using patch: {target_patch.patch_id}')
print(f'Skill: {target_patch.skill_name}')
artifact = target_patch.prompt_artifacts[0]
print(f'Artifact prompt_ref: {artifact.prompt_ref}')
print(f'Artifact content length: {len(artifact.content)} chars')

# Simulate sandbox flow
skills_dir = Path('config/skills')
prompts_dir = Path('config/prompts')
skill_config = skill_configs.get(target_patch.skill_name, {})

with tempfile.TemporaryDirectory() as tmpdir:
    tmp_skills = Path(tmpdir) / 'skills'
    tmp_prompts = Path(tmpdir) / 'prompts'
    tmp_skills.mkdir()
    tmp_prompts.mkdir()

    # Step 3: Copy only the skill being patched + referenced prompts
    src_skill = skills_dir / f'{target_patch.skill_name}.yaml'
    shutil.copy2(src_skill, tmp_skills / src_skill.name)

    strategy = skill_config.get('strategy', {})
    needed_refs = set()
    for section in ('assessment', 'answer_generation'):
        ref = strategy.get(section, {}).get('system_prompt_ref', '')
        if ref:
            needed_refs.add(ref)
    for f in prompts_dir.glob('*.yaml'):
        data = yaml.safe_load(f.read_text('utf-8'))
        if data and data.get('prompt_ref', '') in needed_refs:
            shutil.copy2(f, tmp_prompts / f.name)
            print(f'Copied to sandbox: {f.name}')

    # Apply patch + write prompt artifacts (with auto-generated content)
    applier = PatchApplier(skills_dir, prompts_dir)
    skill = yaml.safe_load((tmp_skills / f'{target_patch.skill_name}.yaml').read_text('utf-8'))
    applier.apply_to_memory(skill, target_patch)
    (tmp_skills / f'{target_patch.skill_name}.yaml').write_text(
        yaml.dump(skill, allow_unicode=True, default_flow_style=False, sort_keys=False),
        encoding='utf-8',
    )
    written = applier.write_prompt_artifacts_to_sandbox(tmp_prompts, target_patch)
    print(f'Prompt artifacts written to sandbox: {[Path(w).name for w in written]}')

    # Verify sandbox contents
    print(f'\nSandbox prompts:')
    for f in sorted(tmp_prompts.glob('*.yaml')):
        data = yaml.safe_load(f.read_text('utf-8'))
        content_len = len(data.get('content', ''))
        ref = data.get('prompt_ref', '')
        print(f'  {f.name}: prompt_ref={ref}, content={content_len} chars')

    # Verify V2 has auto-generated content
    v2_path = tmp_prompts / 'evidence_assessment_system_v2.yaml'
    if v2_path.exists():
        v2_data = yaml.safe_load(v2_path.read_text('utf-8'))
        content = v2_data.get('content', '')
        if 'auto-added by Skill Evolution' in content:
            print(f'\nRESULT: V2 has auto-generated rules: YES')
            print(f'V2 is NOT a copy of V1 - it contains failure-driven improvements')
        else:
            print(f'\nRESULT: V2 has auto-generated rules: NO')
            print(f'V2 is just a copy of V1')
    else:
        print(f'\nRESULT: V2 NOT FOUND in sandbox')
