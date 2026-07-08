"""Real Codex Agent — paper-specific QA via codex exec with gpt-5.5 model.
Confirmed working: codex exec -m gpt-5.5 (only model that works with ChatGPT Plus).

Usage: python run_agent_codex_final.py --limit 5
"""
import io,os,sys,time,certifi,json,subprocess,tempfile,shutil,re
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
os.environ['PYTHONIOENCODING']='utf-8'

from eval_utils import load_all_questions, save_results
from pathlib import Path
import argparse

p=argparse.ArgumentParser()
p.add_argument('--limit',type=int,default=5)
p.add_argument('--model',type=str,default='gpt-5.5')
a=p.parse_args()
qs=load_all_questions()
if a.limit<118: qs=qs[:a.limit]

MD_DIR = r'D:\Desktop\evo\ChemEvoRAG_Phase1-main\data\mineru_output'
CODEX = r'C:\Users\ASUS\AppData\Roaming\npm\codex.cmd'

print(f'Codex Agent (real, {a.model}): {len(qs)} questions')

def run_codex_paper(question, paper_id):
    """Give Codex the paper's markdown file and ask the question."""
    tmpdir = tempfile.mkdtemp(prefix='codex_paper_')
    try:
        paper_md = MD_DIR + '\\' + paper_id + '\\extracted\\full.md'
        if not os.path.exists(paper_md):
            return '[NO PAPER]'

        # Copy to a clean name
        work_dir = os.path.join(tmpdir, 'work')
        os.makedirs(work_dir, exist_ok=True)
        target_md = os.path.join(work_dir, 'paper.md')
        shutil.copy2(paper_md, target_md)

        # Write prompt file
        prompt_file = os.path.join(work_dir, 'prompt.txt')
        output_file = os.path.join(work_dir, 'output.txt')
        err_file = os.path.join(work_dir, 'err.txt')

        prompt = f'Read paper.md and answer: {question}. Answer concisely with exact values/formulas. Output ONLY the answer.'

        with open(prompt_file, 'w', encoding='utf-8') as f:
            f.write(prompt)

        # Run via cmd /c batch — stream stdout live so user sees progress
        batch_file = os.path.join(work_dir, 'run.bat')
        batch = f'@echo off\nchcp 65001 >nul\ncd /d "{work_dir}"\n"{CODEX}" exec -m {a.model} --sandbox workspace-write --ephemeral --skip-git-repo-check < "{prompt_file}" > "{output_file}" 2> "{err_file}"\nexit /b 0\n'
        with open(batch_file, 'w', encoding='utf-8') as f:
            f.write(batch)

        # Use Popen so user can see live output from codex
        proc = subprocess.Popen(
            ['cmd', '/c', batch_file],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, cwd=work_dir
        )
        # Stream output live
        live_output = []
        for line in proc.stdout:
            stripped = line.rstrip()
            if stripped:
                live_output.append(stripped)
                print(f'      [codex] {stripped[:120]}', flush=True)
        proc.wait(timeout=300)

        output = ''
        for of in [output_file, err_file]:
            try:
                with open(of, 'r', encoding='utf-8', errors='replace') as f:
                    output += f.read() + '\n'
            except: pass

        # Clean: remove ANSI, frame chars, Codex header
        output = re.sub(r'\x1b\[[0-9;]*m', '', output)
        # Remove box drawing chars
        output = re.sub(r'[╭╰│├└─┬┴┼]+', '', output)
        # Remove known prefix lines
        lines = []
        skip_prefixes = ['OpenAI Codex', 'workdir:', 'model:', 'provider:', 'approval:',
                        'sandbox:', 'reasoning', 'session id:', 'user', 'Reading prompt',
                        '--------', 'codex', 'tokens used', 'Wall time:', 'exited', 'Output:',
                        'ERROR:', '找不到']
        for line in output.split('\n'):
            stripped = line.strip()
            if not stripped: continue
            if any(stripped.startswith(p) for p in skip_prefixes): continue
            lines.append(stripped)

        answer = '\n'.join(lines).strip()
        if not answer or len(answer) < 2:
            answer = '[EMPTY OUTPUT]'
        return answer[:2000]

    except subprocess.TimeoutExpired:
        return '[TIMEOUT]'
    except Exception as e:
        return f'[ERROR] {e}'
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

rs=[]
for i,q in enumerate(qs):
    paper_id = q['paper_id']
    print(f'  [{i+1}/{len(qs)}] {q["id"]}  paper={paper_id}  intent={q["intent"]}', flush=True)
    t0=time.time()
    ans = run_codex_paper(q['question'], paper_id)
    elapsed = round(time.time()-t0, 2)
    safe = str(ans).encode('ascii','replace').decode('ascii')
    print(f'    -> {elapsed}s  {safe[:120]}')
    rs.append({'question_id':q['id'],'paper_id':q['paper_id'],'paper_title':q['paper_title'],
        'intent':q['intent'],'question':q['question'],'ground_truth':q['ground_truth_answer'],
        'key_entities':q.get('key_entities',[]),'system_answer':str(ans).strip(),
        'elapsed_sec':elapsed})

sys.path.insert(0,'D:/Desktop/evo/ChemEvoRAG_Phase1-main/src')
os.environ.setdefault('API_KEY','sk-gr-02f44cd3404411cd11cdd88da080685ac1c54986')
os.environ.setdefault('BASE_URL','https://endpoint.greatrouter.com')
os.environ.setdefault('LLM_MODEL','DeepSeek-V3')
from judge import judge_answer
print('Scoring...')
for i,r in enumerate(rs):
    j=judge_answer(r['system_answer'],r['ground_truth'],r.get('key_entities',[]))
    r['judge_score']=j.get('score',0); r['entities_found']=j.get('key_entities_found',[]); r['entities_missing']=j.get('key_entities_missing',[]); r['judge_reasoning']=j.get('reasoning','')
    if (i+1)%10==0: print(f'  {i+1}/{len(rs)}')
s=[r['judge_score'] for r in rs]
print(f'\nCodex Agent (real, {a.model}): {sum(s)/len(s):.3f}')
save_results(f'agent_codex_real_{a.model.replace(".","_")}', rs, {'model': a.model, 'mode': 'codex-cli-paper'})
print('Done')
