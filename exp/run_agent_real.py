"""Real Claude Code & Codex Agent driver — paper QA over MinerU markdown.

Uses PowerShell to call CLI tools, avoiding cmd encoding issues.

Usage:
  python run_agent_real.py --agent claude --limit 5
  python run_agent_real.py --agent codex --limit 5
"""
import io,os,sys,time,certifi,json,subprocess,tempfile,shutil
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
os.environ['SSL_CERT_FILE']=certifi.where()
os.environ['PYTHONIOENCODING']='utf-8'

from eval_utils import load_all_questions, save_results
from pathlib import Path
import argparse

p=argparse.ArgumentParser(); p.add_argument('--limit',type=int,default=5)
p.add_argument('--agent',type=str,default='claude', choices=['claude','codex'])
a=p.parse_args()
qs=load_all_questions()
if a.limit<118: qs=qs[:a.limit]

MD_DIR = r'D:\Desktop\evo\ChemEvoRAG_Phase1-main\data\mineru_output'
CLAUDE = r'C:\Users\ASUS\AppData\Roaming\npm\claude.cmd'
CODEX = r'C:\Users\ASUS\AppData\Roaming\npm\codex.cmd'

paper_count = sum(1 for d in Path(MD_DIR).iterdir() if d.is_dir() and (d/'extracted'/'full.md').exists())
print(f'{a.agent.upper()} Agent: {len(qs)} questions, {paper_count} papers')

def run_agent(question, agent_type='claude'):
    """Run agent via PowerShell to avoid encoding issues, capture output via temp file."""
    tmpdir = tempfile.mkdtemp(prefix=f'agent_{agent_type}_')
    try:
        prompt_file = os.path.join(tmpdir, 'prompt.txt').replace('\\', '/')
        output_file = os.path.join(tmpdir, 'output.txt').replace('\\', '/')
        err_file = os.path.join(tmpdir, 'err.txt').replace('\\', '/')

        prompt = f"""You are a chemistry research assistant. Answer the question by searching and reading chemistry papers.

Papers location: D:/Desktop/evo/ChemEvoRAG_Phase1-main/data/mineru_output
Each subdirectory (1, 2, 3, ...23) contains a parsed paper at extracted/full.md (markdown with tables, schemes, and text).

STEPS:
1. Use Grep to search keywords/compound names/labels across all extracted/full.md files to find the correct paper
2. Use Read to examine relevant sections (especially tables, experimental sections)
3. Answer concisely with specific evidence: cite paper number, table number, and key values

QUESTION: {question}

IMPORTANT: Your final message should be ONLY the answer. Start your final answer with 'ANSWER:' on its own line. Do NOT ask follow-up questions."""

        with open(prompt_file.replace('/', '\\'), 'w', encoding='utf-8') as f:
            f.write(prompt)

        if agent_type == 'claude':
            ps_cmd = f'''$OutputEncoding = [Console]::OutputEncoding = [Text.UTF8Encoding]::UTF8
$env:NO_COLOR = "1"
$env:PYTHONIOENCODING = "utf-8"
$prompt = Get-Content -Path "{prompt_file}" -Raw -Encoding UTF8
$tmpfile = New-TemporaryFile
$prompt | Out-File -FilePath $tmpfile -Encoding UTF8 -NoNewline
& "{CLAUDE}" -p --model opus --add-dir "{MD_DIR}" --max-budget-usd 2 --output-format text < $tmpfile > "{output_file}" 2> "{err_file}"
Remove-Item $tmpfile
'''
        else:
            ps_cmd = f'''$OutputEncoding = [Console]::OutputEncoding = [Text.UTF8Encoding]::UTF8
$env:NO_COLOR = "1"
$env:PYTHONIOENCODING = "utf-8"
$prompt = Get-Content -Path "{prompt_file}" -Raw -Encoding UTF8
$tmpfile = New-TemporaryFile
$prompt | Out-File -FilePath $tmpfile -Encoding UTF8 -NoNewline
& "{CODEX}" exec --add-dir "{MD_DIR}" --sandbox workspace-write --ephemeral --skip-git-repo-check < $tmpfile > "{output_file}" 2> "{err_file}"
Remove-Item $tmpfile
'''

        ps_script = os.path.join(tmpdir, 'run.ps1').replace('\\', '/')
        with open(ps_script.replace('/', '\\'), 'w', encoding='utf-8') as f:
            f.write(ps_cmd)

        result = subprocess.run(
            ['powershell', '-ExecutionPolicy', 'Bypass', '-File', ps_script],
            capture_output=True, text=True, timeout=300,
            cwd=r'D:\Desktop\evo\ChemEvoRAG_Phase1-main\exp',
            env={**os.environ, 'PYTHONIOENCODING': 'utf-8'}
        )

        # Read output file
        output = ''
        if os.path.exists(output_file.replace('/', '\\')):
            with open(output_file.replace('/', '\\'), 'r', encoding='utf-8', errors='replace') as f:
                output = f.read()

        # Also check stderr for captured output (some versions of claude write to stderr)
        err_output = ''
        if os.path.exists(err_file.replace('/', '\\')):
            with open(err_file.replace('/', '\\'), 'r', encoding='utf-8', errors='replace') as f:
                err_output = f.read()

        # Some claude versions write answer to stderr
        full_output = (output + '\n' + err_output).strip()

        # Extract ANSWER section
        if 'ANSWER:' in full_output:
            answer = full_output.split('ANSWER:')[-1].strip().split('\n')[0].strip()
        elif full_output:
            # Remove ANSI codes, frame chars
            import re
            clean = re.sub(r'\x1b\[[0-9;]*m', '', full_output)
            clean = re.sub(r'[╭╰│├└─]+', '', clean)
            lines = [l.strip() for l in clean.split('\n') if l.strip() and len(l.strip()) > 10]
            # Take last 5 substantive lines
            answer = '\n'.join(lines[-5:]) if lines else clean.strip()
        else:
            answer = '[EMPTY OUTPUT]'

        return answer[:2000], full_output[:1000]

    except subprocess.TimeoutExpired:
        return '[TIMEOUT]', f'timeout after 300s'
    except Exception as e:
        return f'[ERROR] {e}', ''
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

print(f'Running...')
rs=[]
for i,q in enumerate(qs):
    print(f'  [{i+1}/{len(qs)}] {q["id"]}  intent={q["intent"]}', flush=True)
    t0=time.time()
    ans, raw = run_agent(q['question'], a.agent)
    elapsed = round(time.time()-t0, 2)
    print(f'    -> {elapsed}s  {str(ans)[:120]}...')
    if '[EMPTY' in str(ans) or '[ERROR' in str(ans):
        print(f'    RAW: {raw[:200]}')
    rs.append({'question_id':q['id'],'paper_id':q['paper_id'],'paper_title':q['paper_title'],
        'intent':q['intent'],'question':q['question'],'ground_truth':q['ground_truth_answer'],
        'key_entities':q.get('key_entities',[]),'system_answer':str(ans).strip(),
        'elapsed_sec':elapsed})

sys.path.insert(0,'D:/Desktop/evo/ChemEvoRAG_Phase1-main/src')
from judge import judge_answer
print('Scoring...')
for i,r in enumerate(rs):
    j=judge_answer(r['system_answer'],r['ground_truth'],r.get('key_entities',[]))
    r['judge_score']=j.get('score',0); r['entities_found']=j.get('key_entities_found',[]); r['entities_missing']=j.get('key_entities_missing',[]); r['judge_reasoning']=j.get('reasoning','')
    if (i+1)%10==0: print(f'  {i+1}/{len(rs)}')
s=[r['judge_score'] for r in rs]; print(f'\n{a.agent.upper()} Agent: {sum(s)/len(s):.3f}')
save_results(f'agent_{a.agent}_real', rs, {'model': 'opus', 'mode': f'{a.agent}-cli'})
print('Done')
