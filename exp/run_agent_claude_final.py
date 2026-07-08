"""Real Claude Code Agent — paper-specific QA over MinerU markdown.

Each question is mapped to a specific paper. The agent gets that paper's markdown.
This is the fairest test: Claude Code vs ChemEvoRAG, same paper, same question.

Usage: python run_agent_claude_final.py --limit 5
"""
import io,os,sys,time,certifi,json,subprocess,tempfile,shutil,re
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
os.environ['PYTHONIOENCODING']='utf-8'

from eval_utils import load_all_questions, save_results
from pathlib import Path
import argparse

p=argparse.ArgumentParser(); p.add_argument('--limit',type=int,default=5)
p.add_argument('--model',type=str,default='opus')
a=p.parse_args()
qs=load_all_questions()
if a.limit<118: qs=qs[:a.limit]

MD_DIR = Path('D:/Desktop/evo/ChemEvoRAG_Phase1-main/data/mineru_output')
BASH = r'E:\Git\bin\bash.exe'

print(f'Claude Code Agent ({a.model}): {len(qs)} questions')

def run_claude_paper_specific(question, paper_id):
    """Give Claude the specific paper's full markdown and ask the question."""
    tmpdir = tempfile.mkdtemp(prefix='claude_paper_')
    try:
        # Paper-specific files
        paper_md = MD_DIR / paper_id / 'extracted' / 'full.md'
        if not paper_md.exists():
            return f'[NO PAPER: {paper_id}]', 0

        # Copy paper to a clean location
        work_dir = os.path.join(tmpdir, 'paper')
        os.makedirs(work_dir, exist_ok=True)
        target_md = os.path.join(work_dir, 'paper.md')
        shutil.copy2(str(paper_md), target_md)

        prompt = f"""Read the chemistry paper at paper/paper.md and answer this question about it.

QUESTION: {question}

The paper is in markdown format with tables, chemical structures, experimental sections, etc.
Read it carefully, find the specific answer, and output ONLY the answer starting with "ANSWER:"."""

        prompt_file = os.path.join(tmpdir, 'prompt.txt').replace('\\', '/')
        output_file = os.path.join(tmpdir, 'output.txt').replace('\\', '/')
        err_file = os.path.join(tmpdir, 'err.txt').replace('\\', '/')
        script_file = os.path.join(tmpdir, 'run.sh').replace('\\', '/')

        with open(prompt_file, 'w', encoding='utf-8', newline='\n') as f:
            f.write(prompt)

        # Shell script: change to work directory so paper.md is in context
        script = f'''#!/bin/bash
export NO_COLOR=1
export PYTHONIOENCODING=utf-8
export LANG=en_US.UTF-8
cd "{work_dir}"
claude -p --model {a.model} --max-budget-usd 2 --output-format text < "{prompt_file}" > "{output_file}" 2> "{err_file}"
exit 0
'''
        with open(script_file, 'w', encoding='utf-8', newline='\n') as f:
            f.write(script)

        result = subprocess.run(
            [BASH, script_file],
            capture_output=True, text=True, timeout=300,
            cwd=work_dir
        )

        # Read from output file
        output = ''
        for of in [output_file, err_file]:
            try:
                with open(of, 'r', encoding='utf-8', errors='replace') as f:
                    output += f.read() + '\n'
            except:
                pass

        output = re.sub(r'\x1b\[[0-9;]*m', '', output)

        if 'ANSWER:' in output:
            answer = output.split('ANSWER:')[-1].strip()
            lines = []
            for line in answer.split('\n'):
                line = line.strip()
                if not line: break
                if line.startswith('Perm'): continue
                lines.append(line)
            answer = ' '.join(lines)
        else:
            lines = [l.strip() for l in output.split('\n') if l.strip() and len(l.strip()) > 15 and not l.strip().startswith('Perm')]
            answer = '\n'.join(lines[-10:]) if lines else '[EMPTY OUTPUT]'

        return answer[:2000], len(output)

    except subprocess.TimeoutExpired:
        return '[TIMEOUT]', 0
    except Exception as e:
        return f'[ERROR] {e}', 0
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

rs=[]
for i,q in enumerate(qs):
    paper_id = q['paper_id']  # The paper this question is about
    print(f'  [{i+1}/{len(qs)}] {q["id"]}  paper={paper_id}  intent={q["intent"]}', flush=True)
    t0=time.time()
    ans, out_len = run_claude_paper_specific(q['question'], paper_id)
    elapsed = round(time.time()-t0, 2)
    print(f'    -> {elapsed}s  out={out_len}B  {str(ans)[:150]}...')
    rs.append({'question_id':q['id'],'paper_id':q['paper_id'],'paper_title':q['paper_title'],
        'intent':q['intent'],'question':q['question'],'ground_truth':q['ground_truth_answer'],
        'key_entities':q.get('key_entities',[]),'system_answer':str(ans).strip(),
        'elapsed_sec':elapsed})

sys.path.insert(0,'D:/Desktop/evo/ChemEvoRAG_Phase1-main/src')
from judge import judge_answer
# Ensure API key is set for judge
os.environ.setdefault('API_KEY','sk-gr-02f44cd3404411cd11cdd88da080685ac1c54986')
os.environ.setdefault('BASE_URL','https://endpoint.greatrouter.com')
os.environ.setdefault('LLM_MODEL','DeepSeek-V3')
print('Scoring...')
for i,r in enumerate(rs):
    j=judge_answer(r['system_answer'],r['ground_truth'],r.get('key_entities',[]))
    r['judge_score']=j.get('score',0); r['entities_found']=j.get('key_entities_found',[]); r['entities_missing']=j.get('key_entities_missing',[]); r['judge_reasoning']=j.get('reasoning','')
    if (i+1)%10==0: print(f'  {i+1}/{len(rs)}')
s=[r['judge_score'] for r in rs]; print(f'\nClaude Code Agent (real, {a.model}): {sum(s)/len(s):.3f}')
save_results(f'agent_claude_real_{a.model}', rs, {'model': a.model, 'mode': 'claude-paper-specific'})
print('Done')
