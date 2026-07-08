"""Real Claude Code Agent — paper QA over MinerU markdown via file-based shell scripts.

Uses bash script files + output files to completely avoid subprocess pipe encoding issues.

Usage: python run_agent_claude_real.py --limit 5
"""
import io,os,sys,time,certifi,json,subprocess,tempfile,shutil,re,hashlib
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
os.environ['PYTHONIOENCODING']='utf-8'

from eval_utils import load_all_questions, save_results
from pathlib import Path
import argparse

p=argparse.ArgumentParser(); p.add_argument('--limit',type=int,default=5)
a=p.parse_args()
qs=load_all_questions()
if a.limit<118: qs=qs[:a.limit]

MD_DIR = r'D:\Desktop\evo\ChemEvoRAG_Phase1-main\data\mineru_output'
BASH = r'E:\Git\bin\bash.exe'

paper_count = sum(1 for d in Path(MD_DIR).iterdir() if d.is_dir() and (d/'extracted'/'full.md').exists())
print(f'Claude Code Agent: {len(qs)} questions, {paper_count} papers')

def run_claude(question, qid):
    """Write bash script + run via bash, read output from file."""
    tmpdir = tempfile.mkdtemp(prefix='claude_qa_')
    try:
        # Write prompt to file
        prompt_file = os.path.join(tmpdir, 'prompt.txt')
        output_file = os.path.join(tmpdir, 'output.txt')
        err_file = os.path.join(tmpdir, 'err.txt')
        script_file = os.path.join(tmpdir, 'run.sh')

        prompt = f"""You are a chemistry research assistant. Answer the question by searching and reading chemistry papers.

Papers location: {MD_DIR}
Each numbered subdirectory contains a parsed paper at extracted/full.md (markdown with tables, schemes, chemical structures, and text).

CRITICAL: The question is about a SPECIFIC paper. The question is from paper {qid.split('_')[0] if '_' in qid else 'with matching content'}. Search across ALL papers to find the one that contains the answer, then read that specific paper's tables and text.

INSTRUCTIONS:
1. First, use Grep to search for the specific labels/compounds/conditions mentioned in the question across all extracted/full.md files
2. Once you identify the correct paper, use Read to examine its relevant sections ? especially tables, experimental procedures, and results sections
3. Give a PRECISE, SHORT answer with the exact values/formulas/names requested
4. Start your final answer with "ANSWER:" on its own line

QUESTION: {question}
"""

        with open(prompt_file, 'w', encoding='utf-8') as f:
            f.write(prompt)

        # Write shell script
        script = f'''#!/bin/bash
export NO_COLOR=1
export PYTHONIOENCODING=utf-8
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

claude -p --model opus \\
    --add-dir "{MD_DIR}" \\
    --max-budget-usd 2 \\
    --output-format text \\
    < "{prompt_file}" \\
    > "{output_file}" 2> "{err_file}"
exit 0
'''
        with open(script_file, 'w', encoding='utf-8', newline='\n') as f:
            f.write(script)

        result = subprocess.run(
            [BASH, script_file],
            capture_output=True, text=True,
            timeout=420,
            cwd=r'D:\Desktop\evo\ChemEvoRAG_Phase1-main\exp'
        )

        # Read answer from output file (NOT from pipe)
        output = ''
        for of in [output_file, err_file]:
            try:
                with open(of, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
                    if content.strip():
                        output += content + '\n'
            except:
                pass

        # Clean up output
        output = re.sub(r'\x1b\[[0-9;]*m', '', output)
        output = re.sub(r'[╭╰│├└─]+', '', output)

        # Extract ANSWER
        if 'ANSWER:' in output:
            answer = output.split('ANSWER:')[-1].strip()
            # Take lines until empty line or frame char
            lines = []
            for line in answer.split('\n'):
                line = line.strip()
                if not line:
                    break
                if line.startswith('Permission deny'):
                    continue
                lines.append(line)
            answer = ' '.join(lines)
        else:
            # Get all substantive lines
            lines = [l.strip() for l in output.split('\n')
                     if l.strip() and len(l.strip()) > 15
                     and not l.startswith('Perm')]
            answer = '\n'.join(lines[-10:]) if lines else '[EMPTY OUTPUT]'

        return answer[:2000], len(output) if output else 0

    except subprocess.TimeoutExpired:
        return '[TIMEOUT]', 0
    except Exception as e:
        return f'[ERROR] {e}', 0
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

rs=[]
for i,q in enumerate(qs):
    print(f'  [{i+1}/{len(qs)}] {q["id"]}  intent={q["intent"]}', flush=True)
    t0=time.time()
    ans, out_len = run_claude(q['question'], q['id'])
    elapsed = round(time.time()-t0, 2)
    print(f'    -> {elapsed}s  out={out_len}B  {str(ans)[:150]}...')
    if elapsed < 5 and ('EMPTY' in str(ans) or 'ERROR' in str(ans)):
        print(f'    WARN: very fast failure, likely CLI not found')
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
s=[r['judge_score'] for r in rs]; print(f'\nClaude Code Agent (real): {sum(s)/len(s):.3f}')
save_results('agent_claude_real', rs, {'model': 'opus', 'mode': 'claude-cli-file'})
print('Done')
