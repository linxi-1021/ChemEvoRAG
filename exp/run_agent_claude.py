"""Real Claude Code Agent driver — runs claude -p over MinerU markdown papers.
Each question gets its own Claude invocation with papers directory available.

Usage: python run_agent_claude.py --limit 5
"""
import io,os,sys,time,certifi,json,subprocess,tempfile,shutil
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
os.environ['SSL_CERT_FILE']=certifi.where()

from eval_utils import load_all_questions, save_results
from pathlib import Path
import argparse

p=argparse.ArgumentParser(); p.add_argument('--limit',type=int,default=5)
p.add_argument('--model',type=str,default='opus')
a=p.parse_args()
qs=load_all_questions()
if a.limit<118: qs=qs[:a.limit]
print(f'Claude Code Agent: {len(qs)} questions, model={a.model}')

MD_DIR = Path('D:/Desktop/evo/ChemEvoRAG_Phase1-main/data/mineru_output')
# Verify papers exist
paper_count = sum(1 for d in MD_DIR.iterdir() if d.is_dir() and (d/'extracted'/'full.md').exists())
print(f'  {paper_count} papers in {MD_DIR}')

def ask_claude(question):
    """Call claude -p non-interactively with papers dir accessible."""
    prompt = f"""You are a chemistry research assistant. Answer the following question by reading the relevant chemistry paper(s) in the directory.

The directory {MD_DIR} contains subdirectories numbered 1, 2, 3, ... each containing a parsed chemistry paper at extracted/full.md (markdown format with tables, schemes, and text).

Steps:
1. First, use Grep to search for relevant keywords across the markdown files to find which paper contains the answer.
2. Then, use Read to examine the relevant sections of that paper (especially tables, experimental sections, compound names).
3. Answer the question concisely with specific evidence (cite the paper number, table number, and line content).
4. If you cannot find the answer, say so explicitly.

QUESTION: {question}

Reply with your answer. Start your final answer with "ANSWER: " on a new line."""

    claude_bin = r'C:\Users\ASUS\AppData\Roaming\npm\claude.cmd'
    try:
        result = subprocess.run(
            [claude_bin, '-p', prompt,
             '--add-dir', str(MD_DIR),
             '--model', a.model,
             '--max-budget-usd', '1',
             '--output-format', 'text'],
            capture_output=True, text=True, timeout=300,
            cwd=str(Path(__file__).parent),
            env={**os.environ, 'NO_COLOR': '1'},
            shell=True
        )
        output = result.stdout.strip()
        # Extract ANSWER: section
        if 'ANSWER:' in output:
            answer = output.split('ANSWER:')[-1].strip()
        else:
            answer = output[-1000:] if len(output) > 1000 else output
        return answer, output
    except subprocess.TimeoutExpired:
        return '[TIMEOUT]', ''
    except Exception as e:
        return f'[ERROR] {e}', ''

rs=[]
for i,q in enumerate(qs):
    print(f'  [{i+1}/{len(qs)}] {q["id"]}  intent={q["intent"]}', flush=True)
    t0=time.time()
    ans, raw = ask_claude(q['question'])
    elapsed = round(time.time()-t0, 2)
    print(f'    -> {elapsed}s  answer={ans[:100]}...')
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
s=[r['judge_score'] for r in rs]; print(f'\nClaude Code Agent ({a.model}): {sum(s)/len(s):.3f}')
save_results(f'agent_claude_{a.model}', rs, {'model': a.model, 'mode': 'claude-cli'})
print('Done')
