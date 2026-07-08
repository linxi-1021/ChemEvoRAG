"""Real Codex Agent driver — runs codex exec over MinerU markdown papers.
Each question gets its own Codex invocation with papers directory available.

Usage: python run_agent_codex_real.py --limit 5
"""
import io,os,sys,time,certifi,json,subprocess
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
print(f'Codex Agent (real): {len(qs)} questions, model={a.model}')

MD_DIR = Path('D:/Desktop/evo/ChemEvoRAG_Phase1-main/data/mineru_output')
paper_count = sum(1 for d in MD_DIR.iterdir() if d.is_dir() and (d/'extracted'/'full.md').exists())
print(f'  {paper_count} papers in {MD_DIR}')

def ask_codex(question):
    """Call codex exec non-interactively with papers dir accessible."""
    prompt = f"""You are a chemistry research assistant. Answer the following question by reading the relevant chemistry paper(s).

Papers are in directory: {MD_DIR}
Each subdirectory (1, 2, 3, ...) contains a parsed paper at extracted/full.md (markdown with tables, schemes, text).

Steps:
1. Use grep to search for relevant keywords across the markdown files to find the right paper.
2. Read the relevant sections (tables, experimental sections, compound names).
3. Answer concisely with specific evidence (cite paper number and table/section).
4. If you cannot find the answer, say so.

QUESTION: {question}"""

    codex_bin = r'C:\Users\ASUS\AppData\Roaming\npm\codex.cmd'
    try:
        result = subprocess.run(
            [codex_bin, 'exec', prompt,
             '--add-dir', str(MD_DIR),
             '-m', a.model,
             '--sandbox', 'workspace-write',
             '--ephemeral',
             '-o', '-'],
            capture_output=True, text=True, timeout=300,
            cwd=str(Path(__file__).parent),
            env={**os.environ, 'NO_COLOR': '1'},
            shell=True
        )
        output = result.stdout.strip()
        return output[-2000:] if len(output) > 2000 else output, output
    except subprocess.TimeoutExpired:
        return '[TIMEOUT]', ''
    except Exception as e:
        return f'[ERROR] {e}', ''

rs=[]
for i,q in enumerate(qs):
    print(f'  [{i+1}/{len(qs)}] {q["id"]}  intent={q["intent"]}', flush=True)
    t0=time.time()
    ans, raw = ask_codex(q['question'])
    elapsed = round(time.time()-t0, 2)
    print(f'    -> {elapsed}s  answer={str(ans)[:120]}...')
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
s=[r['judge_score'] for r in rs]; print(f'\nCodex Agent (real, {a.model}): {sum(s)/len(s):.3f}')
save_results(f'agent_codex_real_{a.model}', rs, {'model': a.model, 'mode': 'codex-cli'})
print('Done')
