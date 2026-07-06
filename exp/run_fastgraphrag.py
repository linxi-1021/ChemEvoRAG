"""FastGraphRAG baseline — Graph-enhanced RAG (0.0.5).
Run: conda activate rag_fastgraphrag && python run_fastgraphrag.py --limit 118"""
import io, os, sys, time, shutil, tempfile
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

K = 'sk-gr-02f44cd3404411cd11cdd88da080685ac1c54986'
B = 'https://endpoint.greatrouter.com'
M = 'DeepSeek-V3'  # DeepSeek-V4-Flash returns 500 on GreatRouter as of 2026-07
os.environ['OPENAI_API_KEY'] = K
os.environ['API_KEY'] = K
os.environ['BASE_URL'] = B
os.environ['LLM_MODEL'] = M

from eval_utils import load_all_questions, save_results, PDF_DIR
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import argparse

p = argparse.ArgumentParser()
p.add_argument('--limit', type=int, default=118)
a = p.parse_args()
qs = load_all_questions()
if a.limit < 118:
    qs = qs[:a.limit]
print(f'FastGraphRAG: {len(qs)} questions')

# Extract PDFs into text
print('Loading PDFs...')
pdf_texts = {}
for f in sorted(Path(PDF_DIR).glob('*.pdf')):
    try:
        t = '\n'.join(p.page_content for p in PyPDFLoader(str(f)).load())
        if t.strip():
            pdf_texts[f.stem] = t
    except Exception as e:
        print(f'  x {f.name}: {e}')
print(f'  {len(pdf_texts)} PDFs')

# Chunk PDFs
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_chunks = []
for name, text in pdf_texts.items():
    for ci, chunk in enumerate(splitter.split_text(text)):
        all_chunks.append((name, chunk))
print(f'  {len(all_chunks)} chunks')

# Build FastGraphRAG index (persisted to temp dir)
from fast_graphrag import GraphRAG, QueryParam

tmpdir = tempfile.mkdtemp(prefix='fastgraphrag_')
working_dir = os.path.join(tmpdir, 'graphrag_index')

try:
    print('Building graph index...')
    grag = GraphRAG(
        working_dir=working_dir,
        domain='chemistry research papers — organic synthesis, catalysis, materials',
        example_queries='What catalyst was used? What is the yield? What compound is labeled 1a?',
        entity_types=['CATALYST', 'REAGENT', 'SOLVENT', 'PRODUCT', 'COMPOUND',
                      'REACTION', 'CONDITION', 'TEMPERATURE', 'YIELD', 'METHOD'],
    )

    # Insert all chunks as raw text (FastGraphRAG auto-extracts entities/relations)
    contents = [f'[Paper: {name}] {chunk}' for name, chunk in all_chunks]
    grag.insert(contents, show_progress=True)
    print(f'  Index built')

    # Query
    print('Querying...')
    rs = []
    for i, q in enumerate(qs):
        print(f'  [{i+1}/{len(qs)}] {q["id"]}', flush=True)
        t0 = time.time()
        try:
            resp = grag.query(q['question'])
            if isinstance(resp, str):
                ans = resp
            elif hasattr(resp, 'response'):
                ans = resp.response
            elif isinstance(resp, dict):
                ans = resp.get('response', '') or resp.get('answer', '') or str(resp)[:2000]
            else:
                ans = str(resp)[:2000]
        except Exception as e:
            ans = f'[ERROR] {e}'
        rs.append({
            'question_id': q['id'], 'paper_id': q['paper_id'], 'paper_title': q['paper_title'],
            'intent': q['intent'], 'question': q['question'],
            'ground_truth': q['ground_truth_answer'],
            'key_entities': q.get('key_entities', []),
            'system_answer': str(ans).strip(),
            'elapsed_sec': round(time.time() - t0, 2),
        })

finally:
    shutil.rmtree(tmpdir, ignore_errors=True)

# Score
sys.path.insert(0, 'D:/Desktop/evo/ChemEvoRAG_Phase1-main/src')
from judge import judge_answer
print('Scoring...')
for i, r in enumerate(rs):
    j = judge_answer(r['system_answer'], r['ground_truth'], r.get('key_entities', []))
    r['judge_score'] = j.get('score', 0)
    r['entities_found'] = j.get('key_entities_found', [])
    r['entities_missing'] = j.get('key_entities_missing', [])
    r['judge_reasoning'] = j.get('reasoning', '')
    if (i + 1) % 25 == 0:
        print(f'  {i+1}/{len(rs)}')
s = [r['judge_score'] for r in rs]
print(f'FastGraphRAG: {sum(s)/len(s):.3f}')
save_results('fastgraphrag', rs, {'model': M})
print('Done')
