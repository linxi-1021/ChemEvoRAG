"""LlamaIndex baseline — Data-indexing RAG paradigm. Run: conda activate rag_llamaindex && python run_llamaindex.py --limit 118"""
import io,os,sys,time
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')

K='sk-gr-02f44cd3404411cd11cdd88da080685ac1c54986'; B='https://endpoint.greatrouter.com'; M='DeepSeek-V3'
os.environ['OPENAI_API_KEY']=K; os.environ['API_KEY']=K; os.environ['BASE_URL']=B; os.environ['LLM_MODEL']=M

from eval_utils import load_all_questions, save_results, PDF_DIR
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex, Document as LIDoc, Settings
from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
import argparse

p=argparse.ArgumentParser(); p.add_argument('--limit',type=int,default=118); a=p.parse_args()
qs=load_all_questions()
if a.limit<118: qs=qs[:a.limit]
print(f'LlamaIndex: {len(qs)} questions')

# Load PDFs
print('Loading PDFs...')
pdf_docs=[]
for f in sorted(Path(PDF_DIR).glob('*.pdf')):
    try:
        t='\n'.join(p.page_content for p in PyPDFLoader(str(f)).load())
        if t.strip(): pdf_docs.append(LIDoc(text=t,metadata={'file_name':f.name}))
    except Exception as e: print(f'  x {f.name}: {e}')
print(f'  {len(pdf_docs)} docs')

# Use OpenAILike (bypasses model name whitelist)
Settings.llm = OpenAILike(model=M, api_base=B, api_key=K, temperature=0.0, is_chat_model=True, max_tokens=1024)
Settings.embed_model = OpenAIEmbedding(model='text-embedding-3-small', api_base=B, api_key=K)
Settings.chunk_size = 1000; Settings.chunk_overlap = 200

# Build
print('Building index...')
index = VectorStoreIndex.from_documents(pdf_docs, show_progress=True)
engine = index.as_query_engine()

# Query
rs=[]
for i,q in enumerate(qs):
    print(f'  [{i+1}/{len(qs)}] {q["id"]}',flush=True)
    t0=time.time()
    try: ans=str(engine.query(q['question']))
    except Exception as e: ans=f'[ERROR] {e}'
    rs.append({'question_id':q['id'],'paper_id':q['paper_id'],'paper_title':q['paper_title'],
        'intent':q['intent'],'question':q['question'],'ground_truth':q['ground_truth_answer'],
        'key_entities':q.get('key_entities',[]),'system_answer':ans.strip(),
        'elapsed_sec':round(time.time()-t0,2)})

# Score
sys.path.insert(0,'D:/Desktop/evo/ChemEvoRAG_Phase1-main/src')
from judge import judge_answer
print('Scoring...')
for i,r in enumerate(rs):
    j=judge_answer(r['system_answer'],r['ground_truth'],r.get('key_entities',[]))
    r['judge_score']=j.get('score',0); r['entities_found']=j.get('key_entities_found',[]); r['entities_missing']=j.get('key_entities_missing',[]); r['judge_reasoning']=j.get('reasoning','')
    if (i+1)%25==0: print(f'  {i+1}/{len(rs)}')
s=[r['judge_score'] for r in rs]; print(f'LlamaIndex: {sum(s)/len(s):.3f}')
save_results('llamaindex',rs,{'model':M})
print('Done')
