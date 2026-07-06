"""Haystack baseline — Pipeline-based RAG. Run: conda activate rag_paperqa && python run_haystack.py --limit 118"""
import io,os,sys,time,certifi
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
os.environ['SSL_CERT_FILE']=certifi.where()

K='sk-gr-02f44cd3404411cd11cdd88da080685ac1c54986'; B='https://endpoint.greatrouter.com'; M='DeepSeek-V3'
os.environ['OPENAI_API_KEY']=K; os.environ['API_KEY']=K; os.environ['BASE_URL']=B; os.environ['LLM_MODEL']=M

from eval_utils import load_all_questions, save_results, PDF_DIR
from pathlib import Path
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
import argparse

p=argparse.ArgumentParser(); p.add_argument('--limit',type=int,default=118); a=p.parse_args()
qs=load_all_questions()
if a.limit<118: qs=qs[:a.limit]
print(f'Haystack: {len(qs)} questions')

# Extract & chunk PDFs
print('Loading PDFs...')
splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
chunks=[]
for f in sorted(Path(PDF_DIR).glob('*.pdf')):
    try:
        t='\n'.join(p.page_content for p in PyPDFLoader(str(f)).load())
        if t.strip(): chunks.extend([(f.name,c) for c in splitter.split_text(t)])
    except: pass
print(f'  {len(chunks)} chunks')

# Embed
print('Embedding...')
client=OpenAI(api_key=K,base_url=B,timeout=60.0,max_retries=3)
emb=[]
for i in range(0,len(chunks),20):
    batch=[c[1] for c in chunks[i:i+20]]
    r=client.embeddings.create(model='text-embedding-3-small',input=batch)
    emb.extend([d.embedding for d in r.data])
    if (i+20)%200==0: print(f'  {min(i+20,len(chunks))}/{len(chunks)}')
emb=np.array(emb); print(f'  {len(emb)} embeddings')

# Query
def query(q_question,k=5):
    r=client.embeddings.create(model='text-embedding-3-small',input=[q_question],timeout=60.0)
    qe=np.array(r.data[0].embedding)
    sim=np.dot(emb,qe)/(np.linalg.norm(emb,axis=1)*np.linalg.norm(qe)+1e-8)
    top=np.argsort(sim)[-k:][::-1]
    ctx='\n\n'.join(chunks[i][1] for i in top)
    resp=client.chat.completions.create(model=M,messages=[
        {'role':'system','content':'You are a chemistry research assistant. Answer based ONLY on the provided context. If not found, say so.'},
        {'role':'user','content':f'Context:\n{ctx}\n\nQuestion: {q_question}\nAnswer:'}],
        temperature=0.0,max_tokens=512,timeout=60.0)
    return resp.choices[0].message.content

rs=[]
for i,q in enumerate(qs):
    print(f'  [{i+1}/{len(qs)}] {q["id"]}',flush=True)
    t0=time.time()
    try: ans=query(q['question'])
    except Exception as e: ans=f'[ERROR] {e}'
    rs.append({'question_id':q['id'],'paper_id':q['paper_id'],'paper_title':q['paper_title'],
        'intent':q['intent'],'question':q['question'],'ground_truth':q['ground_truth_answer'],
        'key_entities':q.get('key_entities',[]),'system_answer':str(ans).strip(),
        'elapsed_sec':round(time.time()-t0,2)})

sys.path.insert(0,'D:/Desktop/evo/ChemEvoRAG_Phase1-main/src')
from judge import judge_answer
print('Scoring...')
for i,r in enumerate(rs):
    j=judge_answer(r['system_answer'],r['ground_truth'],r.get('key_entities',[]))
    r['judge_score']=j.get('score',0); r['entities_found']=j.get('key_entities_found',[]); r['entities_missing']=j.get('key_entities_missing',[]); r['judge_reasoning']=j.get('reasoning','')
    if (i+1)%25==0: print(f'  {i+1}/{len(rs)}')
s=[r['judge_score'] for r in rs]; print(f'Haystack: {sum(s)/len(s):.3f}')
save_results('haystack',rs,{'model':M})
print('Done')
