"""R2R (RAG-to-Riches) baseline — R2R-style RAG via LiteLLM embedding + retrieval + LLM (5k stars).
R2R uses LiteLLM for embeddings and completions. This replicates the R2R retrieval pipeline:
  - text-embedding-3-small for chunk embeddings (same as R2R default)
  - Cosine similarity search (same as R2R vector search)
  - DeepSeek-V3 for answer generation

Run: C:/Users/ASUS/.conda/envs/rag_r2r/python.exe run_r2r.py --limit 118"""
import io,os,sys,time,certifi
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
os.environ['SSL_CERT_FILE']=certifi.where()

K='sk-gr-02f44cd3404411cd11cdd88da080685ac1c54986'; B='https://endpoint.greatrouter.com'; M='DeepSeek-V3'
os.environ['OPENAI_API_KEY']=K; os.environ['API_KEY']=K; os.environ['BASE_URL']=B; os.environ['LLM_MODEL']=M

from eval_utils import load_all_questions, save_results, PDF_DIR
from pathlib import Path
from openai import OpenAI
from pypdf import PdfReader
import numpy as np
import argparse

# Simple text splitter (no langchain dep)
def simple_split(text, chunk_size=512, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap if end < len(text) else end
    return chunks

p=argparse.ArgumentParser(); p.add_argument('--limit',type=int,default=118); a=p.parse_args()
qs=load_all_questions()
if a.limit<118: qs=qs[:a.limit]
print(f'R2R-style RAG: {len(qs)} questions')

# R2R default: chunk_size=512, chunk_overlap=50
print('Loading & chunking PDFs (R2R defaults: chunk=512, overlap=50)...')
chunks=[]  # (filename, chunk_text)
for f in sorted(Path(PDF_DIR).glob('*.pdf')):
    try:
        reader = PdfReader(str(f))
        t='\n'.join(p.extract_text() or '' for p in reader.pages)
        if t.strip():
            for c in simple_split(t, chunk_size=512, overlap=50):
                chunks.append((f.name,c))
    except: pass
print(f'  {len(chunks)} chunks')

# Embed (OpenAI text-embedding-3-small, same as R2R default)
print('Embedding (text-embedding-3-small, R2R default)...')
client=OpenAI(api_key=K,base_url=B)
embeddings=[]
for i in range(0,len(chunks),20):
    batch=[c[1] for c in chunks[i:i+20]]
    r=client.embeddings.create(model='text-embedding-3-small',input=batch)
    embeddings.extend([d.embedding for d in r.data])
    if (i+20)%200==0: print(f'  {min(i+20,len(chunks))}/{len(chunks)}')
emb=np.array(embeddings); print(f'  {len(emb)} embeddings ({emb.shape[1]}d)')

# R2R-style RAG: vector search + LLM generation
def query(q_question,k=5):
    r=client.embeddings.create(model='text-embedding-3-small',input=[q_question])
    qe=np.array(r.data[0].embedding)
    sim=np.dot(emb,qe)/(np.linalg.norm(emb,axis=1)*np.linalg.norm(qe)+1e-8)
    top=np.argsort(sim)[-k:][::-1]
    ctx='\n\n'.join(chunks[i][1] for i in top)
    resp=client.chat.completions.create(model=M,messages=[
        {'role':'system','content':'You are a chemistry research assistant. Answer based ONLY on the provided context. If not found, say so.'},
        {'role':'user','content':f'Context:\n{ctx}\n\nQuestion: {q_question}\nAnswer:'}],
        temperature=0.0,max_tokens=512)
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
s=[r['judge_score'] for r in rs]; print(f'R2R-style: {sum(s)/len(s):.3f}')
save_results('r2r',rs,{'model':M,'method':'r2r-style','embedder':'text-embedding-3-small','chunk_size':512,'chunk_overlap':50})
print('Done')
