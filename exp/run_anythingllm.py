"""AnythingLLM baseline — REST API RAG (48k stars).
Requires AnythingLLM Docker server running:
  docker pull mintplexlabs/anythingllm
  docker run -d -p 3001:3001 --cap-add SYS_ADMIN mintplexlabs/anythingllm

Then configure via http://localhost:3001 (one-time):
  1. Set LLM provider > Generic OpenAI > URL=https://endpoint.greatrouter.com, Key=sk-...
  2. Set Embedder > OpenAI > same endpoint
  3. Set Vector DB > LanceDB (built-in)
  4. Generate API key in Settings > Developer API

Run: C:/Users/ASUS/.conda/envs/rag_direct/python.exe run_anythingllm.py --limit 118
"""
import io,os,sys,time,certifi,json
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
os.environ['SSL_CERT_FILE']=certifi.where()

K='sk-gr-02f44cd3404411cd11cdd88da080685ac1c54986'; B='https://endpoint.greatrouter.com'; M='DeepSeek-V3'
os.environ['OPENAI_API_KEY']=K; os.environ['API_KEY']=K; os.environ['BASE_URL']=B; os.environ['LLM_MODEL']=M

from eval_utils import load_all_questions, save_results, PDF_DIR
from pathlib import Path
import argparse, requests

p=argparse.ArgumentParser(); p.add_argument('--limit',type=int,default=118)
p.add_argument('--url',type=str,default=os.environ.get('ANYTHINGLLM_URL','http://localhost:3001'))
p.add_argument('--api-key',type=str,default=os.environ.get('ANYTHINGLLM_API_KEY',''))
p.add_argument('--workspace',type=str,default='chem-eval')
a=p.parse_args()

if not a.api_key:
    print('ERROR: Set ANYTHINGLLM_API_KEY env var or pass --api-key')
    print('Get key from AnythingLLM Settings -> Developer API -> Generate API Key')
    sys.exit(1)

qs=load_all_questions()
if a.limit<118: qs=qs[:a.limit]
print(f'AnythingLLM: {len(qs)} questions')
print(f'Server: {a.url}  Workspace: {a.workspace}')

BASE=f'{a.url}/api/v1'
auth={'Authorization':f'Bearer {a.api_key}','Content-Type':'application/json'}

# Check server
try:
    r=requests.get(f'{BASE}/auth',headers=auth,timeout=10)
    print(f'Auth check: {r.status_code}')
except Exception as e:
    print(f'ERROR: Cannot reach {a.url}: {e}')
    sys.exit(1)

# Create workspace (idempotent)
print(f'Ensuring workspace "{a.workspace}"...')
ws_slug=a.workspace
try:
    r=requests.post(f'{BASE}/workspace',headers=auth,json={'name':a.workspace},timeout=30)
    if r.status_code==200:
        ws_slug=r.json().get('workspace',{}).get('slug',a.workspace)
        print(f'  Created: {ws_slug}')
    else:
        r2=requests.get(f'{BASE}/workspaces',headers=auth,timeout=30)
        found=None
        for w in r2.json().get('workspaces',[]):
            if w.get('slug')==a.workspace or w.get('name')==a.workspace:
                found=w; break
        if found:
            ws_slug=found.get('slug',a.workspace)
            print(f'  Found existing: {ws_slug}')
        else:
            print(f'  Create returned {r.status_code}: {r.text[:200]}')
except Exception as e:
    print(f'  Workspace: {e}')

# Upload & embed PDFs
print('Uploading PDFs...')
pdf_list=sorted(Path(PDF_DIR).glob('*.pdf'))
doc_ids=[]
for i,f in enumerate(pdf_list):
    try:
        with open(f,'rb') as fh:
            ur=requests.post(f'{BASE}/document/upload',
                headers={'Authorization':f'Bearer {a.api_key}'},
                files={'file':(f.name,fh,'application/pdf')},
                data={'addToWorkspaces':ws_slug},
                timeout=120)
        if ur.status_code==200:
            jd=ur.json()
            did=jd.get('document_id') or jd.get('document',{}).get('id','')
            if did: doc_ids.append(did)
        if (i+1)%5==0: print(f'  Uploaded {i+1}/{len(pdf_list)}')
    except Exception as e:
        print(f'  x {f.name}: {e}')

print(f'  {len(doc_ids)}/{len(pdf_list)} uploaded')

# Wait for embedding (poll)
if doc_ids:
    print('Waiting for embedding...')
    time.sleep(10)  # initial buffer
    for attempt in range(12):
        try:
            sr=requests.post(f'{BASE}/workspace/{ws_slug}/chat',
                headers=auth,json={'message':'ping','mode':'query'},timeout=30)
            if sr.status_code==200: print('  Ready'); break
        except: pass
        time.sleep(5)

# Query
print('Querying...')
rs=[]
for i,q in enumerate(qs):
    print(f'  [{i+1}/{len(qs)}] {q["id"]}',flush=True)
    t0=time.time()
    try:
        cr=requests.post(f'{BASE}/workspace/{ws_slug}/chat',
            headers=auth,json={'message':q['question'],'mode':'query'},timeout=120)
        if cr.status_code==200:
            ans=cr.json().get('textResponse','') or json.dumps(cr.json())
        else:
            ans=f'[API {cr.status_code}] {cr.text[:300]}'
    except Exception as e:
        ans=f'[ERROR] {e}'
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
s=[r['judge_score'] for r in rs]; print(f'AnythingLLM: {sum(s)/len(s):.3f}')
save_results('anythingllm',rs,{'model':M,'server':a.url,'workspace':ws_slug})
print('Done')
