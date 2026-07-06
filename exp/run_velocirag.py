"""VelociRAG baseline — ONNX-powered retrieval + external LLM for answer generation.
Run: C:/Users/ASUS/.conda/envs/rag_velocirag/python.exe run_velocirag.py --limit 118

Note: VelociRAG is retrieval-only (no answer generation).
Chunks are retrieved via ONNX embeddings, then DeepSeek generates answers."""
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
import argparse, shutil, tempfile

p=argparse.ArgumentParser(); p.add_argument('--limit',type=int,default=118); a=p.parse_args()
qs=load_all_questions()
if a.limit<118: qs=qs[:a.limit]
print(f'VelociRAG: {len(qs)} questions')

# VelociRAG ingests .md files (uses header-aware markdown chunker)
# Convert PDF chunks to .md files in temp dir
print('Preparing documents (PDF -> markdown chunks)...')
splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
tmpdir=tempfile.mkdtemp(prefix='velocirag_')
try:
    chunk_count=0
    for f in sorted(Path(PDF_DIR).glob('*.pdf')):
        try:
            t='\n'.join(p.page_content for p in PyPDFLoader(str(f)).load())
            if t.strip():
                for ci,c in enumerate(splitter.split_text(t)):
                    cpath=os.path.join(tmpdir,f'{f.stem}_c{ci:04d}.md')
                    with open(cpath,'w',encoding='utf-8') as cf:
                        # Add markdown heading for header-aware chunking
                        cf.write(f'# {f.stem} (chunk {ci})\n\n{c}')
                    chunk_count+=1
        except: pass
    print(f'  {chunk_count} .md chunks -> {tmpdir}')

    # Build VelociRAG index (ONNX embeddings, all-MiniLM-L6-v2)
    print('Building VelociRAG index (ONNX all-MiniLM-L6-v2)...')
    from velocirag import Embedder, VectorStore, Searcher
    embedder = Embedder()  # downloads ~23MB ONNX model on first use
    vdb_path = os.path.join(tmpdir, '_velocirag_vdb')
    store = VectorStore(vdb_path, embedder)
    store.add_directory(tmpdir)  # chunk + embed + index all .md files
    searcher = Searcher(store, embedder)
    print(f'  Index built (ONNX, CPU, <1GB RAM)')

    # External LLM for answer generation
    client=OpenAI(api_key=K,base_url=B)

    def query(q_question,k=5):
        # Step 1: VelociRAG retrieval (16ms avg warm, 4-layer fusion potential)
        vresults = searcher.search(q_question, limit=k)
        ctx_parts=[]
        for vr in vresults.get('results',[]):
            ctx_parts.append(vr.get('content','') or vr.get('text','') or str(vr))
        ctx='\n\n'.join(ctx_parts)
        # Step 2: LLM answer generation
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

finally:
    shutil.rmtree(tmpdir,ignore_errors=True)

sys.path.insert(0,'D:/Desktop/evo/ChemEvoRAG_Phase1-main/src')
from judge import judge_answer
print('Scoring...')
for i,r in enumerate(rs):
    j=judge_answer(r['system_answer'],r['ground_truth'],r.get('key_entities',[]))
    r['judge_score']=j.get('score',0); r['entities_found']=j.get('key_entities_found',[]); r['entities_missing']=j.get('key_entities_missing',[]); r['judge_reasoning']=j.get('reasoning','')
    if (i+1)%25==0: print(f'  {i+1}/{len(rs)}')
s=[r['judge_score'] for r in rs]; print(f'VelociRAG: {sum(s)/len(s):.3f}')
save_results('velocirag',rs,{'model':M,'retrieval':'VelociRAG-ONNX','embedder':'all-MiniLM-L6-v2'})
print('Done')
