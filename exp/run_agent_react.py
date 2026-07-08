"""ReAct Agent baseline — Multi-round retrieval over basic RAG (same embedding index as Haystack).
Replicates the ChemEvoRAG ReAct loop but uses simple chunk embeddings instead of
structured evidence cards.

Run: C:/Users/ASUS/.conda/envs/rag_paperqa/python.exe run_agent_react.py --limit 118
"""
import io,os,sys,time,certifi,json,re
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

p=argparse.ArgumentParser(); p.add_argument('--limit',type=int,default=118); p.add_argument('--max-rounds',type=int,default=3)
a=p.parse_args()
qs=load_all_questions()
if a.limit<118: qs=qs[:a.limit]
print(f'ReAct Agent: {len(qs)} questions, max {a.max_rounds} rounds')

# Standard chunked RAG (same as Haystack)
print('Indexing PDFs...')
splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
chunks=[]  # (filename, chunk_text)
for f in sorted(Path(PDF_DIR).glob('*.pdf')):
    try:
        t='\n'.join(p.page_content for p in PyPDFLoader(str(f)).load())
        if t.strip():
            for c in splitter.split_text(t): chunks.append((f.name,c))
    except: pass
print(f'  {len(chunks)} chunks')

client=OpenAI(api_key=K,base_url=B)
print('Embedding...')
embs=[]
for i in range(0,len(chunks),20):
    batch=[c[1] for c in chunks[i:i+20]]
    r=client.embeddings.create(model='text-embedding-3-small',input=batch)
    embs.extend([d.embedding for d in r.data])
    if (i+20)%200==0: print(f'  {min(i+20,len(chunks))}/{len(chunks)}')
emb=np.array(embs); print(f'  {len(emb)} embeddings')

def search(query, k=8):
    r=client.embeddings.create(model='text-embedding-3-small',input=[query])
    qe=np.array(r.data[0].embedding)
    sim=np.dot(emb,qe)/(np.linalg.norm(emb,axis=1)*np.linalg.norm(qe)+1e-8)
    top=np.argsort(sim)[-k:][::-1]
    results=[]
    for idx in top:
        results.append({'id': f'chunk_{idx}', 'source': chunks[idx][0], 'score': float(sim[idx]),
                        'text': chunks[idx][1][:500]})
    return results

ASSESS_PROMPT = """You are a chemistry research assistant. Given a question and retrieved evidence chunks, assess if the evidence is sufficient to answer.

Reply in JSON: {"sufficient": true/false, "reason": "...", "refined_query": "..." (if insufficient)}
- sufficient=true only if the evidence contains the specific information asked for
- If the question asks "what does label X refer to?" and the evidence doesn't define X, it's insufficient
- refined_query should rephrase the question to find the missing information (e.g. focus on the specific label, try alternate spellings)
"""

ANSWER_PROMPT = """You are a chemistry research assistant. Answer based ONLY on the provided context. If not found, say so.
Reply in JSON: {"answer": "...", "confidence": 0.X}"""

def react_search(question):
    """ReAct loop: retrieve → assess → refine → retrieve"""
    all_chunks=[]
    seen=set()
    current_q=question
    trace=[]

    for rd in range(a.max_rounds):
        results=search(current_q, k=8)
        new_chunks=[]
        for r in results:
            if r['id'] not in seen:
                seen.add(r['id'])
                new_chunks.append(r)
        all_chunks.extend(new_chunks)
        trace.append(f'Round {rd+1}: {len(new_chunks)} new chunks from "{current_q[:60]}"')

        # Assess (skip on last round)
        if rd < a.max_rounds-1:
            try:
                ctx='\n'.join(f"[{c['id']}] ({c['source']}) {c['text'][:300]}" for c in all_chunks[:10])
                resp=client.chat.completions.create(model=M, messages=[
                    {'role':'system','content':ASSESS_PROMPT},
                    {'role':'user','content':f'QUESTION: {question}\n\nEVIDENCE:\n{ctx}'}],
                    temperature=0.0, max_tokens=512)
                raw=resp.choices[0].message.content.strip()
                if raw.startswith('```'): raw=raw.split('```')[1].split('```')[0].strip()
                if raw.startswith('json'): raw=raw[4:]
                assessment=json.loads(raw)
                if assessment.get('sufficient'):
                    trace.append(f'  -> sufficient: {assessment.get("reason","")[:100]}')
                    break
                refined=assessment.get('refined_query','')
                if refined:
                    current_q=refined
                    trace.append(f'  -> refined: {refined[:80]}...')
            except Exception as e:
                trace.append(f'  -> assessment error: {e}')

    # Generate answer from accumulated evidence
    ctx='\n'.join(f"[{c['id']}] {c['text'][:400]}" for c in all_chunks[:10])
    resp=client.chat.completions.create(model=M, messages=[
        {'role':'system','content':ANSWER_PROMPT},
        {'role':'user','content':f'Context:\n{ctx}\n\nQuestion: {question}\nAnswer:'}],
        temperature=0.0, max_tokens=512)
    ans_raw=resp.choices[0].message.content.strip()
    try:
        if ans_raw.startswith('```'): ans_raw=ans_raw.split('```')[1].split('```')[0].strip()
        if ans_raw.startswith('json'): ans_raw=ans_raw[4:]
        parsed=json.loads(ans_raw)
        answer=parsed.get('answer',ans_raw)
    except:
        answer=ans_raw
    return answer, '\n'.join(trace)

rs=[]
for i,q in enumerate(qs):
    print(f'  [{i+1}/{len(qs)}] {q["id"]}  intent={q["intent"]}',flush=True)
    t0=time.time()
    try:
        ans,trace=react_search(q['question'])
    except Exception as e:
        ans=f'[ERROR] {e}'; trace=''
    elapsed=round(time.time()-t0,2)
    rounds=trace.count('Round')
    print(f'    -> {rounds} rounds, {elapsed}s')
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
    if (i+1)%25==0: print(f'  {i+1}/{len(rs)}')
s=[r['judge_score'] for r in rs]; print(f'\nReAct Agent: {sum(s)/len(s):.3f}')
save_results('agent_react',rs,{'model':M,'max_rounds':a.max_rounds})
print('Done')
