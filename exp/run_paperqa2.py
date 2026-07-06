"""PaperQA2 full pipeline - rag_paperqa env"""
import asyncio, io, json, os, sys, time, certifi
os.environ["SSL_CERT_FILE"] = certifi.where()
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
API_KEY="sk-gr-02f44cd3404411cd11cdd88da080685ac1c54986"
BASE_URL="https://endpoint.greatrouter.com"
MODEL="DeepSeek-V4-Flash"
os.environ["API_KEY"]=API_KEY; os.environ["BASE_URL"]=BASE_URL; os.environ["LLM_MODEL"]=MODEL
os.environ["OPENAI_API_KEY"]=API_KEY

# LiteLLM patch
import litellm
def _fix(kw):
    m=kw.get('model','')
    if 'embed' in str(m).lower(): kw['model']='openai/text-embedding-3-small'
    elif 'gpt' in str(m).lower(): kw['model']=f'openai/{MODEL}'
    kw.setdefault('api_base',BASE_URL); kw.setdefault('api_key',API_KEY)
    return kw
_c=litellm.completion; litellm.completion=lambda *a,**kw:_c(*a,**_fix(kw))
_ac=litellm.acompletion; litellm.acompletion=lambda *a,**kw:_ac(*a,**_fix(kw))
_e=litellm.embedding
def _pe(*a,**kw): kw=_fix(kw); kw['model']='openai/text-embedding-3-small'; return _e(*a,**kw)
litellm.embedding=_pe
_ae=litellm.aembedding
async def _pae(*a,**kw): kw=_fix(kw); kw['model']='openai/text-embedding-3-small'; return await _ae(*a,**kw)
litellm.aembedding=_pae

from eval_utils import load_all_questions, save_results, PDF_DIR
from paperqa import Settings, Docs
from paperqa.types import Doc, Text
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from lmi import embedding_model_factory
from pathlib import Path

questions = load_all_questions()
import argparse; p=argparse.ArgumentParser(); p.add_argument("--limit",type=int,default=0); a=p.parse_args()
if a.limit>0: questions=questions[:a.limit]
print(f"PaperQA2: {len(questions)} questions")

# Extract PDFs
pdf_texts={}
for fp in sorted(Path(PDF_DIR).glob("*.pdf")):
    try:
        pages=PyPDFLoader(str(fp)).load()
        pdf_texts[fp.stem]="\n\n".join(pp.page_content for pp in pages)
    except: pass

# Build index
llm_cfg={"model_list":[{"model_name":"text-embedding-3-small","litellm_params":{"model":"openai/text-embedding-3-small","api_base":BASE_URL,"api_key":API_KEY}}]}
emb_model=embedding_model_factory("text-embedding-3-small",**llm_cfg)
settings=Settings(llm=f'openai/{MODEL}',summary_llm=f'openai/{MODEL}',embedding=f'openai/text-embedding-3-small',temperature=0.0)
settings.parsing.use_doc_details=False

splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
docs=Docs()
for i,(name,text) in enumerate(pdf_texts.items()):
    if not text.strip(): continue
    chunks=splitter.split_text(text)
    pqa_texts=[Text(text=c,name=f"{name}_c{j}",doc=Doc(docname=name,citation=f"Paper: {name}",dockey=f"pqa_{i}_{j}")) for j,c in enumerate(chunks)]
    try:
        loop=asyncio.get_event_loop()
        loop.run_until_complete(docs.aadd_texts(pqa_texts,Doc(docname=name,citation=f"Paper: {name}",dockey=f"pqa_{i}"),embedding_model=emb_model))
    except: pass
print(f"  {len(docs.docs)} docs, {len(docs.texts)} texts")

# Query
async def run():
    results=[]
    for i,q in enumerate(questions):
        print(f"  [{i+1}/{len(questions)}] {q['id']}",flush=True)
        t0=time.time()
        try:
            result=await docs.aquery(q["question"],settings=settings)
            ans=result.answer if hasattr(result,'answer') else str(result)
        except Exception as e: ans=f"[ERROR] {e}"
        results.append({"question_id":q["id"],"paper_id":q["paper_id"],"paper_title":q["paper_title"],"intent":q["intent"],"question":q["question"],"ground_truth":q["ground_truth_answer"],"key_entities":q.get("key_entities",[]),"system_answer":str(ans).strip(),"elapsed_sec":round(time.time()-t0,2)})
    return results

loop2=asyncio.new_event_loop(); asyncio.set_event_loop(loop2)
results=loop2.run_until_complete(run())

sys.path.insert(0,'D:/Desktop/evo/ChemEvoRAG_Phase1-main/src')
from judge import judge_answer
print("Scoring...")
for i,r in enumerate(results):
    j=judge_answer(r["system_answer"],r["ground_truth"],r.get("key_entities",[]))
    r["judge_score"]=j.get("score",0); r["entities_found"]=j.get("key_entities_found",[]); r["entities_missing"]=j.get("key_entities_missing",[]); r["judge_reasoning"]=j.get("reasoning","")
    if (i+1)%30==0: print(f"  {i+1}/{len(results)}")
s=[r["judge_score"] for r in results]; print(f"PaperQA2: {sum(s)/len(s):.3f}")
save_results("paperqa2", results, {"model":MODEL})
print("Done")
