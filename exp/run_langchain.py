"""LangChain RAG baseline - rag_langchain env"""
import io, json, os, sys, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
API_KEY="sk-gr-02f44cd3404411cd11cdd88da080685ac1c54986"
BASE_URL="https://endpoint.greatrouter.com"
MODEL="DeepSeek-V4-Flash"
os.environ["API_KEY"]=API_KEY; os.environ["BASE_URL"]=BASE_URL; os.environ["LLM_MODEL"]=MODEL
os.environ["OPENAI_API_KEY"]=API_KEY

from eval_utils import load_all_questions, save_results, PDF_DIR
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

questions = load_all_questions()
import argparse; p=argparse.ArgumentParser(); p.add_argument("--limit",type=int,default=0); a=p.parse_args()
if a.limit>0: questions=questions[:a.limit]
print(f"LangChain RAG: {len(questions)} questions")

# Extract PDFs
pdf_texts={}
for fp in sorted(Path(PDF_DIR).glob("*.pdf")):
    try:
        pages=PyPDFLoader(str(fp)).load()
        pdf_texts[fp.name]="\n\n".join(pp.page_content for pp in pages)
    except: pass
print(f"PDFs: {len(pdf_texts)}")

# Build index
splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
all_docs=[]
for fname,text in pdf_texts.items():
    for j,chunk in enumerate(splitter.split_text(text)):
        all_docs.append(Document(page_content=chunk,metadata={"source":fname}))
print(f"Chunks: {len(all_docs)}")

embeddings=OpenAIEmbeddings(model="text-embedding-3-small",openai_api_key=API_KEY,openai_api_base=BASE_URL)
vectordb=Chroma.from_documents(all_docs,embeddings)
retriever=vectordb.as_retriever(search_kwargs={"k":5})
llm=ChatOpenAI(model=MODEL,openai_api_key=API_KEY,openai_api_base=BASE_URL,temperature=0.0)
prompt=ChatPromptTemplate.from_template("Answer based ONLY on context.\n\nContext: {context}\n\nQuestion: {question}\nAnswer:")
chain=({"context":retriever|(lambda docs:"\n\n".join(d.page_content for d in docs)),"question":RunnablePassthrough()}|prompt|llm|StrOutputParser())

results=[]
for i,q in enumerate(questions):
    print(f"  [{i+1}/{len(questions)}] {q['id']}",flush=True)
    t0=time.time()
    try: ans=chain.invoke(q["question"])
    except Exception as e: ans=f"[ERROR] {e}"
    results.append({"question_id":q["id"],"paper_id":q["paper_id"],"paper_title":q["paper_title"],"intent":q["intent"],"question":q["question"],"ground_truth":q["ground_truth_answer"],"key_entities":q.get("key_entities",[]),"system_answer":str(ans).strip(),"elapsed_sec":round(time.time()-t0,2)})

sys.path.insert(0,'D:/Desktop/evo/ChemEvoRAG_Phase1-main/src')
from judge import judge_answer
print("Scoring...")
for i,r in enumerate(results):
    j=judge_answer(r["system_answer"],r["ground_truth"],r.get("key_entities",[]))
    r["judge_score"]=j.get("score",0); r["entities_found"]=j.get("key_entities_found",[]); r["entities_missing"]=j.get("key_entities_missing",[]); r["judge_reasoning"]=j.get("reasoning","")
    if (i+1)%30==0: print(f"  {i+1}/{len(results)}")
s=[r["judge_score"] for r in results]; print(f"LangChain RAG: {sum(s)/len(s):.3f}")
save_results("langchain", results, {"model":MODEL})
print("Done")
