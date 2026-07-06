"""Direct LLM baseline - rag_direct env"""
import io, json, os, sys, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
API_KEY="sk-gr-02f44cd3404411cd11cdd88da080685ac1c54986"
BASE_URL="https://endpoint.greatrouter.com"
MODEL="DeepSeek-V4-Flash"
os.environ["API_KEY"]=API_KEY; os.environ["BASE_URL"]=BASE_URL; os.environ["LLM_MODEL"]=MODEL

from eval_utils import load_all_questions, save_results
from openai import OpenAI
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

questions = load_all_questions()
import argparse; p=argparse.ArgumentParser(); p.add_argument("--limit",type=int,default=0); a=p.parse_args()
if a.limit>0: questions=questions[:a.limit]

print(f"Direct LLM: {len(questions)} questions")
results=[]
for i,q in enumerate(questions):
    print(f"  [{i+1}/{len(questions)}] {q['id']}",flush=True)
    t0=time.time()
    try:
        r=client.chat.completions.create(model=MODEL,messages=[{"role":"user","content":q["question"]}],temperature=0.0,max_tokens=1024)
        ans=r.choices[0].message.content
    except Exception as e: ans=f"[ERROR] {e}"
    results.append({"question_id":q["id"],"paper_id":q["paper_id"],"paper_title":q["paper_title"],"intent":q["intent"],"question":q["question"],"ground_truth":q["ground_truth_answer"],"key_entities":q.get("key_entities",[]),"system_answer":str(ans).strip(),"elapsed_sec":round(time.time()-t0,2)})

sys.path.insert(0,'D:/Desktop/evo/ChemEvoRAG_Phase1-main/src')
from judge import judge_answer
print("Scoring...")
for i,r in enumerate(results):
    j=judge_answer(r["system_answer"],r["ground_truth"],r.get("key_entities",[]))
    r["judge_score"]=j.get("score",0); r["entities_found"]=j.get("key_entities_found",[]); r["entities_missing"]=j.get("key_entities_missing",[]); r["judge_reasoning"]=j.get("reasoning","")
    if (i+1)%30==0: print(f"  {i+1}/{len(results)}")
s=[r["judge_score"] for r in results]; print(f"Direct LLM: {sum(s)/len(s):.3f}")
save_results("direct", results, {"model":MODEL})
print("Done")
