import json
f = r"D:\Desktop\evo\ChemEvoRAG_Phase1-main\data\baseline_results\fastgraphrag_20260614_194804.json"
d = json.load(open(f, encoding="utf-8"))
print("Keys:", list(d.keys()))
print("Total:", d["total_questions"])
print("Metadata:", d.get("metadata", {}))
# Print first result
r0 = d["results"][0]
print("\nFirst result keys:", list(r0.keys()))
print(f"  question_id: {r0['question_id']}")
print(f"  system_answer[:150]: {r0.get('system_answer','')[:150]}")
print(f"  judge_score: {r0.get('judge_score')}")
print(f"  elapsed_sec: {r0.get('elapsed_sec')}")
# Score distribution
scores = [r.get("judge_score",0) for r in d["results"]]
print(f"\nOverall: {sum(scores)/len(scores):.4f}")
print(f"Non-zero: {sum(1 for s in scores if s>0)}/{len(scores)}")
