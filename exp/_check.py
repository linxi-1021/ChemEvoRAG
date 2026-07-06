import fast_graphrag
import inspect
# Check GraphRAG init signature
sig = inspect.signature(fast_graphrag.GraphRAG.__init__)
print("GraphRAG.__init__:", sig)
# Check key methods
for m in ["query", "insert", "add", "ingest", "build", "search"]:
    obj = getattr(fast_graphrag.GraphRAG, m, None)
    if obj:
        print(f"GraphRAG.{m}:", inspect.signature(obj))
# For insert method, try other variations
for m in dir(fast_graphrag.GraphRAG):
    if not m.startswith("_") and "doc" in m.lower() or "text" in m.lower() or "insert" in m.lower() or "add" in m.lower():
        print(f"  GraphRAG.{m}")
