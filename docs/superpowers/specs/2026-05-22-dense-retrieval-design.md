# Dense Retrieval — Design Spec

Date: 2026-05-22 | Phase 1 Gap #3

## 1. Motivation

Current retrieval performs keyword matching (`text_score` in `src/retrieval/common.py`). A query for "solvent" never matches "tetrahydrofuran" even though the semantic relationship is obvious. Phase 1 requires five retrieval types; `dense_search` is the only one not yet implemented.

## 2. Architecture

```
RetrievalRouter.retrieve(query)
  ├─ entity_search        (existing, untouched)
  ├─ lexical_search       (existing, untouched)
  ├─ reaction_event_search(existing, untouched)
  ├─ provenance_backtrack (existing, untouched)
  └─ dense_search         (NEW — this spec)
       │
       ├─ SentenceTransformer("all-MiniLM-L6-v2")
       │    query → 384-dim vector
       │
       └─ QdrantStore
            .search(vector, top_k)
            → CandidateEvidence[]
```

`dense_search` is an additional channel. It does not replace any existing channel.

## 3. Components

### 3.1 QdrantStore — `src/storage/qdrant_store.py`

Encapsulates Qdrant operations behind a simple interface.

```
QdrantStore(host="localhost", port=6333, collection="chemevorag")
  .ensure_collection(vector_size=384)
  .index(chunks: list[EvidenceChunk])          # embed + upsert
  .search(query_vec: ndarray, top_k: int)      # → ScoredChunk[]
  .count()                                      # → int
```

**Vector storage**: Qdrant local mode — data persisted under `data/indexes/qdrant/`. No Docker required. Collection name: `chemevorag`. Vector size: 384 (fixed by `all-MiniLM-L6-v2`). Metric: cosine.

**Payload per point**:

```json
{
  "evidence_id": "mol_card_0001_01",
  "evidence_type": "molecule",
  "doc_id": "paper_001",
  "text": "ethanol | EtOH | CCO | ...",
  "block_id": null
}
```

**Design decisions**:
- Local mode, not client-server — aligns with Phase 1 "file storage first" principle
- `vector_size=384` — fixed by the embedding model
- Payload stores evidence metadata so we can build CandidateEvidence from search results without a second lookup

### 3.2 DenseRetriever — `src/retrieval/dense.py`

Wraps the embedding model and QdrantStore.

```
DenseRetriever(store: QdrantStore, model_name="all-MiniLM-L6-v2")
  .index(evidence_store: LocalStore)             # walk evidence files, embed, upsert
  .search(query: str, top_k: int)                # embed query, search Qdrant, normalize scores
    → CandidateEvidence[]
```

**`index()` walks**:
- `store.load_blocks(doc_id)` → `block_search_text(block)` → embed → Qdrant
- `store.load_molecules(doc_id)` → `molecule_search_text(mol)` → embed → Qdrant
- `store.load_reactions(doc_id)` → `reaction_summary(rxn)` → embed → Qdrant
- `store.load_facts(doc_id)` → `fact_search_text(fact)` → embed → Qdrant

Reuses existing `common.py` helpers (`block_search_text`, `molecule_search_text`, etc.).

**`search()` returns CandidateEvidence** with confidence derived from cosine similarity. Reuses `block_to_candidate`, `molecule_to_candidate`, etc.

### 3.3 RetrievalRouter update — `src/retrieval/router.py`

Modify `retrieve()` method:

```python
elif resolved_intent == "paper_local_question":
    path = ["lexical_retrieval", "dense_retrieval", "provenance_backtracking"]
    lex_candidates = lexical_search(query, self.store, ...)
    dense_candidates = dense_search(query, self.store, ...)  # NEW
    candidates = _dedupe_candidates(lex_candidates + dense_candidates)[:limit]
```

### 3.4 Indexing script — `scripts/index_evidence.py`

One-shot CLI to build the vector index from stored evidence.

```bash
python scripts/index_evidence.py              # index all doc_ids
python scripts/index_evidence.py --doc-id paper_001  # index one document
```

Called after `build_evidence.py` or on first run. Idempotent — re-indexing updates points in place.

## 4. Data Flow

```
Index phase (one-shot):
  LocalStore → DenseRetriever.index()
    → SentenceTransformer.encode(text)  →  [384-dim vector]
    → QdrantStore.upsert(vector, payload)

Query phase:
  query → infer_intent → dense_search(query)
    → SentenceTransformer.encode(query)  →  [384-dim vector]
    → QdrantStore.search(vector, top_k) → ScoredChunk[]
    → chunk_to_candidate()              → CandidateEvidence[]
    → EvidencePackage
```

## 5. File Changes

| File | Action | Lines (est.) |
|------|--------|:--:|
| `src/storage/qdrant_store.py` | New | ~80 |
| `src/retrieval/dense.py` | New | ~120 |
| `src/retrieval/router.py` | Modify — add dense channel | ~10 |
| `src/retrieval/__init__.py` | Modify — export DenseRetriever | ~3 |
| `scripts/index_evidence.py` | New | ~50 |
| `tests/test_dense_retrieval.py` | New | ~100 |

## 6. Dependencies

- `qdrant-client` — already in `environment.yml` (install not needed, but verify via `pip install qdrant-client`)
- `sentence-transformers==2.7.0` — already installed, compatible with `transformers==4.36.2`
- `all-MiniLM-L6-v2` model — downloaded on first use to `HF_HOME` cache (~80MB)

## 7. Testing

- **Unit**: `TestQdrantStore` with in-memory Qdrant — test index, search, update
- **Unit**: `TestDenseRetriever` with fake store — test embed paths, CandidateEvidence conversion
- **Integration**: `test_dense_search` with real LocalStore seeded in `test_retrieval.py`
- **Integration**: `test_router_includes_dense_channel` — verify intent routing

## 8. Acceptance Criteria

1. `dense_search("solvent", store)` returns evidence containing "ethanol", "tetrahydrofuran" without either word appearing in the query
2. `RetrievalRouter.retrieve("What solvent was used?")` includes `dense_retrieval` in `retrieval_path`
3. `scripts/index_evidence.py` runs to completion without errors
4. Existing tests still pass (lexical, entity, reaction channels unchanged)
