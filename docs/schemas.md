# ChemEvoRAG Phase 1 Schemas

This document defines the first stable data contracts for ChemEvoRAG Phase 1.

The schemas are intentionally conservative. Every object keeps provenance, confidence, raw upstream payloads, and errors so the system can explain both successful answers and extraction failures.

## Design Rules

1. Evidence objects must be traceable to the original PDF.
2. Parser-specific outputs must be normalized before entering retrieval.
3. Failed or partial extraction should produce structured records, not silent drops.
4. Molecule, reaction, and fact evidence must be queryable independently.
5. The final answer must cite evidence IDs and source locations.

## SourceProvenance

Represents a source location in the original document.

```json
{
  "doc_id": "paper_001",
  "source_file": "data/raw_pdfs/paper_001.pdf",
  "page": 6,
  "bbox": [120, 300, 800, 600],
  "block_id": "block_0032",
  "figure_id": "scheme_1",
  "table_id": null,
  "section": "Experimental Section",
  "text_span": [0, 128]
}
```

## DocumentBlock

Represents a parsed text/table/figure/caption/procedure block.

```json
{
  "block_id": "block_0032",
  "doc_id": "paper_001",
  "block_type": "procedure",
  "text": "Compound 7 was synthesized according to General Procedure B...",
  "page": 6,
  "bbox": [120, 300, 800, 600],
  "section": "Experimental Section",
  "prev_block_id": "block_0031",
  "next_block_id": "block_0033",
  "source_file": "paper_001.pdf",
  "confidence": 0.91,
  "raw_payload": {},
  "errors": []
}
```

## MoleculeCard

Represents one molecule or compound mention cluster.

```json
{
  "molecule_card_id": "mol_card_0001",
  "doc_id": "paper_001",
  "local_ids": ["compound 7", "intermediate 3a"],
  "names": ["DCHA"],
  "raw_smiles": "C1CCC(CC1)NC2CCCCC2",
  "canonical_smiles": "C1CCC(CC1)NC2CCCCC2",
  "inchi_key": "...",
  "iupac_name": "dicyclohexylamine",
  "aliases": ["DCHA", "dicyclohexylamine", "二环己胺"],
  "source_mentions": [],
  "source_images": [],
  "linked_elementkg_id": null,
  "confidence": 0.92,
  "normalization_status": "success",
  "raw_payload": {},
  "errors": []
}
```

## ReactionEventCard

Represents one reaction event assembled from text, tables, figures, and procedures.

```json
{
  "reaction_event_id": "rxn_event_0001",
  "doc_id": "paper_001",
  "reaction_smiles": "...",
  "reactants": [],
  "products": [],
  "reagents": ["aniline"],
  "catalysts": [],
  "solvents": ["ethanol"],
  "temperature": "25 °C",
  "time": "4 h",
  "yield_value": {
    "value": 92,
    "unit": "%",
    "normalized_value": 0.92
  },
  "procedure_text": "...",
  "source": {},
  "confidence": 0.88,
  "evidence_completeness": 0.81,
  "raw_payload": {},
  "errors": []
}
```

## FactCard

Represents a claim, property, mechanism statement, comparison, or reaction condition fact.

```json
{
  "fact_card_id": "fact_0001",
  "doc_id": "paper_001",
  "fact_type": "reaction_condition",
  "claim": "The reaction gives 92% yield in ethanol at 25 °C.",
  "entities": ["mol_card_0001", "rxn_event_0001"],
  "supporting_blocks": ["block_0032"],
  "supporting_figures": ["scheme_1"],
  "source": {},
  "confidence": 0.86,
  "raw_payload": {},
  "errors": []
}
```

## EvidencePackage

Represents all evidence sent to the answer generation stage.

```json
{
  "query": "Which reaction has the highest yield?",
  "intent": "reaction_comparison",
  "resolved_entities": [],
  "candidate_evidence": [],
  "provenance": [],
  "missing_slots": [],
  "conflicts": [],
  "retrieval_path": [
    "reaction_event_retrieval",
    "provenance_backtracking"
  ]
}
```

## GroundedAnswer

Represents the final structured answer returned by ChemEvoRAG.

```json
{
  "answer": "Reaction rxn_event_0001 has the highest extracted yield: 92%.",
  "supporting_evidence": [],
  "provenance": [],
  "involved_entities": [],
  "retrieval_path": [],
  "confidence": 0.82,
  "uncertainty": "Only reactions with extractable yield values were compared."
}
```

