# PubChem Seed Identity Test

This is a minimal test flow that does not use papers or PDF parsing.

## Build seed data

```bash
conda run -n chemevorag python scripts/bootstrap_pubchem_seed.py
```

The script fetches a few molecules from PubChem and writes:

- `data/evidence/pubchem_seed.molecules.json`
- `data/evidence/pubchem_seed.facts.json`
- `data/kg/pubchem_seed_graph.json`

Seed molecules:

- dicyclohexylamine, with alias `DCHA`
- ethanol, with alias `EtOH`
- benzaldehyde
- aniline
- aspirin

## Test questions

```bash
conda run -n chemevorag python scripts/run_query.py "What is DCHA?" --doc-id pubchem_seed
conda run -n chemevorag python scripts/run_query.py "DCHA是什么？" --doc-id pubchem_seed
```

Expected behavior: the local entity retrieval should return PubChem molecule
`pubchem:7582` and include SMILES plus InChIKey in the answer.
