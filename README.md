# ChemEvoRAG Phase 1

ChemEvoRAG Phase 1 is the MVP stage for building a chemistry-literature RAG system.

The goal of this stage is not to finish the full self-evolving ChemEvoRAG system, but to build a reliable first pipeline:

```text
Chemical PDF
  -> chemistry-aware parsing
  -> Molecule / Reaction / Fact evidence objects
  -> chemical identity normalization
  -> searchable storage
  -> grounded answer with evidence and provenance
```

## Core Combination

Phase 1 uses the following project combination:

| Layer | Component | Role |
|---|---|---|
| Chemical PDF parsing | OpenChemIE first, ChemEagle optional | Extract molecules, reactions, figures, tables, text evidence |
| Chemical normalization | RDKit, PubChem/OPSIN optional | Canonical SMILES, InChIKey, validation, alias expansion |
| Evidence storage | PostgreSQL, Neo4j, Qdrant or pgvector | Store structured evidence, graph relations, vector retrieval |
| RAG retrieval | LightRAG first, RAG-Anything optional | Text and graph retrieval over processed evidence |
| Solver orchestration | ROMA | Query planning, retrieval routing, tool execution, answer synthesis |

## Phase 1 Output

By the end of Phase 1, the project should support:

- Ingesting a small batch of chemistry PDFs.
- Extracting text blocks, figures, molecule candidates, and reaction candidates.
- Building `MoleculeCard`, `ReactionEventCard`, and `FactCard`.
- Normalizing molecules with RDKit.
- Storing evidence with provenance.
- Answering chemistry literature questions with supporting evidence and source locations.

See [docs/phase1_goal_and_stack.md](docs/phase1_goal_and_stack.md) for the detailed plan.

## Planning Docs

- [Phase 1 goal and stack](docs/phase1_goal_and_stack.md)
- [External dependency record](docs/external_dependencies.md)
- [Implementation steps](docs/implementation_steps.md)
- [Evidence schemas](docs/schemas.md)
- [Neo4j molecule identity interface](docs/neo4j_identity_interface.md)

## Conda Environment

The initial development environment is:

```bash
conda activate chemevorag
```

It is also recorded in [environment.yml](environment.yml).
