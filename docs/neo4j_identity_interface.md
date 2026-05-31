# Neo4j Molecule Identity Interface

This interface queries a user-provided Neo4j molecule graph for canonical molecule identity data.

It does not hard-code graph labels, property names, or credentials. Copy `config/neo4j.yaml.example` to `config/neo4j.yaml` and fill in your own graph schema.

## Configuration

```yaml
neo4j:
  uri: "bolt://localhost:7687"
  username: "neo4j"
  password: "CHANGE_ME"
  database: "neo4j"
  molecule_label: "Molecule"
  id_field: "elementkg_id"
  canonical_name_field: "canonical_name"
  name_fields:
    - "canonical_name"
    - "name"
    - "iupac_name"
  alias_fields:
    - "aliases"
  smiles_field: "smiles"
  inchikey_field: "inchi_key"
  result_limit: 10
```

Only identifiers matching `[A-Za-z_][A-Za-z0-9_]*` are accepted for labels and property names.

## Usage

```python
from normalization import Neo4jIdentityClient, Neo4jIdentityConfig

config = Neo4jIdentityConfig.from_yaml("config/neo4j.yaml")
client = Neo4jIdentityClient(config)

try:
    candidates = client.find_by_name("dicyclohexylamine")
finally:
    client.close()
```

## MoleculeCard Resolution

```python
resolved = client.resolve_molecule(molecule_card)
```

Lookup priority:

```text
InChIKey -> SMILES -> names/local IDs -> aliases
```

When a candidate is found, the returned `MoleculeCard` is copied and enriched with:

```text
linked_elementkg_id
canonical_smiles
inchi_key
iupac_name / canonical name
aliases
normalization_status
raw_payload.neo4j_identity_candidate
```

When no candidate is found, the returned copy keeps the original fields and receives:

```text
normalization_status = "failed"
errors += ["No Neo4j molecule identity candidate found."]
```

