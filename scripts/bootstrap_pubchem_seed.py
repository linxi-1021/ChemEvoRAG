#!/usr/bin/env python
"""Bootstrap a tiny PubChem-backed molecule identity graph."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evidence import FactCard, MoleculeCard  # noqa: E402
from normalization import RDKitNormalizer  # noqa: E402
from storage import LocalStore  # noqa: E402


DOC_ID = "pubchem_seed"
PUBCHEM_PROPERTIES = (
    "CanonicalSMILES,IsomericSMILES,InChIKey,IUPACName,"
    "MolecularFormula,MolecularWeight"
)

DEFAULT_SEEDS: list[dict[str, Any]] = [
    {
        "query_name": "dicyclohexylamine",
        "preferred_name": "dicyclohexylamine",
        "aliases": ["DCHA", "di(cyclohexyl)amine"],
    },
    {"query_name": "ethanol", "preferred_name": "ethanol", "aliases": ["EtOH"]},
    {
        "query_name": "benzaldehyde",
        "preferred_name": "benzaldehyde",
        "aliases": [],
    },
    {"query_name": "aniline", "preferred_name": "aniline", "aliases": []},
    {"query_name": "aspirin", "preferred_name": "aspirin", "aliases": ["acetylsalicylic acid"]},
]


Fetcher = Callable[[str], dict[str, Any]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", default=str(PROJECT_ROOT), help="Project base directory.")
    parser.add_argument("--doc-id", default=DOC_ID, help="Doc id used for local evidence files.")
    args = parser.parse_args()

    artifacts = bootstrap_pubchem_seed(
        LocalStore(base_dir=args.base_dir),
        doc_id=args.doc_id,
    )
    print(json.dumps(artifacts, ensure_ascii=False, indent=2))
    return 0


def bootstrap_pubchem_seed(
    store: LocalStore,
    *,
    doc_id: str = DOC_ID,
    seeds: list[dict[str, Any]] | None = None,
    fetcher: Fetcher | None = None,
) -> dict[str, Any]:
    """Fetch seed molecules from PubChem and persist local evidence plus KG JSON."""

    selected_seeds = seeds or DEFAULT_SEEDS
    pubchem_fetcher = fetcher or fetch_pubchem_properties
    molecules: list[MoleculeCard] = []

    for seed in selected_seeds:
        payload = pubchem_fetcher(seed["query_name"])
        molecule = molecule_from_pubchem(seed, payload, doc_id=doc_id)
        molecules.append(molecule)

    # Phase 1 — RDKit normalization
    normalizer = RDKitNormalizer()
    if normalizer.available:
        normalized_count = 0
        for i, card in enumerate(molecules):
            result = normalizer.normalize(card)
            if result.normalization_status == "success":
                normalized_count += 1
            molecules[i] = result
        if normalized_count:
            print(
                f"[normalize] {normalized_count}/{len(molecules)} molecules normalized",
                file=sys.stderr,
            )
    else:
        print("[normalize] RDKit not installed — skipping normalization", file=sys.stderr)

    facts = [identity_fact_from_molecule(molecule) for molecule in molecules]
    graph = build_identity_graph(doc_id, molecules)

    molecules_path = store.save_molecules(doc_id, molecules)
    facts_path = store.save_facts(doc_id, facts)
    kg_path = store.base_dir / "data" / "kg" / f"{doc_id}_graph.json"
    _write_json(kg_path, graph)

    return {
        "doc_id": doc_id,
        "molecule_count": len(molecules),
        "fact_count": len(facts),
        "molecules_path": str(molecules_path),
        "facts_path": str(facts_path),
        "kg_path": str(kg_path),
        "test_question": "What is DCHA?",
    }


def fetch_pubchem_properties(name: str) -> dict[str, Any]:
    encoded_name = urllib.parse.quote(name)
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
        f"{encoded_name}/property/{PUBCHEM_PROPERTIES}/JSON"
    )
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "ChemEvoRAG-Phase1/0.1"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def molecule_from_pubchem(
    seed: dict[str, Any],
    payload: dict[str, Any],
    *,
    doc_id: str,
) -> MoleculeCard:
    properties = _first_pubchem_property(payload)
    cid = properties["CID"]
    iupac_name = properties.get("IUPACName")
    canonical_smiles = (
        properties.get("CanonicalSMILES")
        or properties.get("SMILES")
        or properties.get("ConnectivitySMILES")
        or properties.get("IsomericSMILES")
    )
    preferred_name = seed.get("preferred_name") or seed["query_name"]

    names = _unique_strings([preferred_name, iupac_name])
    aliases = _unique_strings([*seed.get("aliases", []), seed["query_name"], iupac_name])

    return MoleculeCard(
        molecule_card_id=f"pubchem_{cid}",
        doc_id=doc_id,
        local_ids=_unique_strings([f"CID:{cid}", *seed.get("aliases", [])]),
        names=names,
        raw_smiles=canonical_smiles,
        canonical_smiles=canonical_smiles,
        inchi_key=properties.get("InChIKey"),
        iupac_name=iupac_name,
        aliases=aliases,
        linked_elementkg_id=f"pubchem:{cid}",
        confidence=1.0,
        normalization_status="success",
        raw_payload={
            "source": "PubChem PUG REST",
            "query_name": seed["query_name"],
            "properties": properties,
        },
    )


def identity_fact_from_molecule(molecule: MoleculeCard) -> FactCard:
    display_name = molecule.names[0] if molecule.names else molecule.molecule_card_id
    alias_text = ", ".join(molecule.aliases)
    claim = f"{display_name} has PubChem id {molecule.linked_elementkg_id}."
    if alias_text:
        claim += f" Known aliases include {alias_text}."
    return FactCard(
        fact_card_id=f"fact_identity_{molecule.molecule_card_id}",
        doc_id=molecule.doc_id,
        fact_type="identity",
        claim=claim,
        entities=_unique_strings(
            [
                molecule.molecule_card_id,
                molecule.linked_elementkg_id,
                *molecule.names,
                *molecule.aliases,
            ]
        ),
        confidence=1.0,
        raw_payload={"source": "PubChem seed bootstrap"},
    )


def build_identity_graph(doc_id: str, molecules: list[MoleculeCard]) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    for molecule in molecules:
        molecule_node_id = molecule.linked_elementkg_id or molecule.molecule_card_id
        nodes.append(
            {
                "id": molecule_node_id,
                "type": "molecule",
                "properties": {
                    "molecule_card_id": molecule.molecule_card_id,
                    "names": molecule.names,
                    "aliases": molecule.aliases,
                    "canonical_smiles": molecule.canonical_smiles,
                    "inchi_key": molecule.inchi_key,
                    "iupac_name": molecule.iupac_name,
                },
            }
        )
        for alias in molecule.aliases:
            alias_node_id = f"alias:{alias}"
            nodes.append(
                {
                    "id": alias_node_id,
                    "type": "alias",
                    "properties": {"name": alias},
                }
            )
            edges.append(
                {
                    "source": alias_node_id,
                    "target": molecule_node_id,
                    "type": "alias_of",
                }
            )
        for local_id in molecule.local_ids:
            id_node_id = f"identifier:{local_id}"
            nodes.append(
                {
                    "id": id_node_id,
                    "type": "identifier",
                    "properties": {"value": local_id},
                }
            )
            edges.append(
                {
                    "source": id_node_id,
                    "target": molecule_node_id,
                    "type": "identifies",
                }
            )

    return {
        "graph_id": doc_id,
        "source": "PubChem PUG REST",
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes": nodes,
        "edges": edges,
    }


def _first_pubchem_property(payload: dict[str, Any]) -> dict[str, Any]:
    properties = payload.get("PropertyTable", {}).get("Properties", [])
    if not properties:
        raise ValueError("PubChem response did not contain any compound properties.")
    first = properties[0]
    if "CID" not in first:
        raise ValueError("PubChem response did not include a CID.")
    return first


def _unique_strings(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if not text or text.lower() in seen:
            continue
        seen.add(text.lower())
        result.append(text)
    return result


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return path


if __name__ == "__main__":
    raise SystemExit(main())
