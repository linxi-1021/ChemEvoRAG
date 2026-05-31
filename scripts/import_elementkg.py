#!/usr/bin/env python
"""Import ElementKG CSV into Neo4j.

Imports Molecule, Reactant, Reagent, Product nodes with their properties
(SMILES, InChIKey, IUPAC name, etc.) for entity resolution.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_dotenv = PROJECT_ROOT / ".env"
if _dotenv.exists():
    from dotenv import load_dotenv
    load_dotenv(_dotenv)

from neo4j import GraphDatabase

BATCH_SIZE = 5000

# 需要导入的节点类型
IMPORT_TYPES = {"Molecule", "Reactant", "Reagent", "Product"}

# 关系类型 → 属性名映射（将 literal 值存为节点属性）
PROPERTY_RELATIONS = {
    # Molecule properties (from PubChem)
    "PUBCHEM_IUPAC_INCHIKEY_IS": "inchikey",
    "PUBCHEM_IUPAC_NAME_IS": "iupac_name",
    "PUBCHEM_IUPAC_TRADITIONAL_NAME_IS": "traditional_name",
    "PUBCHEM_IUPAC_CAS_NAME_IS": "cas_name",
    "PUBCHEM_IUPAC_SYSTEMATIC_NAME_IS": "systematic_name",
    "PUBCHEM_CONNECTIVITY_SMILES_IS": "smiles",
    "PUBCHEM_MOLECULAR_FORMULA_IS": "molecular_formula",
    "PUBCHEM_MOLECULAR_WEIGHT_IS": "molecular_weight",
    "PUBCHEM_EXACT_MASS_IS": "exact_mass",
    "PUBCHEM_IUPAC_INCHI_IS": "inchi",
    # Reactant/Reagent/Product
    "SMILES_IS": "smiles",
    "NAME_IS": "name",
    "QUANTITY_IS": "quantity",
    "STATE_IS": "state",
    "ROLE_IS": "role",
    "CHEMICAL_FORMULA_IS": "chemical_formula",
    "PURPOSE_IS": "purpose",
    "AMOUNT_IS": "amount",
    "DESCRIPTION_IS": "description",
    "CONCLUSION_IS": "conclusion",
}

# 关系类型 → 关系名映射（节点间关系）
EDGE_RELATIONS = {
    "HAS_FUNCTIONALGROUP": "HAS_FUNCTIONALGROUP",
    "PARTICIPATES_IN": "PARTICIPATES_IN",
    "USED_IN": "USED_IN",
    "IS_MOLECULE": "IS_MOLECULE",
}


def _get_driver():
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "chemevorag")
    return GraphDatabase.driver(uri, auth=(user, password))


def create_indexes(driver):
    """为每种节点类型创建索引。"""
    with driver.session() as session:
        for label in IMPORT_TYPES:
            session.run(f"CREATE INDEX IF NOT EXISTS FOR (n:{label}) ON (n.id)")
        # Molecule 的 InChIKey 索引（用于快速查找）
        session.run("CREATE INDEX IF NOT EXISTS FOR (n:Molecule) ON (n.inchikey)")
    print("Indexes created.")


def import_csv(driver, csv_path: Path, batch_size: int = BATCH_SIZE):
    """分批导入 CSV 到 Neo4j。"""
    total_rows = 0
    start = time.time()

    # 累积批次数据
    node_props: dict[str, dict] = {}  # "Type:id" -> {properties}
    edges: list[tuple[str, str, str, str]] = []  # (head_type, head_id, rel_type, tail_value_or_id)

    with driver.session() as session:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                total_rows += 1

                ht = row.get("head_type", "").strip()
                hv = row.get("head_value", "").strip()
                rel = row.get("relation", "").strip()
                tt = row.get("tail_type", "").strip()
                tv = row.get("tail_value", "").strip()

                if not ht or not hv or not rel:
                    continue

                # 只处理指定的节点类型
                if ht not in IMPORT_TYPES:
                    continue

                node_key = f"{ht}:{hv}"

                # 确保节点存在
                if node_key not in node_props:
                    node_props[node_key] = {"id": hv, "_type": ht}

                # 处理关系
                if rel in PROPERTY_RELATIONS:
                    # literal 属性 → 存为节点属性
                    prop_name = PROPERTY_RELATIONS[rel]
                    node_props[node_key][prop_name] = tv
                elif rel in EDGE_RELATIONS:
                    # 节点间关系
                    if tt in IMPORT_TYPES:
                        edges.append((ht, hv, EDGE_RELATIONS[rel], f"{tt}:{tv}"))
                else:
                    # 其他关系，存为节点属性（用原始 relation 名）
                    prop_name = rel.lower()
                    node_props[node_key][prop_name] = tv

                # 批量写入
                if len(node_props) >= batch_size:
                    _flush_batch(session, node_props, edges)
                    elapsed = time.time() - start
                    print(f"  {total_rows} rows, {len(node_props)} nodes ({elapsed:.1f}s)")
                    node_props.clear()
                    edges.clear()

        # 最后一批
        if node_props:
            _flush_batch(session, node_props, edges)

    elapsed = time.time() - start
    print(f"\nDone: {total_rows} rows processed in {elapsed:.1f}s")


def _flush_batch(session, node_props: dict, edges: list):
    """批量写入节点属性和关系。"""
    # 按类型分组写入节点
    by_type: dict[str, list[dict]] = {}
    for key, props in node_props.items():
        ntype = props.pop("_type")
        by_type.setdefault(ntype, []).append(props)

    for ntype, props_list in by_type.items():
        # 使用 UNWIND 批量合并节点
        session.run(
            f"UNWIND $props AS p MERGE (n:{ntype} {{id: p.id}}) SET n += p",
            props=props_list,
        )

    # 批量写入关系
    for ht, hv, rel_type, tail_key in edges:
        tt, tv = tail_key.split(":", 1)
        session.run(
            f"MATCH (a:{ht} {{id: $aid}}), (b:{tt} {{id: $bid}}) "
            f"MERGE (a)-[r:{rel_type}]->(b)",
            aid=hv, bid=tv,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        default=str(PROJECT_ROOT / "data" / "kg" / "10m_elementkg_release.csv"),
        help="Path to ElementKG CSV file.",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}", file=sys.stderr)
        return 1

    print(f"Connecting to Neo4j ...")
    driver = _get_driver()

    try:
        print("Creating indexes ...")
        create_indexes(driver)

        print(f"Importing {csv_path.name} ...")
        import_csv(driver, csv_path, args.batch_size)
    finally:
        driver.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
