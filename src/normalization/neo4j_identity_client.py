"""Neo4j-backed molecule identity lookup.

The client is intentionally schema-configurable: graph labels and property names
come from ``Neo4jIdentityConfig`` instead of being hard-coded.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

import yaml
from neo4j import GraphDatabase
from neo4j.exceptions import AuthError, Neo4jError, ServiceUnavailable
from pydantic import Field, SecretStr, field_validator

from evidence import MoleculeCard, MoleculeIdentityCandidate
from evidence.schemas import ChemEvoBaseModel, MoleculeMatchType


class Neo4jIdentityConfig(ChemEvoBaseModel):
    """Configuration for Neo4j molecule identity lookup."""

    uri: str
    username: str
    password: SecretStr
    database: str | None = None
    molecule_label: str
    id_field: str
    canonical_name_field: str | None = None
    name_fields: list[str] = Field(default_factory=list)
    alias_fields: list[str] = Field(default_factory=list)
    smiles_field: str
    inchikey_field: str
    result_limit: int = Field(default=10, ge=1, le=100)

    @field_validator(
        "molecule_label",
        "id_field",
        "canonical_name_field",
        "smiles_field",
        "inchikey_field",
    )
    @classmethod
    def validate_identifier(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if not _is_safe_identifier(value):
            raise ValueError(f"Unsafe Neo4j identifier: {value!r}")
        return value

    @field_validator("name_fields", "alias_fields")
    @classmethod
    def validate_identifier_list(cls, values: list[str]) -> list[str]:
        for value in values:
            if not _is_safe_identifier(value):
                raise ValueError(f"Unsafe Neo4j identifier: {value!r}")
        return values

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Neo4jIdentityConfig":
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Neo4j config file not found: {config_path}")

        with config_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}

        neo4j_payload = payload.get("neo4j", payload)
        return cls(**neo4j_payload)


class Neo4jIdentityClient:
    """Query molecule identity candidates from a Neo4j graph."""

    def __init__(self, config: Neo4jIdentityConfig, driver: Any | None = None) -> None:
        self.config = config
        self._driver = driver or GraphDatabase.driver(
            config.uri,
            auth=(config.username, config.password.get_secret_value()),
        )

    def close(self) -> None:
        self._driver.close()

    def find_by_name(self, name: str) -> list[MoleculeIdentityCandidate]:
        return self._query_fields(
            fields=self.config.name_fields,
            value=name,
            match_type="name",
        )

    def find_by_alias(self, alias: str) -> list[MoleculeIdentityCandidate]:
        return self._query_fields(
            fields=self.config.alias_fields,
            value=alias,
            match_type="alias",
            allow_list_fields=True,
        )

    def find_by_smiles(self, smiles: str) -> list[MoleculeIdentityCandidate]:
        return self._query_fields(
            fields=[self.config.smiles_field],
            value=smiles,
            match_type="smiles",
        )

    def find_by_inchikey(self, inchi_key: str) -> list[MoleculeIdentityCandidate]:
        return self._query_fields(
            fields=[self.config.inchikey_field],
            value=inchi_key,
            match_type="inchikey",
        )

    def resolve_molecule(self, molecule: MoleculeCard) -> MoleculeCard:
        """Return a copy of ``molecule`` enriched with the best graph candidate."""

        candidates = self._resolve_candidates(molecule)
        if not candidates:
            return molecule.model_copy(
                update={
                    "normalization_status": "failed",
                    "errors": molecule.errors
                    + ["No Neo4j molecule identity candidate found."],
                },
                deep=True,
            )

        best = candidates[0]
        update: dict[str, Any] = {
            "linked_elementkg_id": best.elementkg_id,
            "canonical_smiles": best.smiles or molecule.canonical_smiles,
            "inchi_key": best.inchi_key or molecule.inchi_key,
            "aliases": _dedupe([*molecule.aliases, *best.aliases]),
            "normalization_status": "success"
            if best.smiles and best.inchi_key
            else "partial",
            "raw_payload": {
                **molecule.raw_payload,
                "neo4j_identity_candidate": best.model_dump(mode="json"),
            },
        }
        if best.canonical_name:
            update["iupac_name"] = best.canonical_name
            update["names"] = _dedupe([*molecule.names, best.canonical_name])

        return molecule.model_copy(update=update, deep=True)

    def _resolve_candidates(
        self, molecule: MoleculeCard
    ) -> list[MoleculeIdentityCandidate]:
        if molecule.inchi_key:
            candidates = self.find_by_inchikey(molecule.inchi_key)
            if candidates:
                return candidates

        for smiles in _dedupe([molecule.canonical_smiles, molecule.raw_smiles]):
            if smiles:
                candidates = self.find_by_smiles(smiles)
                if candidates:
                    return candidates

        for name in _dedupe([*molecule.names, *molecule.local_ids]):
            candidates = self.find_by_name(name)
            if candidates:
                return candidates

        for alias in _dedupe(molecule.aliases):
            candidates = self.find_by_alias(alias)
            if candidates:
                return candidates

        return []

    def _query_fields(
        self,
        *,
        fields: Iterable[str],
        value: str,
        match_type: MoleculeMatchType,
        allow_list_fields: bool = False,
    ) -> list[MoleculeIdentityCandidate]:
        clean_fields = [field for field in fields if field]
        if not clean_fields:
            raise ValueError(f"No Neo4j fields configured for {match_type} lookup.")
        if not value:
            return []

        conditions = []
        for field in clean_fields:
            if allow_list_fields:
                conditions.append(f"m.`{field}` = $value OR $value IN m.`{field}`")
            else:
                conditions.append(f"m.`{field}` = $value")

        cypher = (
            f"MATCH (m:`{self.config.molecule_label}`) "
            f"WHERE {' OR '.join(conditions)} "
            "RETURN properties(m) AS node "
            "LIMIT $limit"
        )
        records = self._run_query(
            cypher,
            {"value": value, "limit": self.config.result_limit},
        )
        return [
            self._candidate_from_node(record["node"], match_type=match_type)
            for record in records
        ]

    def _run_query(self, cypher: str, parameters: dict[str, Any]) -> list[dict[str, Any]]:
        session_kwargs = {}
        if self.config.database:
            session_kwargs["database"] = self.config.database

        try:
            with self._driver.session(**session_kwargs) as session:
                result = session.run(cypher, parameters)
                return [dict(record) for record in result]
        except AuthError as exc:
            raise ConnectionError("Neo4j authentication failed.") from exc
        except ServiceUnavailable as exc:
            raise ConnectionError(
                f"Neo4j service unavailable for URI {self.config.uri!r}."
            ) from exc
        except Neo4jError as exc:
            raise RuntimeError(f"Neo4j query failed: {exc}") from exc

    def _candidate_from_node(
        self, node: dict[str, Any], *, match_type: MoleculeMatchType
    ) -> MoleculeIdentityCandidate:
        canonical_name = _first_present(
            node,
            [
                self.config.canonical_name_field,
                *self.config.name_fields,
            ],
        )
        aliases = _collect_aliases(node, self.config.alias_fields)

        # 置信度分级：匹配越精确，置信度越高
        score_map: dict[str, float] = {
            "inchikey": 1.0,    # 精确匹配
            "smiles": 0.95,     # SMILES 匹配
            "name": 0.9,        # 名称匹配
            "alias": 0.85,      # 别名匹配
            "unknown": 0.7,     # 未知匹配类型
        }
        score = score_map.get(match_type, 0.7)

        return MoleculeIdentityCandidate(
            elementkg_id=_as_optional_str(node.get(self.config.id_field)),
            canonical_name=_as_optional_str(canonical_name),
            aliases=aliases,
            smiles=_as_optional_str(node.get(self.config.smiles_field)),
            inchi_key=_as_optional_str(node.get(self.config.inchikey_field)),
            match_type=match_type,
            score=score,
            raw_node=node,
        )


def _is_safe_identifier(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value))


def _first_present(node: dict[str, Any], fields: Iterable[str | None]) -> Any | None:
    for field in fields:
        if field and node.get(field) not in (None, ""):
            return node[field]
    return None


def _collect_aliases(node: dict[str, Any], fields: Iterable[str]) -> list[str]:
    aliases: list[str] = []
    for field in fields:
        value = node.get(field)
        if isinstance(value, list):
            aliases.extend(str(item) for item in value if item not in (None, ""))
        elif value not in (None, ""):
            aliases.append(str(value))
    return _dedupe(aliases)


def _dedupe(values: Iterable[str | None]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not value:
            continue
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _as_optional_str(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)
