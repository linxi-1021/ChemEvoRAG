import pytest
from neo4j.exceptions import AuthError, ServiceUnavailable

from evidence import MoleculeCard, MoleculeIdentityCandidate
from normalization import Neo4jIdentityClient, Neo4jIdentityConfig


class FakeSession:
    def __init__(self, records):
        self.records = records
        self.calls = []
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.closed = True

    def run(self, cypher, parameters):
        self.calls.append((cypher, parameters))
        return self.records


class FakeDriver:
    def __init__(self, records=None):
        self.records = records or []
        self.sessions = []
        self.session_kwargs = []
        self.closed = False

    def session(self, **kwargs):
        self.session_kwargs.append(kwargs)
        session = FakeSession(self.records)
        self.sessions.append(session)
        return session

    def close(self):
        self.closed = True


class FailingDriver:
    def __init__(self, error):
        self.error = error

    def session(self, **kwargs):
        raise self.error

    def close(self):
        pass


def make_config(**overrides):
    payload = {
        "uri": "bolt://example:7687",
        "username": "neo4j",
        "password": "secret",
        "database": "neo4j",
        "molecule_label": "Molecule",
        "id_field": "elementkg_id",
        "canonical_name_field": "canonical_name",
        "name_fields": ["canonical_name", "name", "iupac_name"],
        "alias_fields": ["aliases"],
        "smiles_field": "smiles",
        "inchikey_field": "inchi_key",
        "result_limit": 5,
    }
    payload.update(overrides)
    return Neo4jIdentityConfig(**payload)


def test_find_by_name_uses_configured_fields_and_parameters():
    driver = FakeDriver(
        records=[
            {
                "node": {
                    "elementkg_id": "mol_123",
                    "canonical_name": "dicyclohexylamine",
                    "aliases": ["DCHA"],
                    "smiles": "C1CCC(CC1)NC2CCCCC2",
                    "inchi_key": "ABC",
                }
            }
        ]
    )
    client = Neo4jIdentityClient(make_config(), driver=driver)

    results = client.find_by_name("dicyclohexylamine")

    cypher, params = driver.sessions[0].calls[0]
    assert "MATCH (m:`Molecule`)" in cypher
    assert "m.`canonical_name` = $value" in cypher
    assert "m.`name` = $value" in cypher
    assert "LIMIT $limit" in cypher
    assert params == {"value": "dicyclohexylamine", "limit": 5}
    assert driver.session_kwargs[0] == {"database": "neo4j"}
    assert results[0].elementkg_id == "mol_123"
    assert results[0].match_type == "name"


def test_find_by_alias_supports_list_alias_fields():
    driver = FakeDriver(records=[])
    client = Neo4jIdentityClient(make_config(), driver=driver)

    results = client.find_by_alias("DCHA")

    cypher, params = driver.sessions[0].calls[0]
    assert "m.`aliases` = $value OR $value IN m.`aliases`" in cypher
    assert params["value"] == "DCHA"
    assert results == []


def test_find_by_smiles_and_inchikey_use_exact_structural_fields():
    driver = FakeDriver(records=[])
    client = Neo4jIdentityClient(make_config(), driver=driver)

    assert client.find_by_smiles("CCO") == []
    assert client.find_by_inchikey("LFQSCWFLJHTTHZ-UHFFFAOYSA-N") == []

    smiles_cypher, smiles_params = driver.sessions[0].calls[0]
    inchikey_cypher, inchikey_params = driver.sessions[1].calls[0]
    assert "m.`smiles` = $value" in smiles_cypher
    assert smiles_params["value"] == "CCO"
    assert "m.`inchi_key` = $value" in inchikey_cypher
    assert inchikey_params["value"] == "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"


def test_candidate_from_node_maps_configured_fields():
    driver = FakeDriver(
        records=[
            {
                "node": {
                    "elementkg_id": 123,
                    "canonical_name": "ethanol",
                    "aliases": ["ethyl alcohol", "EtOH"],
                    "smiles": "CCO",
                    "inchi_key": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
                    "extra": "kept",
                }
            }
        ]
    )
    client = Neo4jIdentityClient(make_config(), driver=driver)

    result = client.find_by_smiles("CCO")[0]

    assert isinstance(result, MoleculeIdentityCandidate)
    assert result.elementkg_id == "123"
    assert result.canonical_name == "ethanol"
    assert result.aliases == ["ethyl alcohol", "EtOH"]
    assert result.smiles == "CCO"
    assert result.raw_node["extra"] == "kept"


def test_missing_lookup_fields_raise_clear_error():
    client = Neo4jIdentityClient(make_config(name_fields=[]), driver=FakeDriver())

    with pytest.raises(ValueError, match="No Neo4j fields configured"):
        client.find_by_name("ethanol")


def test_unsafe_config_identifier_is_rejected():
    with pytest.raises(ValueError, match="Unsafe Neo4j identifier"):
        make_config(molecule_label="Molecule) DETACH DELETE m //")


def test_resolve_molecule_prefers_inchikey_then_updates_card():
    driver = FakeDriver(
        records=[
            {
                "node": {
                    "elementkg_id": "mol_ethanol",
                    "canonical_name": "ethanol",
                    "aliases": ["ethyl alcohol"],
                    "smiles": "CCO",
                    "inchi_key": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
                }
            }
        ]
    )
    client = Neo4jIdentityClient(make_config(), driver=driver)
    molecule = MoleculeCard(
        molecule_card_id="mol_card_1",
        doc_id="paper_001",
        names=["EtOH"],
        aliases=["EtOH"],
        inchi_key="LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
    )

    resolved = client.resolve_molecule(molecule)

    cypher, params = driver.sessions[0].calls[0]
    assert "m.`inchi_key` = $value" in cypher
    assert params["value"] == "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"
    assert resolved.linked_elementkg_id == "mol_ethanol"
    assert resolved.canonical_smiles == "CCO"
    assert resolved.iupac_name == "ethanol"
    assert resolved.normalization_status == "success"
    assert resolved.aliases == ["EtOH", "ethyl alcohol"]


def test_resolve_molecule_returns_failed_copy_when_not_found():
    client = Neo4jIdentityClient(make_config(), driver=FakeDriver(records=[]))
    molecule = MoleculeCard(
        molecule_card_id="mol_card_1",
        doc_id="paper_001",
        names=["unknown"],
    )

    resolved = client.resolve_molecule(molecule)

    assert resolved is not molecule
    assert resolved.normalization_status == "failed"
    assert "No Neo4j molecule identity candidate found." in resolved.errors


def test_client_close_closes_driver():
    driver = FakeDriver()
    client = Neo4jIdentityClient(make_config(), driver=driver)

    client.close()

    assert driver.closed is True


def test_auth_error_is_reported_as_connection_error():
    client = Neo4jIdentityClient(
        make_config(),
        driver=FailingDriver(AuthError("bad credentials")),
    )

    with pytest.raises(ConnectionError, match="authentication failed"):
        client.find_by_name("ethanol")


def test_service_unavailable_is_reported_as_connection_error():
    client = Neo4jIdentityClient(
        make_config(),
        driver=FailingDriver(ServiceUnavailable("offline")),
    )

    with pytest.raises(ConnectionError, match="service unavailable"):
        client.find_by_name("ethanol")
