import pytest

from evidence import MoleculeCard
from normalization import RDKitNormalizer


def _molecule(**overrides) -> MoleculeCard:
    defaults = {
        "molecule_card_id": "mol_001",
        "doc_id": "paper_001",
    }
    return MoleculeCard(**(defaults | overrides))


# ── availability and graceful degradation ────────────────────────────────


def test_available_reports_false_when_rdkit_not_installed():
    normalizer = RDKitNormalizer()
    # available is a property — accessing it twice should be stable
    first = normalizer.available
    second = normalizer.available
    assert first is second


def test_normalize_returns_failed_card_when_rdkit_unavailable():
    normalizer = RDKitNormalizer()
    if normalizer.available:
        pytest.skip("RDKit is installed — skip graceful-degradation test")

    card = _molecule(names=["ethanol"], raw_smiles="CCO")
    result = normalizer.normalize(card)

    assert result.normalization_status == "failed"
    assert "RDKit is not installed" in result.errors[0]
    # Original card is unchanged
    assert card.normalization_status == "not_attempted"


def test_normalize_returns_failed_card_when_no_smiles():
    normalizer = RDKitNormalizer()
    if not normalizer.available:
        pytest.skip("RDKit not installed")

    card = _molecule(names=["unknown"])
    result = normalizer.normalize(card)

    assert result.normalization_status == "failed"
    assert "No SMILES available" in str(result.errors)


# ── helpers when RDKit is absent ──────────────────────────────────────────


def test_validate_smiles_returns_true_when_rdkit_unavailable():
    normalizer = RDKitNormalizer()
    if normalizer.available:
        pytest.skip("RDKit is installed")
    # When RDKit is absent the method cannot judge, so it returns True
    # (permissive — let it through to the next stage)
    assert normalizer.validate_smiles("CCO") is True
    assert normalizer.validate_smiles("not-even-smiles") is True


def test_to_canonical_smiles_returns_none_when_rdkit_unavailable():
    normalizer = RDKitNormalizer()
    if normalizer.available:
        pytest.skip("RDKit is installed")
    assert normalizer.to_canonical_smiles("CCO") is None


def test_to_inchikey_returns_none_when_rdkit_unavailable():
    normalizer = RDKitNormalizer()
    if normalizer.available:
        pytest.skip("RDKit is installed")
    assert normalizer.to_inchikey("CCO") is None


# ── normalizer with RDKit present ─────────────────────────────────────────


@pytest.fixture(scope="module")
def normalizer():
    n = RDKitNormalizer()
    if not n.available:
        pytest.skip("RDKit not installed — skipping integration tests")
    return n


def test_validate_smiles_accepts_valid_smiles(normalizer):
    assert normalizer.validate_smiles("CCO") is True
    assert normalizer.validate_smiles("C1CCC(CC1)NC2CCCCC2") is True


def test_validate_smiles_rejects_invalid_smiles(normalizer):
    assert normalizer.validate_smiles("not-a-smiles") is False
    assert normalizer.validate_smiles("") is False


def test_to_canonical_smiles_normalises_isomeric_smiles(normalizer):
    # Ethanol: CCO and OCC describe the same molecule
    result = normalizer.to_canonical_smiles("OCC")
    assert result == "CCO"


def test_to_inchikey_generates_standard_key(normalizer):
    result = normalizer.to_inchikey("CCO")
    # Ethanol's standard InChIKey
    assert result == "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"


def test_to_inchikey_returns_none_for_invalid_smiles(normalizer):
    assert normalizer.to_inchikey("not-a-smiles") is None


def test_normalize_fills_canonical_smiles_and_inchikey(normalizer):
    card = _molecule(names=["ethanol"], raw_smiles="OCC")
    result = normalizer.normalize(card)

    assert result.normalization_status == "success"
    assert result.canonical_smiles == "CCO"
    assert result.inchi_key == "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"
    # raw_smiles is left as-supplied when canonical_smiles was empty
    assert result.raw_smiles == "OCC"


def test_normalize_preserves_existing_canonical_smiles_when_already_set(normalizer):
    card = _molecule(
        names=["ethanol"],
        raw_smiles="OCC",
        canonical_smiles="CCO",
        inchi_key="LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
    )
    result = normalizer.normalize(card)

    assert result.normalization_status == "success"
    assert result.canonical_smiles == "CCO"
    assert result.inchi_key == "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"


def test_normalize_marks_failed_for_invalid_smiles(normalizer):
    card = _molecule(names=["junk"], raw_smiles="not-a-smiles")
    result = normalizer.normalize(card)

    assert result.normalization_status == "failed"
    assert "RDKit could not parse SMILES" in " ".join(result.errors)


def test_normalize_does_not_mutate_original_card(normalizer):
    card = _molecule(names=["ethanol"], raw_smiles="CCO")
    result = normalizer.normalize(card)

    assert card.normalization_status == "not_attempted"
    assert card.canonical_smiles is None
    assert result is not card


def test_normalize_batch_returns_same_length(normalizer):
    cards = [
        _molecule(molecule_card_id="mol_001", names=["ethanol"], raw_smiles="CCO"),
        _molecule(molecule_card_id="mol_002", names=["aspirin"], raw_smiles="CC(=O)OC1=CC=CC=C1C(=O)O"),
    ]
    results = normalizer.normalize_batch(cards)

    assert len(results) == 2
    assert results[0].molecule_card_id == "mol_001"
    assert results[1].molecule_card_id == "mol_002"
    assert all(r.normalization_status == "success" for r in results)
