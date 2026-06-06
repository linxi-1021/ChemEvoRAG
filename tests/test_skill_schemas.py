"""Tests for Skill YAML Pydantic models."""

import pytest
from pydantic import ValidationError
from skill_evolution.schemas import (
    SkillYAML, Trigger, Scope, Dependencies, Interface,
    Strategy, RetrievalRouting, QueryRewrite, EvidenceExpansion,
    Assessment, AnswerGeneration, Template, Evolution,
    FailureToPatchEntry, MutationPolicy, Safety, RiskPolicy,
)


def test_trigger_defaults():
    t = Trigger(intent="property_query")
    assert t.intent == "property_query"
    assert t.confidence_threshold == 0.7
    assert t.aliases == []
    assert t.required_signals == []


def test_scope_defaults():
    s = Scope()
    assert s.supported_properties == []
    assert s.supported_units == []


def test_strategy_defaults():
    st = Strategy()
    assert st.retrieval_routing.primary_channels == []
    assert st.query_rewrite.enabled is True
    assert st.evidence_expansion.enabled is True


def test_template_required_fields():
    with pytest.raises(ValidationError):
        Template()  # name is required


def test_template_with_applicable_when():
    t = Template(
        name="yield_extraction",
        description="Extract yield",
        applicable_when={"property_name": ["isolated_yield"]},
        not_applicable_when=["not a table query"],
        slots=["compound_id", "yield_value"],
    )
    assert t.name == "yield_extraction"
    assert "property_name" in t.applicable_when
    assert len(t.not_applicable_when) == 1


def test_evolution_defaults():
    e = Evolution()
    assert e.enabled is True
    assert e.mutable_paths == []
    assert e.frozen_paths == []


def test_failure_to_patch_entry():
    entry = FailureToPatchEntry(
        target_paths=["strategy.assessment"],
        allowed_operations=["update"],
        allowed_fields=["system_prompt_ref"],
    )
    assert entry.target_paths == ["strategy.assessment"]


def test_safety_defaults():
    s = Safety()
    assert s.elementkg_modification == "forbidden"
    assert s.max_patches_per_round == 2
    assert s.patch_confidence_threshold == 0.75
    assert s.uncertain_patch_queue is True


def test_skill_yaml_minimal():
    yaml_data = {
        "name": "property_query",
        "trigger": {"intent": "property_query"},
    }
    skill = SkillYAML(**yaml_data)
    assert skill.name == "property_query"
    assert skill.schema_version == "1.0"
    assert skill.status == "stable"


def test_skill_yaml_full():
    yaml_data = {
        "name": "property_query",
        "schema_version": "1.0",
        "skill_version": "1.0.0",
        "status": "stable",
        "trigger": {"intent": "property_query", "confidence_threshold": 0.8},
        "scope": {"supported_properties": ["isolated_yield"]},
        "evolution": {
            "enabled": True,
            "mutable_paths": ["strategy.assessment"],
            "frozen_paths": ["name", "trigger.intent"],
            "failure_to_patch_mapping": {
                "assessment_false_negative": {
                    "target_paths": ["strategy.assessment"],
                    "allowed_operations": ["update"],
                    "allowed_fields": ["system_prompt_ref"],
                }
            },
        },
        "safety": {"max_patches_per_round": 3},
    }
    skill = SkillYAML(**yaml_data)
    assert skill.evolution.enabled is True
    assert skill.safety.max_patches_per_round == 3
    assert "assessment_false_negative" in skill.evolution.failure_to_patch_mapping


def test_skill_yaml_from_yaml_roundtrip(tmp_path):
    """from_yaml() loads a YAML file and produces a valid SkillYAML."""
    import yaml
    data = {
        "name": "property_query",
        "trigger": {"intent": "property_query"},
        "evolution": {
            "enabled": True,
            "mutable_paths": ["strategy.assessment", "strategy.query_rewrite.fallback_terms"],
        },
    }
    path = tmp_path / "property_query.yaml"
    path.write_text(yaml.dump(data, allow_unicode=True), encoding="utf-8")
    skill = SkillYAML.from_yaml(path)
    assert skill.name == "property_query"
    assert skill.trigger.intent == "property_query"


def test_mutable_paths_cannot_overlap_frozen_paths():
    """Evolution with overlapping mutable and frozen paths should raise."""
    with pytest.raises(ValidationError, match="frozen|overlap"):
        SkillYAML(
            name="test",
            trigger={"intent": "test"},
            evolution={
                "enabled": True,
                "mutable_paths": ["strategy.assessment"],
                "frozen_paths": ["strategy.assessment"],
            },
        )
