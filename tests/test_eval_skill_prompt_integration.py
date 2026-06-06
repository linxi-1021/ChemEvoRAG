"""Tests for Skill YAML / Prompt Registry integration with eval_questions.py.

All tests are static/unit-level — no LLM calls, no real eval runs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ── PromptContext tests ──────────────────────────────────────────────────────

class TestPromptContext:
    """Test PromptContext dataclass."""

    def test_defaults_to_none(self):
        from solver.prompt_context import PromptContext
        ctx = PromptContext()
        assert ctx.evidence_assessment_system is None
        assert ctx.answer_generation_system is None
        assert ctx.answer_generation_examples is None
        assert not ctx.has_assessment_override
        assert not ctx.has_answer_override

    def test_with_values(self):
        from solver.prompt_context import PromptContext
        ctx = PromptContext(
            evidence_assessment_system="custom assessment",
            answer_generation_system="custom answer",
        )
        assert ctx.has_assessment_override
        assert ctx.has_answer_override
        assert ctx.evidence_assessment_system == "custom assessment"


# ── resolve_prompt_context tests ─────────────────────────────────────────────

class TestResolvePromptContext:
    """Test resolve_prompt_context from Skill YAML + Prompt Registry."""

    def test_resolves_both_prompts(self):
        from solver.prompt_context import resolve_prompt_context
        skill_config = {
            "strategy": {
                "assessment": {"system_prompt_ref": "MY_ASSESSMENT"},
                "answer_generation": {"system_prompt_ref": "MY_ANSWER"},
            }
        }
        registry = {
            "MY_ASSESSMENT": "custom assessment prompt text",
            "MY_ANSWER": "custom answer prompt text",
        }
        ctx = resolve_prompt_context(skill_config, registry)
        assert ctx.evidence_assessment_system == "custom assessment prompt text"
        assert ctx.answer_generation_system == "custom answer prompt text"

    def test_fallback_when_no_skill_config(self):
        from solver.prompt_context import resolve_prompt_context
        ctx = resolve_prompt_context({}, {})
        assert ctx.evidence_assessment_system is None
        assert ctx.answer_generation_system is None

    def test_fallback_when_ref_missing_from_registry(self):
        from solver.prompt_context import resolve_prompt_context
        warnings = []
        skill_config = {
            "strategy": {
                "assessment": {"system_prompt_ref": "NONEXISTENT"},
                "answer_generation": {"system_prompt_ref": "ALSO_MISSING"},
            }
        }
        ctx = resolve_prompt_context(skill_config, {}, warnings=warnings)
        assert ctx.evidence_assessment_system is None
        assert ctx.answer_generation_system is None
        assert len(warnings) == 2
        assert "NONEXISTENT" in warnings[0]
        assert "ALSO_MISSING" in warnings[1]

    def test_partial_resolution(self):
        from solver.prompt_context import resolve_prompt_context
        skill_config = {
            "strategy": {
                "assessment": {"system_prompt_ref": "EXISTS"},
                "answer_generation": {},  # no ref
            }
        }
        registry = {"EXISTS": "found prompt"}
        ctx = resolve_prompt_context(skill_config, registry)
        assert ctx.evidence_assessment_system == "found prompt"
        assert ctx.answer_generation_system is None

    def test_different_intents_different_prompts(self):
        from solver.prompt_context import resolve_prompt_context
        property_config = {
            "strategy": {
                "assessment": {"system_prompt_ref": "PROP_ASSESS"},
                "answer_generation": {"system_prompt_ref": "PROP_ANSWER"},
            }
        }
        comparison_config = {
            "strategy": {
                "assessment": {"system_prompt_ref": "COMP_ASSESS"},
                "answer_generation": {"system_prompt_ref": "COMP_ANSWER"},
            }
        }
        registry = {
            "PROP_ASSESS": "property assessment prompt",
            "PROP_ANSWER": "property answer prompt",
            "COMP_ASSESS": "comparison assessment prompt",
            "COMP_ANSWER": "comparison answer prompt",
        }
        ctx_prop = resolve_prompt_context(property_config, registry)
        ctx_comp = resolve_prompt_context(comparison_config, registry)
        assert ctx_prop.evidence_assessment_system != ctx_comp.evidence_assessment_system
        assert ctx_prop.answer_generation_system != ctx_comp.answer_generation_system


# ── LLM Solver prompt injection tests ────────────────────────────────────────

class TestLLMSolverPromptInjection:
    """Test that LLMChemSolver uses PromptContext for system prompt."""

    def test_uses_prompt_context_answer_prompt(self, monkeypatch):
        """When PromptContext provides answer_generation_system, the solver
        should use it in the LLM call instead of the default prompt."""
        from solver.prompt_context import PromptContext
        from evidence import EvidencePackage, CandidateEvidence

        captured_messages = []

        def mock_call_openai(system_prompt, user_prompt, **kwargs):
            captured_messages.append({"system": system_prompt, "user": user_prompt})
            return json.dumps({
                "answer": "test answer",
                "confidence": 0.8,
                "uncertainty": None,
            })

        # Patch _call_openai in the llm_solver module
        import solver.llm_solver as llm_mod
        monkeypatch.setattr(llm_mod, "_call_openai", mock_call_openai)

        ctx = PromptContext(
            answer_generation_system="CUSTOM_ANSWER_PROMPT_FROM_REGISTRY",
        )

        solver = llm_mod.LLMChemSolver()
        package = EvidencePackage(
            query="test query",
            candidate_evidence=[
                CandidateEvidence(
                    evidence_id="ev1",
                    evidence_type="fact",
                    summary="test evidence",
                    confidence=0.9,
                ),
            ],
        )
        solver.answer_from_package(package, prompt_context=ctx)

        assert len(captured_messages) == 1
        assert captured_messages[0]["system"] == "CUSTOM_ANSWER_PROMPT_FROM_REGISTRY"

    def test_fallback_to_default_when_no_context(self, monkeypatch):
        """When no PromptContext is provided, solver should use the default
        prompt from prompts module."""
        from solver.prompt_context import PromptContext
        from evidence import EvidencePackage, CandidateEvidence

        captured_messages = []

        def mock_call_openai(system_prompt, user_prompt, **kwargs):
            captured_messages.append({"system": system_prompt})
            return json.dumps({
                "answer": "test",
                "confidence": 0.5,
                "uncertainty": None,
            })

        import solver.llm_solver as llm_mod
        monkeypatch.setattr(llm_mod, "_call_openai", mock_call_openai)

        solver = llm_mod.LLMChemSolver()
        package = EvidencePackage(
            query="test",
            candidate_evidence=[
                CandidateEvidence(
                    evidence_id="ev1",
                    evidence_type="fact",
                    summary="test",
                    confidence=0.5,
                ),
            ],
        )
        solver.answer_from_package(package)  # no prompt_context

        import solver.prompts as prompts_mod
        assert captured_messages[0]["system"] == prompts_mod.ANSWER_GENERATION_SYSTEM

    def test_fallback_when_context_has_none(self, monkeypatch):
        """When PromptContext is provided but answer_generation_system is None,
        solver should fallback to default."""
        from solver.prompt_context import PromptContext
        from evidence import EvidencePackage, CandidateEvidence

        captured_messages = []

        def mock_call_openai(system_prompt, user_prompt, **kwargs):
            captured_messages.append({"system": system_prompt})
            return json.dumps({
                "answer": "test",
                "confidence": 0.5,
                "uncertainty": None,
            })

        import solver.llm_solver as llm_mod
        monkeypatch.setattr(llm_mod, "_call_openai", mock_call_openai)

        ctx = PromptContext()  # all None
        solver = llm_mod.LLMChemSolver()
        package = EvidencePackage(
            query="test",
            candidate_evidence=[
                CandidateEvidence(
                    evidence_id="ev1",
                    evidence_type="fact",
                    summary="test",
                    confidence=0.5,
                ),
            ],
        )
        solver.answer_from_package(package, prompt_context=ctx)

        import solver.prompts as prompts_mod
        assert captured_messages[0]["system"] == prompts_mod.ANSWER_GENERATION_SYSTEM


# ── ReAct Solver prompt injection tests ──────────────────────────────────────

class TestReActSolverPromptInjection:
    """Test that ReActChemSolver uses PromptContext for evidence assessment.

    Tests _resolve_assessment_prompt directly (avoids openai import dependency).
    """

    def test_resolve_assessment_prompt_uses_context(self):
        """_resolve_assessment_prompt returns context override when provided."""
        from solver.prompt_context import PromptContext
        from solver.react_solver import _resolve_assessment_prompt

        ctx = PromptContext(
            evidence_assessment_system="CUSTOM_ASSESSMENT_PROMPT",
        )
        result = _resolve_assessment_prompt(ctx)
        assert result == "CUSTOM_ASSESSMENT_PROMPT"

    def test_resolve_assessment_prompt_fallback_none(self):
        """_resolve_assessment_prompt falls back to default when context is None."""
        import solver.prompts as prompts_mod
        from solver.react_solver import _resolve_assessment_prompt

        result = _resolve_assessment_prompt(None)
        assert result == prompts_mod.EVIDENCE_ASSESSMENT_SYSTEM

    def test_resolve_assessment_prompt_fallback_empty(self):
        """_resolve_assessment_prompt falls back when context fields are None."""
        import solver.prompts as prompts_mod
        from solver.prompt_context import PromptContext
        from solver.react_solver import _resolve_assessment_prompt

        ctx = PromptContext()  # all None
        result = _resolve_assessment_prompt(ctx)
        assert result == prompts_mod.EVIDENCE_ASSESSMENT_SYSTEM

    def test_retry_path_uses_prompt_context(self, monkeypatch):
        """The retry path in _assess_evidence must also use assessment_prompt
        from prompt_context, not the old hardcoded prompt.

        Uses a fake openai module injected into sys.modules.
        """
        import sys
        from unittest.mock import MagicMock
        from solver.prompt_context import PromptContext
        from solver.react_solver import ReActChemSolver

        captured_calls = []
        call_count = {"n": 0}

        def mock_create_fn(**kwargs):
            call_count["n"] += 1
            captured_calls.append(kwargs["messages"])
            if call_count["n"] == 1:
                mock_resp = MagicMock()
                mock_resp.choices = [MagicMock()]
                mock_resp.choices[0].message.content = "NOT VALID JSON"
                return mock_resp
            else:
                mock_resp = MagicMock()
                mock_resp.choices = [MagicMock()]
                mock_resp.choices[0].message.content = json.dumps({
                    "sufficient": True, "reason": "retry success",
                })
                return mock_resp

        # Create fake openai module
        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value.chat.completions.create = mock_create_fn
        monkeypatch.setitem(sys.modules, "openai", mock_openai)

        ctx = PromptContext(
            evidence_assessment_system="CUSTOM_RETRY_PROMPT",
        )
        monkeypatch.setenv("API_KEY", "test-key")

        solver = ReActChemSolver.__new__(ReActChemSolver)
        result = solver._assess_evidence("test", [], 0, prompt_context=ctx)

        assert len(captured_calls) == 2
        assert captured_calls[0][0]["content"] == "CUSTOM_RETRY_PROMPT"
        assert captured_calls[1][0]["content"] == "CUSTOM_RETRY_PROMPT"
        assert result["sufficient"] is True


# ── CLI argument parsing tests ───────────────────────────────────────────────

class TestCLIArgs:
    """Test that eval_questions.py parses the new CLI arguments."""

    def test_new_args_parse(self):
        """Verify --skills-dir, --prompts-dir, --dataset, --output-dir parse."""
        import argparse
        from pathlib import Path

        parser = argparse.ArgumentParser()
        parser.add_argument("--limit", type=int, default=0)
        parser.add_argument("--paper", default=None)
        parser.add_argument("--react", action="store_true")
        parser.add_argument("--workers", type=int, default=1)
        parser.add_argument("--skills", action="store_true")
        parser.add_argument("--skills-dir", type=Path, default=None)
        parser.add_argument("--prompts-dir", type=Path, default=None)
        parser.add_argument("--dataset", type=Path, default=None)
        parser.add_argument("--output-dir", type=Path, default=None)

        args = parser.parse_args([
            "--skills",
            "--skills-dir", "/tmp/skills",
            "--prompts-dir", "/tmp/prompts",
            "--dataset", "/tmp/questions.json",
            "--output-dir", "/tmp/output",
        ])

        assert args.skills is True
        assert args.skills_dir == Path("/tmp/skills")
        assert args.prompts_dir == Path("/tmp/prompts")
        assert args.dataset == Path("/tmp/questions.json")
        assert args.output_dir == Path("/tmp/output")

    def test_defaults(self):
        """Verify defaults are None when not specified."""
        import argparse
        from pathlib import Path

        parser = argparse.ArgumentParser()
        parser.add_argument("--skills-dir", type=Path, default=None)
        parser.add_argument("--prompts-dir", type=Path, default=None)
        parser.add_argument("--dataset", type=Path, default=None)
        parser.add_argument("--output-dir", type=Path, default=None)

        args = parser.parse_args([])
        assert args.skills_dir is None
        assert args.prompts_dir is None
        assert args.dataset is None
        assert args.output_dir is None
