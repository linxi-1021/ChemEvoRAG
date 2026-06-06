"""Tests for skill_evolution.regression module."""
import pytest
from pathlib import Path
from skill_evolution.regression import (
    RegressionRunner,
    RegressionResult,
    RegressionComparison,
    decide_after_apply,
)


class TestCompare:
    def test_pass_when_score_improves(self):
        before = RegressionResult(average_score=0.85, total_questions=10, by_intent={
            "a": {"average_score": 0.8, "total": 5, "failed": 1},
        })
        after = RegressionResult(average_score=0.90, total_questions=10, by_intent={
            "a": {"average_score": 0.9, "total": 5, "failed": 0},
        })
        runner = RegressionRunner(Path("."))  # dummy root
        result = runner.compare(before, after, min_score=0.80, max_score_drop=0.01)
        assert result.passed is True
        assert result.score_drop < 0

    def test_fail_when_score_drops(self):
        before = RegressionResult(average_score=0.90, total_questions=10, by_intent={
            "a": {"average_score": 0.9, "total": 5, "failed": 0},
        })
        after = RegressionResult(average_score=0.85, total_questions=10, by_intent={
            "a": {"average_score": 0.85, "total": 5, "failed": 1},
        })
        runner = RegressionRunner(Path("."))
        result = runner.compare(before, after, min_score=0.80, max_score_drop=0.01)
        assert result.passed is False
        assert result.score_drop > 0.01

    def test_fail_when_below_min_score(self):
        before = RegressionResult(average_score=0.85, total_questions=10, by_intent={})
        after = RegressionResult(average_score=0.75, total_questions=10, by_intent={})
        runner = RegressionRunner(Path("."))
        result = runner.compare(before, after, min_score=0.80)
        assert result.passed is False
        assert any("below minimum" in e for e in result.errors)

    def test_fail_when_failed_cases_increase(self):
        before = RegressionResult(average_score=0.90, total_questions=10, by_intent={
            "a": {"average_score": 0.9, "total": 10, "failed": 1},
        })
        after = RegressionResult(average_score=0.90, total_questions=10, by_intent={
            "a": {"average_score": 0.9, "total": 10, "failed": 4},
        })
        runner = RegressionRunner(Path("."))
        result = runner.compare(before, after, min_score=0.80, max_failed_cases_increase=1)
        assert result.passed is False
        assert result.failed_cases_increase > 1

    def test_fail_with_before_errors(self):
        before = RegressionResult(errors=["eval crashed"])
        after = RegressionResult(average_score=0.90, total_questions=10)
        runner = RegressionRunner(Path("."))
        result = runner.compare(before, after)
        assert result.passed is False
        assert "errors" in result.reason.lower() or "error" in result.reason.lower()


class TestDecideAfterApply:
    def test_keep_when_score_maintained(self):
        before = {"summary": {"average_score": 0.89}, "results": []}
        after = {"summary": {"average_score": 0.89}, "results": []}
        result = decide_after_apply(before, after)
        assert result["decision"] == "keep"

    def test_keep_when_score_improved(self):
        before = {"summary": {"average_score": 0.85}, "results": []}
        after = {"summary": {"average_score": 0.90}, "results": []}
        result = decide_after_apply(before, after)
        assert result["decision"] == "keep"

    def test_rollback_when_score_drops(self):
        before = {"summary": {"average_score": 0.90}, "results": [
            {"id": "q1", "score": 0.9}, {"id": "q2", "score": 0.9},
        ]}
        after = {"summary": {"average_score": 0.85}, "results": [
            {"id": "q1", "score": 0.8}, {"id": "q2", "score": 0.9},
        ]}
        result = decide_after_apply(before, after, max_score_drop=0.01)
        assert result["decision"] == "rollback"
        assert "dropped" in result["reason"].lower()

    def test_rollback_when_failed_cases_increase(self):
        before = {"summary": {"average_score": 0.89}, "results": [
            {"id": "q1", "score": 0.9}, {"id": "q2", "score": 0.9},
            {"id": "q3", "score": 0.9},
        ]}
        after = {"summary": {"average_score": 0.89}, "results": [
            {"id": "q1", "score": 0.9}, {"id": "q2", "score": 0.0},
            {"id": "q3", "score": 0.0}, {"id": "q4", "score": 0.0},
        ]}
        result = decide_after_apply(before, after, max_failed_cases_increase=1)
        assert result["decision"] == "rollback"

    def test_manual_review_on_slight_drop(self):
        before = {"summary": {"average_score": 0.90}, "results": []}
        after = {"summary": {"average_score": 0.895}, "results": []}
        result = decide_after_apply(before, after, max_score_drop=0.01)
        assert result["decision"] == "manual_review"
