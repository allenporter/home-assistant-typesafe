"""Unit tests for SimpleDecisionResolver in Stage 5."""

from __future__ import annotations

from custom_components.typesafe.speculative.models import ChoiceAnswer
from custom_components.typesafe.speculative.request.models import ParsedRequest
from custom_components.typesafe.speculative.resolution.simple import (
    SimpleDecisionResolver,
)
from custom_components.typesafe.speculative.retrieval.models import RetrievedCandidates
from custom_components.typesafe.speculative.scoring.engine import PredictionResult


def test_simple_decision_resolver_success() -> None:
    """Test SimpleDecisionResolver passes on confident intent."""
    simple = SimpleDecisionResolver(confidence_threshold=0.8)
    request = ParsedRequest(raw_text="Turn on lights", normalized_text="turn on lights")
    prediction = PredictionResult(
        answers={
            "intent": ChoiceAnswer(choice="HassTurnOn", confidence=0.85),
        }
    )

    decision = simple.resolve(prediction, request, RetrievedCandidates())
    assert not decision.should_escalate
    assert decision.intent_name == "HassTurnOn"
    assert decision.confidence == 0.85
    assert decision.slots == {}


def test_simple_decision_resolver_low_confidence() -> None:
    """Test SimpleDecisionResolver escalates on low confidence."""
    simple = SimpleDecisionResolver(confidence_threshold=0.8)
    request = ParsedRequest(raw_text="Hello", normalized_text="hello")
    prediction = PredictionResult(
        answers={
            "intent": ChoiceAnswer(choice="HassTurnOn", confidence=0.75),
        }
    )

    decision = simple.resolve(prediction, request, RetrievedCandidates())
    assert decision.should_escalate
    assert decision.confidence == 0.75
    assert decision.escalation_reason == "Unhandled intent or low confidence"
