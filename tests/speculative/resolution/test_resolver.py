"""Unit tests for DefaultDecisionResolver in Stage 5."""

from __future__ import annotations

import pytest

from custom_components.typesafe.speculative.models import (
    ChoiceAnswer,
    NoulAnswer,
)
from custom_components.typesafe.speculative.request.models import ParsedRequest
from custom_components.typesafe.speculative.resolution.resolver import (
    DecisionResolver,
    DefaultDecisionResolver,
)
from custom_components.typesafe.speculative.retrieval.models import (
    AreaCandidate,
    RetrievedCandidates,
)
from custom_components.typesafe.speculative.scoring.engine import PredictionResult


@pytest.fixture(name="resolver")
def resolver_fixture() -> DecisionResolver:
    """Fixture providing a DefaultDecisionResolver."""
    return DefaultDecisionResolver(confidence_threshold=0.7, compound_threshold=0.5)


def test_resolve_entity_target_and_bind_brightness(
    resolver: DecisionResolver,
) -> None:
    """Test resolving entity target and binding raw percentage to brightness for light domain."""
    request = ParsedRequest(
        raw_text="Set kitchen light to 50%",
        normalized_text="set kitchen light to 50%",
        tokens={"set", "kitchen", "light", "to", "50"},
        raw_percentages=[50],
        raw_numbers=[50.0],
    )
    prediction = PredictionResult(
        answers={
            "intent": ChoiceAnswer(choice="HassLightSet", confidence=0.95),
            "target_type": ChoiceAnswer(choice="entity", confidence=0.9),
            "target_entity": ChoiceAnswer(choice="light.kitchen_light", confidence=0.9),
            "is_compound": NoulAnswer(noul=0.01),
        }
    )

    decision = resolver.resolve(prediction, request, RetrievedCandidates())
    assert not decision.should_escalate
    assert not decision.is_compound
    assert decision.intent_name == "HassLightSet"
    assert decision.entity_id == "light.kitchen_light"
    assert decision.slots["entity_id"] == "light.kitchen_light"
    assert decision.slots["brightness"] == 50


def test_resolve_area_target(
    resolver: DecisionResolver,
) -> None:
    """Test resolving area broadcast target."""
    candidates = RetrievedCandidates(
        areas=[AreaCandidate(area_id="kitchen_123", area_name="Kitchen", score=1.0)]
    )
    request = ParsedRequest(
        raw_text="Turn off the kitchen",
        normalized_text="turn off the kitchen",
        tokens={"turn", "off", "the", "kitchen"},
    )
    prediction = PredictionResult(
        answers={
            "intent": ChoiceAnswer(choice="HassTurnOff", confidence=0.92),
            "target_type": ChoiceAnswer(choice="area", confidence=0.88),
            "target_area": ChoiceAnswer(choice="kitchen_123", confidence=0.9),
            "is_compound": NoulAnswer(noul=0.01),
        }
    )

    decision = resolver.resolve(prediction, request, candidates)
    assert not decision.should_escalate
    assert decision.intent_name == "HassTurnOff"
    assert decision.area_name == "Kitchen"
    assert decision.slots["area"] == "Kitchen"


def test_resolve_compound_escalation(
    resolver: DecisionResolver,
) -> None:
    """Test compound command triggers escalation."""
    request = ParsedRequest(
        raw_text="Turn off fan and turn on light",
        normalized_text="turn off fan and turn on light",
        tokens={"turn", "off", "fan", "and", "turn", "on", "light"},
    )
    prediction = PredictionResult(
        answers={
            "intent": ChoiceAnswer(choice="HassTurnOff", confidence=0.9),
            "is_compound": NoulAnswer(noul=0.85),
        }
    )

    decision = resolver.resolve(prediction, request, RetrievedCandidates())
    assert decision.should_escalate
    assert decision.is_compound
    assert decision.escalation_reason == "Compound command detected"


def test_resolve_low_confidence_escalation(
    resolver: DecisionResolver,
) -> None:
    """Test low confidence prediction triggers escalation."""
    request = ParsedRequest(
        raw_text="Do something ambiguous",
        normalized_text="do something ambiguous",
        tokens={"do", "something", "ambiguous"},
    )
    prediction = PredictionResult(
        answers={
            "intent": ChoiceAnswer(choice="HassTurnOn", confidence=0.4),
            "is_compound": NoulAnswer(noul=0.02),
        }
    )

    decision = resolver.resolve(prediction, request, RetrievedCandidates())
    assert decision.should_escalate
    assert decision.escalation_reason == "Unhandled intent or low confidence"
