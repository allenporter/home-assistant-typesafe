"""Unit tests for TargetBindingDecisionResolver in Stage 5."""

from __future__ import annotations

import pytest

from custom_components.typesafe.speculative.models import (
    ChoiceAnswer,
    NoulAnswer,
)
from custom_components.typesafe.speculative.request.models import ParsedRequest
from custom_components.typesafe.speculative.resolution.target_binding import (
    TargetBindingDecisionResolver,
)
from custom_components.typesafe.speculative.retrieval.models import (
    AreaCandidate,
    EntityCandidate,
    RetrievedCandidates,
)
from custom_components.typesafe.speculative.scoring.engine import PredictionResult


@pytest.fixture(name="resolver")
def resolver_fixture() -> TargetBindingDecisionResolver:
    """Fixture providing a TargetBindingDecisionResolver."""
    return TargetBindingDecisionResolver(
        confidence_threshold=0.7, compound_threshold=0.5
    )


def test_resolve_entity_target_and_bind_brightness(
    resolver: TargetBindingDecisionResolver,
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
    resolver: TargetBindingDecisionResolver,
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
    resolver: TargetBindingDecisionResolver,
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
    resolver: TargetBindingDecisionResolver,
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


def test_resolve_low_confidence_target_entity_escalation(
    resolver: TargetBindingDecisionResolver,
) -> None:
    """Test low confidence on target entity triggers escalation."""
    request = ParsedRequest(
        raw_text="Turn on kitchen light",
        normalized_text="turn on kitchen light",
        tokens={"turn", "on", "kitchen", "light"},
    )
    prediction = PredictionResult(
        answers={
            "intent": ChoiceAnswer(choice="HassTurnOn", confidence=0.95),
            "target_type": ChoiceAnswer(choice="entity", confidence=0.9),
            "target_entity": ChoiceAnswer(
                choice="light.kitchen_light", confidence=0.35
            ),
            "is_compound": NoulAnswer(noul=0.01),
        }
    )

    decision = resolver.resolve(prediction, request, RetrievedCandidates())
    assert decision.should_escalate
    assert decision.confidence == 0.35
    assert decision.escalation_reason == "Low confidence on target entity"


def test_resolve_low_confidence_target_area_escalation(
    resolver: TargetBindingDecisionResolver,
) -> None:
    """Test low confidence on target area triggers escalation."""
    candidates = RetrievedCandidates(
        areas=[AreaCandidate(area_id="living_room", area_name="Living Room", score=1.0)]
    )
    request = ParsedRequest(
        raw_text="Turn off the living room",
        normalized_text="turn off the living room",
        tokens={"turn", "off", "the", "living", "room"},
    )
    prediction = PredictionResult(
        answers={
            "intent": ChoiceAnswer(choice="HassTurnOff", confidence=0.95),
            "target_type": ChoiceAnswer(choice="area", confidence=0.9),
            "target_area": ChoiceAnswer(choice="living_room", confidence=0.3),
            "is_compound": NoulAnswer(noul=0.01),
        }
    )

    decision = resolver.resolve(prediction, request, candidates)
    assert decision.should_escalate
    assert decision.confidence == 0.3
    assert decision.escalation_reason == "Low confidence on target area"


def test_resolve_missing_target_entity_escalation(
    resolver: TargetBindingDecisionResolver,
) -> None:
    """Test missing target entity when entity target type was specified."""
    request = ParsedRequest(
        raw_text="Turn on the device",
        normalized_text="turn on the device",
        tokens={"turn", "on", "the", "device"},
    )
    prediction = PredictionResult(
        answers={
            "intent": ChoiceAnswer(choice="HassTurnOn", confidence=0.95),
            "target_type": ChoiceAnswer(choice="entity", confidence=0.9),
            "is_compound": NoulAnswer(noul=0.01),
        }
    )

    decision = resolver.resolve(prediction, request, RetrievedCandidates())
    assert decision.should_escalate
    assert decision.escalation_reason == "Missing target entity"


def test_resolve_controllable_domain_matching_in_area(
    resolver: TargetBindingDecisionResolver,
) -> None:
    """Test matching controllable domains like fan in area commands."""
    candidates = RetrievedCandidates(
        areas=[AreaCandidate(area_id="patio", area_name="Patio", score=1.0)]
    )
    request = ParsedRequest(
        raw_text="Turn off the patio fans",
        normalized_text="turn off the patio fans",
        tokens={"turn", "off", "the", "patio", "fans"},
    )
    prediction = PredictionResult(
        answers={
            "intent": ChoiceAnswer(choice="HassTurnOff", confidence=0.92),
            "target_type": ChoiceAnswer(choice="area", confidence=0.9),
            "target_area": ChoiceAnswer(choice="patio", confidence=0.88),
            "is_compound": NoulAnswer(noul=0.01),
        }
    )

    decision = resolver.resolve(prediction, request, candidates)
    assert not decision.should_escalate
    assert decision.area_name == "Patio"
    assert decision.slots["area"] == "Patio"


def test_resolve_entity_preserves_domain_and_area_metadata(
    resolver: TargetBindingDecisionResolver,
) -> None:
    """Test resolving entity target preserves domain and preferred area from entity candidate."""
    candidates = RetrievedCandidates(
        entities=[
            EntityCandidate(
                entity_id="cover.garage_door",
                domain="cover",
                friendly_name="Garage Door",
                area_id="garage",
                area_name="Garage",
                score=1.0,
            )
        ]
    )
    request = ParsedRequest(
        raw_text="Open the garage door",
        normalized_text="open the garage door",
        tokens={"open", "the", "garage", "door"},
    )
    prediction = PredictionResult(
        answers={
            "intent": ChoiceAnswer(choice="HassOpenCover", confidence=0.95),
            "target_type": ChoiceAnswer(choice="entity", confidence=0.9),
            "target_entity": ChoiceAnswer(choice="cover.garage_door", confidence=0.92),
            "is_compound": NoulAnswer(noul=0.01),
        }
    )

    decision = resolver.resolve(prediction, request, candidates)
    assert not decision.should_escalate
    assert decision.intent_name == "HassOpenCover"
    assert decision.entity_id == "cover.garage_door"
    assert decision.domain == "cover"
    assert decision.slots["domain"] == "cover"
    assert decision.slots["preferred_area_id"] == "garage"
