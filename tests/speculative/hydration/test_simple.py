"""Unit tests for SimpleCandidateHydrator in Stage 3."""

from __future__ import annotations

from typing import cast

from custom_components.typesafe.speculative.hydration.simple import (
    SimpleCandidateHydrator,
)
from custom_components.typesafe.speculative.models import ChoiceQuestion
from custom_components.typesafe.speculative.retrieval.models import (
    AreaCandidate,
    EntityCandidate,
    IntentCandidate,
    RetrievedCandidates,
)


def test_simple_candidate_hydrator() -> None:
    """Test SimpleCandidateHydrator generates only the intent question."""
    hydrator = SimpleCandidateHydrator()
    candidates = RetrievedCandidates(
        intents=[
            IntentCandidate(
                intent_type="HassTurnOff",
                score=2.0,
                description="Turn off devices",
            )
        ],
        areas=[AreaCandidate(area_id="kitchen", area_name="Kitchen", score=1.0)],
        entities=[
            EntityCandidate(
                entity_id="light.kitchen_lights",
                domain="light",
                friendly_name="Kitchen Lights",
                score=3.0,
            )
        ],
    )

    questions = hydrator.hydrate(candidates)
    assert list(questions.keys()) == ["intent"]
    intent_q = cast(ChoiceQuestion, questions["intent"])
    assert "HassTurnOff" in intent_q.criteria
