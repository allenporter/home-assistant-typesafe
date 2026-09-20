"""Unit tests for DefaultCandidateHydrator in Stage 3."""

from __future__ import annotations

from typing import cast

import pytest
from custom_components.typesafe.speculative.hydration.hydrator import (
    DefaultCandidateHydrator,
)
from custom_components.typesafe.speculative.models import ChoiceQuestion
from custom_components.typesafe.speculative.retrieval.models import (
    AreaCandidate,
    EntityCandidate,
    IntentCandidate,
    RetrievedCandidates,
)


@pytest.fixture(name="hydrator")
def hydrator_fixture() -> DefaultCandidateHydrator:
    """Fixture providing a default hydrator."""
    return DefaultCandidateHydrator()


def test_hydrate_empty_candidates_defaults(
    hydrator: DefaultCandidateHydrator,
) -> None:
    """Test candidate hydration builds intent, entity, area, and compound questions."""
    candidates = RetrievedCandidates(
        intents=[
            IntentCandidate(
                intent_type="HassTurnOff",
                score=2.0,
                description="Turn off or deactivate a device",
            )
        ],
        areas=[AreaCandidate(area_id="kitchen", area_name="Kitchen", score=1.0)],
        entities=[
            EntityCandidate(
                entity_id="light.kitchen_lights",
                domain="light",
                friendly_name="Kitchen Lights",
                area_name="Kitchen",
                score=3.0,
            )
        ],
    )

    questions = hydrator.hydrate(candidates)

    assert "intent" in questions
    intent_q = cast(ChoiceQuestion, questions["intent"])
    assert "HassTurnOff" in intent_q.criteria
    assert "unmatched" in intent_q.criteria

    assert "is_compound" in questions

    assert "target_entity" in questions
    entity_q = cast(ChoiceQuestion, questions["target_entity"])
    assert "light.kitchen_lights" in entity_q.criteria
    assert "none" in entity_q.criteria

    assert "target_area" in questions
    area_q = cast(ChoiceQuestion, questions["target_area"])
    assert "kitchen" in area_q.criteria
    assert "none" in area_q.criteria

    assert "target_type" in questions
    type_q = cast(ChoiceQuestion, questions["target_type"])
    assert "entity" in type_q.criteria
    assert "area" in type_q.criteria
