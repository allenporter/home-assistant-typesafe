"""Unit tests for candidate hydrators in Stage 3."""

from __future__ import annotations

from typing import cast

import pytest
from custom_components.typesafe.speculative.hydration.hydrator import (
    HierarchicalCandidateHydrator,
    SimpleCandidateHydrator,
)
from custom_components.typesafe.speculative.models import ChoiceQuestion
from custom_components.typesafe.speculative.retrieval.models import (
    AreaCandidate,
    EntityCandidate,
    IntentCandidate,
    RetrievedCandidates,
)


@pytest.fixture(name="hydrator")
def hydrator_fixture() -> HierarchicalCandidateHydrator:
    """Fixture providing a hierarchical hydrator."""
    return HierarchicalCandidateHydrator()


def test_hydrate_empty_candidates_defaults(
    hydrator: HierarchicalCandidateHydrator,
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

    assert "is_compound" in questions

    assert "target_entity" in questions
    entity_q = cast(ChoiceQuestion, questions["target_entity"])
    assert "light.kitchen_lights" in entity_q.criteria

    assert "target_area" in questions
    area_q = cast(ChoiceQuestion, questions["target_area"])
    assert "kitchen" in area_q.criteria

    assert "target_type" in questions
    type_q = cast(ChoiceQuestion, questions["target_type"])
    assert "entity" in type_q.criteria
    assert "area" in type_q.criteria


def test_hydrate_does_not_prune_candidates(
    hydrator: HierarchicalCandidateHydrator,
) -> None:
    """Test that hydration does not slice or prune candidates, leaving that to retrieval."""
    intents = [
        IntentCandidate(
            intent_type=f"CustomIntent{i}",
            score=1.0,
            description=f"Description {i}",
        )
        for i in range(12)
    ]
    areas = [
        AreaCandidate(area_id=f"area_{i}", area_name=f"Area {i}", score=1.0)
        for i in range(15)
    ]
    entities = [
        EntityCandidate(
            entity_id=f"light.light_{i}",
            domain="light",
            friendly_name=f"Light {i}",
            score=1.0,
        )
        for i in range(35)
    ]

    candidates = RetrievedCandidates(intents=intents, areas=areas, entities=entities)
    questions = hydrator.hydrate(candidates)

    intent_q = cast(ChoiceQuestion, questions["intent"])
    assert len(intent_q.criteria) == 12

    entity_q = cast(ChoiceQuestion, questions["target_entity"])
    assert len(entity_q.criteria) == 35

    area_q = cast(ChoiceQuestion, questions["target_area"])
    assert len(area_q.criteria) == 15


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
