"""Unit tests for EngineScorer and DecisionEngine in Stage 4."""

from __future__ import annotations

import pytest

from typing import cast

from custom_components.typesafe.speculative.hydration.models import HydratedPayload
from custom_components.typesafe.speculative.models import (
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    NoulQuestion,
)
from custom_components.typesafe.speculative.scoring.scorer import (
    DecisionScorer,
    EngineScorer,
)
from custom_components.typesafe.speculative.testing.engine import FakeDecisionEngine


@pytest.fixture(name="scorer")
def scorer_fixture() -> DecisionScorer:
    """Fixture providing an EngineScorer."""
    return EngineScorer()


@pytest.fixture(name="engine")
def engine_fixture() -> FakeDecisionEngine:
    """Fixture providing a FakeDecisionEngine."""
    return FakeDecisionEngine(
        default_answers={
            "intent": ChoiceAnswer(choice="HassTurnOn", confidence=0.96),
            "is_compound": NoulAnswer(noul=0.02),
        }
    )


async def test_engine_scorer_evaluates_questions(
    scorer: DecisionScorer, engine: FakeDecisionEngine
) -> None:
    """Test that EngineScorer passes state and questions to engine and returns PredictionResult."""
    payload = HydratedPayload(
        questions={
            "intent": ChoiceQuestion(
                instructions="Determine intent",
                criteria={"HassTurnOn": "Turn on"},
            ),
            "is_compound": NoulQuestion(instructions="Is compound?"),
        },
        state={"utterance": "Turn on light"},
    )

    result = await scorer.score(payload, engine)
    assert len(engine.calls) == 1
    assert engine.calls[0]["state"] == {"utterance": "Turn on light"}
    assert "intent" in result.answers
    intent_answer = cast(ChoiceAnswer, result.answers["intent"])
    assert intent_answer.choice == "HassTurnOn"
    assert intent_answer.confidence == 0.96
