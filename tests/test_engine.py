"""Unit tests for TypeSafeDecisionEngine adapter."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from custom_components.typesafe.client import TypeSafeClient
from custom_components.typesafe.engine import TypeSafeDecisionEngine
from custom_components.typesafe.speculative.models import (
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
)


@pytest.fixture(name="mock_client")
def mock_client_fixture() -> AsyncMock:
    """Fixture providing a mocked TypeSafeClient."""
    return AsyncMock(spec=TypeSafeClient)


@pytest.fixture(name="engine")
def engine_fixture(mock_client: AsyncMock) -> TypeSafeDecisionEngine:
    """Fixture providing a TypeSafeDecisionEngine initialized with mock_client."""
    return TypeSafeDecisionEngine(mock_client)


async def test_engine_async_predict_question_serialization(
    engine: TypeSafeDecisionEngine, mock_client: AsyncMock
) -> None:
    """Test that canonical Question primitives are correctly serialized for TypeSafe API."""
    mock_client.async_evaluate.return_value = {
        "model": "jev-latest",
        "answers": {
            "intent": {"type": "choice", "choice": "HassTurnOn", "confidence": 0.95},
            "is_compound": {"type": "noul", "noul": 0.05, "confidence": 0.0},
            "quality": {"type": "score", "score": 2.5, "confidence": 0.8},
        },
        "usage": {"input_tokens": 100, "output_tokens": 20},
    }

    questions = {
        "intent": ChoiceQuestion(
            instructions="Select intent",
            criteria={"HassTurnOn": "Turn on light"},
        ),
        "is_compound": NoulQuestion(instructions="Is it compound?"),
        "quality": ScoreQuestion(
            instructions="Rate quality",
            criteria=["Low", "Medium", "High"],
        ),
        "raw_dict": {"type": "choice", "instructions": "Custom", "criteria": {}},
    }

    result = await engine.async_predict(
        state={"utterance": "Turn on lights"}, questions=questions
    )

    assert mock_client.async_evaluate.call_count == 1
    call_args = mock_client.async_evaluate.call_args[1]
    assert call_args["state"] == {"utterance": "Turn on lights"}
    serialized = call_args["questions"]

    assert serialized["intent"] == {
        "type": "choice",
        "instructions": "Select intent",
        "criteria": {"HassTurnOn": "Turn on light"},
    }
    assert serialized["is_compound"] == {
        "type": "noul",
        "instructions": "Is it compound?",
    }
    assert serialized["quality"] == {
        "type": "score",
        "instructions": "Rate quality",
        "criteria": ["Low", "Medium", "High"],
    }
    assert serialized["raw_dict"] == {
        "type": "choice",
        "instructions": "Custom",
        "criteria": {},
    }

    assert result.model == "jev-latest"
    assert result.usage == {"input_tokens": 100, "output_tokens": 20}
    assert isinstance(result.answers["intent"], ChoiceAnswer)
    assert result.answers["intent"].choice == "HassTurnOn"
    assert result.answers["intent"].confidence == 0.95
    assert isinstance(result.answers["is_compound"], NoulAnswer)
    assert result.answers["is_compound"].noul == 0.05
    assert isinstance(result.answers["quality"], ScoreAnswer)
    assert result.answers["quality"].score == 2.5


async def test_engine_async_predict_inferred_types_and_none_choice(
    engine: TypeSafeDecisionEngine, mock_client: AsyncMock
) -> None:
    """Test inferring answer types when 'type' field is missing, and handling None choice."""
    mock_client.async_evaluate.return_value = {
        "model": "jev-latest",
        "answers": {
            "intent": {"choice": None, "confidence": 0.9},
            "is_compound": {"noul": 0.1},
            "rating": {"score": 1.5, "probabilities": {"0": 0.2, "1": 0.8}},
            "non_dict": "invalid",
        },
    }

    questions = {
        "intent": ChoiceQuestion(instructions="Select intent", criteria={}),
        "is_compound": NoulQuestion(instructions="Is it compound?"),
        "rating": ScoreQuestion(instructions="Rate", criteria=["A", "B"]),
    }

    result = await engine.async_predict(state="Test", questions=questions)

    # None choice should become empty string, not string "None"
    assert isinstance(result.answers["intent"], ChoiceAnswer)
    assert result.answers["intent"].choice == ""
    assert result.answers["intent"].confidence == 0.9
    assert isinstance(result.answers["is_compound"], NoulAnswer)
    assert result.answers["is_compound"].noul == 0.1
    assert isinstance(result.answers["rating"], ScoreAnswer)
    assert result.answers["rating"].score == 1.5
    assert "non_dict" not in result.answers
