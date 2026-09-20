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
    import typesafe_sdk

    mock_client.async_system_one.return_value = {
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

    assert mock_client.async_system_one.call_count == 1
    call_args = mock_client.async_system_one.call_args[1]
    assert call_args["state"] == {"utterance": "Turn on lights"}
    sdk_q = call_args["questions"]

    assert isinstance(sdk_q["intent"], typesafe_sdk.Choice)
    assert sdk_q["intent"].instructions == "Select intent"
    assert sdk_q["intent"].criteria == {"HassTurnOn": "Turn on light"}

    assert isinstance(sdk_q["is_compound"], typesafe_sdk.Noul)
    assert sdk_q["is_compound"].instructions == "Is it compound?"

    assert isinstance(sdk_q["quality"], typesafe_sdk.Score)
    assert sdk_q["quality"].instructions == "Rate quality"
    assert sdk_q["quality"].criteria == ["Low", "Medium", "High"]

    assert isinstance(sdk_q["raw_dict"], typesafe_sdk.Choice)
    assert sdk_q["raw_dict"].instructions == "Custom"

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
    mock_client.async_system_one.return_value = {
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


async def test_engine_async_predict_system_one_response(
    engine: TypeSafeDecisionEngine, mock_client: AsyncMock
) -> None:
    """Test engine handling of typed SystemOneResponse directly from async_system_one."""
    import typesafe_sdk

    mock_client.async_system_one.return_value = typesafe_sdk.SystemOneResponse(
        model="jev-latest",
        usage=typesafe_sdk.Usage(input_tokens=50, output_tokens=10),
        answers={
            "intent": typesafe_sdk.ChoiceAnswer(
                choice="HassTurnOn",
                confidence=0.97,
                probabilities={"HassTurnOn": 0.97},
            ),
            "is_compound": typesafe_sdk.NoulAnswer(noul=0.02),
        },
    )

    questions = {
        "intent": ChoiceQuestion(
            instructions="Intent", criteria={"HassTurnOn": "Turn on"}
        ),
        "is_compound": NoulQuestion(instructions="Compound?"),
    }

    result = await engine.async_predict(state="Turn on the lights", questions=questions)

    assert mock_client.async_system_one.call_count == 1
    assert result.model == "jev-latest"
    assert result.usage == {"input_tokens": 50, "output_tokens": 10}
    assert isinstance(result.answers["intent"], ChoiceAnswer)
    assert result.answers["intent"].choice == "HassTurnOn"
    assert result.answers["intent"].confidence == 0.97
    assert isinstance(result.answers["is_compound"], NoulAnswer)
    assert result.answers["is_compound"].noul == 0.02
