"""Unit tests for TypeSafeDecisionEngine adapter."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from custom_components.typesafe.client import TypeSafeClient, TypeSafeError
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
    """Test that canonical Question primitives are correctly converted to SDK models."""
    import typesafe_sdk

    mock_client.async_system_one.return_value = typesafe_sdk.SystemOneResponse(
        model="jev-latest",
        usage=typesafe_sdk.Usage(input_tokens=100, output_tokens=20),
        answers={
            "intent": typesafe_sdk.ChoiceAnswer(
                choice="HassTurnOn",
                confidence=0.95,
                probabilities={"HassTurnOn": 0.95},
            ),
            "is_compound": typesafe_sdk.NoulAnswer(noul=0.05),
            "quality": typesafe_sdk.ScoreAnswer(
                score=2.5,
                confidence=0.8,
                legend={0: "Low", 1: "Medium", 2: "High"},
                probabilities={0: 0.1, 1: 0.3, 2: 0.6},
            ),
        },
    )

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

    assert result.model == "jev-latest"
    assert result.usage == {"input_tokens": 100, "output_tokens": 20}
    assert isinstance(result.answers["intent"], ChoiceAnswer)
    assert result.answers["intent"].choice == "HassTurnOn"
    assert result.answers["intent"].confidence == 0.95
    assert isinstance(result.answers["is_compound"], NoulAnswer)
    assert result.answers["is_compound"].noul == 0.05
    assert isinstance(result.answers["quality"], ScoreAnswer)
    assert result.answers["quality"].score == 2.5


async def test_engine_async_predict_empty_choice_handled(
    engine: TypeSafeDecisionEngine, mock_client: AsyncMock
) -> None:
    """Test that empty string choice in SDK ChoiceAnswer is preserved."""
    import typesafe_sdk

    mock_client.async_system_one.return_value = typesafe_sdk.SystemOneResponse(
        model="jev-latest",
        usage=typesafe_sdk.Usage(input_tokens=50, output_tokens=10),
        answers={
            "intent": typesafe_sdk.ChoiceAnswer(
                choice="",
                confidence=0.9,
                probabilities={},
            ),
            "is_compound": typesafe_sdk.NoulAnswer(noul=0.1),
        },
    )

    questions = {
        "intent": ChoiceQuestion(instructions="Select intent", criteria={}),
        "is_compound": NoulQuestion(instructions="Is it compound?"),
    }

    result = await engine.async_predict(state="Test", questions=questions)

    assert isinstance(result.answers["intent"], ChoiceAnswer)
    assert result.answers["intent"].choice == ""
    assert result.answers["intent"].confidence == 0.9
    assert isinstance(result.answers["is_compound"], NoulAnswer)
    assert result.answers["is_compound"].noul == 0.1


async def test_engine_async_predict_with_sdk_primitives_directly(
    engine: TypeSafeDecisionEngine, mock_client: AsyncMock
) -> None:
    """Test passing typesafe_sdk primitives directly without canonical wrappers."""
    import typesafe_sdk

    mock_client.async_system_one.return_value = typesafe_sdk.SystemOneResponse(
        model="jev-latest",
        usage=typesafe_sdk.Usage(input_tokens=20, output_tokens=5),
        answers={
            "direct_choice": typesafe_sdk.ChoiceAnswer(
                choice="opt1",
                confidence=0.99,
                probabilities={"opt1": 0.99},
            ),
        },
    )

    questions = {
        "direct_choice": typesafe_sdk.Choice(
            instructions="Pick one", criteria={"opt1": "First", "opt2": "Second"}
        ),
    }

    result = await engine.async_predict(state="Test direct", questions=questions)

    assert mock_client.async_system_one.call_count == 1
    call_args = mock_client.async_system_one.call_args[1]
    assert call_args["questions"]["direct_choice"] is questions["direct_choice"]
    choice_answer = result.answers["direct_choice"]
    assert isinstance(choice_answer, ChoiceAnswer)
    assert choice_answer.choice == "opt1"
    assert choice_answer.confidence == 0.99


async def test_engine_async_predict_client_exception_propagates(
    engine: TypeSafeDecisionEngine, mock_client: AsyncMock
) -> None:
    """Test that TypeSafeError raised by client propagates from async_predict."""
    mock_client.async_system_one.side_effect = TypeSafeError("API down")

    with pytest.raises(TypeSafeError, match="API down"):
        await engine.async_predict(
            state="Test",
            questions={"q": ChoiceQuestion(instructions="Pick", criteria={})},
        )


async def test_engine_async_predict_empty_questions(
    engine: TypeSafeDecisionEngine, mock_client: AsyncMock
) -> None:
    """Test that predicting with empty questions returns empty answers."""
    import typesafe_sdk

    mock_client.async_system_one.return_value = typesafe_sdk.SystemOneResponse(
        model="jev-latest",
        usage=typesafe_sdk.Usage(input_tokens=0, output_tokens=0),
        answers={},
    )

    result = await engine.async_predict(state="Test", questions={})
    assert result.answers == {}
