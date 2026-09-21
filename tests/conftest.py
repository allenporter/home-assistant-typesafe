"""Fixtures for TypeSafe integration tests."""

from __future__ import annotations

from collections.abc import Generator, Mapping
from typing import Any
from unittest.mock import patch

import pytest

from homeassistant.components import conversation
from homeassistant.core import HomeAssistant
from homeassistant.helpers import intent
from homeassistant.setup import async_setup_component

from pytest_homeassistant_custom_component.common import MockConfigEntry
from typesafe_sdk import (
    Choice,
    ChoiceAnswer as SDKChoiceAnswer,
    Noul,
    NoulAnswer as SDKNoulAnswer,
    Score,
    ScoreAnswer as SDKScoreAnswer,
    SystemOneResponse,
    Usage,
)

from custom_components.typesafe.const import (
    CONF_API_KEY,
    CONF_MODEL,
    DEFAULT_NAME,
    DOMAIN,
)
from custom_components.typesafe.speculative.models import Question

pytest_plugins = [
    "tests.eval.fixtures_standard",
    "tests.eval.fixtures_typesafe",
]


@pytest.fixture(autouse=True)
def auto_enable_custom_integrations(
    enable_custom_integrations: None,
) -> Generator[None, None, None]:
    """Enable custom integrations."""
    _ = enable_custom_integrations
    yield


@pytest.fixture(autouse=True)
async def mock_dependencies(hass: HomeAssistant) -> None:
    """Set up homeassistant, intent, and conversation core components."""
    assert await async_setup_component(hass, "homeassistant", {})
    assert await async_setup_component(hass, "intent", {})
    assert await async_setup_component(hass, "conversation", {})
    await hass.async_block_till_done()


class MockTypeSafeClient:
    """Mock TypeSafeClient for hermetic unit testing."""

    def __init__(
        self,
        validate_result: bool = True,
        validate_error: Exception | None = None,
        answers: dict[str, Any] | None = None,
        evaluate_error: Exception | None = None,
    ) -> None:
        """Initialize mock client."""
        self.validate_result = validate_result
        self.validate_error = validate_error
        self.answers: dict[str, Any] | None = answers or {}
        self.evaluate_error = evaluate_error
        self.calls: list[dict[str, Any]] = []

    def set_answers(self, answers: dict[str, Any] | None) -> None:
        """Set answers returned by async_system_one."""
        self.answers = answers

    async def async_validate_key(self) -> bool:
        """Validate key mock."""
        if self.validate_error:
            raise self.validate_error
        return self.validate_result

    async def async_system_one(
        self,
        state: Any,
        questions: Mapping[str, Question],
        model: str | None = None,
    ) -> SystemOneResponse:
        """System one mock."""
        serialized = {
            k: v.model_dump() if isinstance(v, (Choice, Noul, Score)) else v
            for k, v in questions.items()
        }
        self.calls.append(
            {
                "state": state,
                "questions": serialized,
                "raw_questions": questions,
                "model": model or "jev-latest",
            }
        )
        if self.evaluate_error:
            raise self.evaluate_error

        sdk_answers: dict[str, Any] = {}
        if self.answers:
            for k, answer_val in self.answers.items():
                if isinstance(
                    answer_val, (SDKChoiceAnswer, SDKNoulAnswer, SDKScoreAnswer)
                ):
                    sdk_answers[k] = answer_val
                elif isinstance(answer_val, dict):
                    if "choice" in answer_val:
                        raw_choice = answer_val["choice"]
                        conf = float(answer_val.get("confidence", 1.0))
                        probs = answer_val.get("probabilities")
                        if probs is None:
                            probs = {raw_choice: conf} if raw_choice is not None else {}
                        sdk_answers[k] = SDKChoiceAnswer(
                            choice=raw_choice,
                            confidence=conf,
                            probabilities=probs,
                        )
                    elif "noul" in answer_val:
                        sdk_answers[k] = SDKNoulAnswer(noul=float(answer_val["noul"]))
                    elif "score" in answer_val:
                        sdk_answers[k] = SDKScoreAnswer(
                            score=float(answer_val["score"]),
                            confidence=float(answer_val.get("confidence", 1.0)),
                            legend=answer_val.get("legend", {}),
                            probabilities=answer_val.get("probabilities", {}),
                        )
                    else:
                        sdk_answers[k] = answer_val
                else:
                    sdk_answers[k] = answer_val

        return SystemOneResponse(
            model=model or "jev-latest",
            answers=sdk_answers,
            usage=Usage(input_tokens=10, output_tokens=5),
        )


@pytest.fixture(name="mock_client")
def mock_client_fixture() -> MockTypeSafeClient:
    """Provide a default MockTypeSafeClient."""
    return MockTypeSafeClient(
        validate_result=True,
        answers={
            "intent": {
                "choice": "HassTurnOn",
                "confidence": 0.95,
                "probabilities": {"HassTurnOn": 0.95},
            },
            "target_entity": {
                "choice": "light.kitchen_lights",
                "confidence": 0.95,
            },
            "is_compound": {"noul": 0.01},
        },
    )


@pytest.fixture(name="mock_typesafe_client", autouse=True)
def mock_typesafe_client_fixture(
    mock_client: MockTypeSafeClient,
) -> Generator[MockTypeSafeClient, None, None]:
    """Patch TypeSafeClient with mock_client in custom component modules."""
    with (
        patch(
            "custom_components.typesafe.TypeSafeClient",
            return_value=mock_client,
        ),
        patch(
            "custom_components.typesafe.config_flow.TypeSafeClient",
            return_value=mock_client,
        ),
    ):
        yield mock_client


class MockFallbackAgent(conversation.AbstractConversationAgent):
    """Mock fallback conversation agent for escalation tests."""

    def __init__(self) -> None:
        """Initialize mock agent."""
        self.calls: list[conversation.ConversationInput] = []

    @property
    def supported_languages(self) -> list[str]:
        """Return supported languages."""
        return ["en"]

    async def async_process(
        self, user_input: conversation.ConversationInput
    ) -> conversation.ConversationResult:
        """Record input and return dummy conversation response."""
        self.calls.append(user_input)
        res = intent.IntentResponse(language=user_input.language)
        res.async_set_speech(f"Fallback response: {user_input.text}")
        return conversation.ConversationResult(
            response=res, conversation_id=user_input.conversation_id
        )


@pytest.fixture(name="mock_fallback_agent")
def mock_fallback_agent_fixture(hass: HomeAssistant) -> MockFallbackAgent:
    """Register and return a mock fallback conversation agent."""
    agent = MockFallbackAgent()
    manager = conversation.get_agent_manager(hass)
    manager.async_set_agent("mock_fallback_agent", agent)
    return agent


class MockBaseIntentHandler(intent.IntentHandler):
    """Base mock intent handler with handled_intents list."""

    handled_intents: list[intent.Intent]


class MockTurnOnIntentHandler(MockBaseIntentHandler):
    """Mock handler for HassTurnOn."""

    intent_type = "HassTurnOn"
    description = "Turn on a device"

    def __init__(self) -> None:
        """Initialize handler."""
        self.handled_intents = []

    async def async_handle(self, intent_obj: intent.Intent) -> intent.IntentResponse:
        """Handle intent."""
        self.handled_intents.append(intent_obj)
        res = intent.IntentResponse(language=intent_obj.language)
        res.async_set_speech("Turned on device")
        return res


class MockTurnOffIntentHandler(MockBaseIntentHandler):
    """Mock handler for HassTurnOff."""

    intent_type = "HassTurnOff"
    description = "Turn off a device"

    def __init__(self) -> None:
        """Initialize handler."""
        self.handled_intents = []

    async def async_handle(self, intent_obj: intent.Intent) -> intent.IntentResponse:
        """Handle intent."""
        self.handled_intents.append(intent_obj)
        res = intent.IntentResponse(language=intent_obj.language)
        res.async_set_speech("Turned off device")
        return res


class MockLightSetIntentHandler(MockBaseIntentHandler):
    """Mock handler for HassLightSet."""

    intent_type = "HassLightSet"
    description = "Adjust brightness or color of a light"

    def __init__(self) -> None:
        """Initialize handler."""
        self.handled_intents = []

    async def async_handle(self, intent_obj: intent.Intent) -> intent.IntentResponse:
        """Handle intent."""
        self.handled_intents.append(intent_obj)
        res = intent.IntentResponse(language=intent_obj.language)
        res.async_set_speech("Adjusted light")
        return res


@pytest.fixture(name="mock_intent_handlers")
def mock_intent_handlers_fixture(
    hass: HomeAssistant,
) -> dict[str, MockBaseIntentHandler]:
    """Register dummy intent handlers for testing."""
    turn_on = MockTurnOnIntentHandler()
    turn_off = MockTurnOffIntentHandler()
    light_set = MockLightSetIntentHandler()

    intent.async_register(hass, turn_on)
    intent.async_register(hass, turn_off)
    intent.async_register(hass, light_set)

    return {
        "HassTurnOn": turn_on,
        "HassTurnOff": turn_off,
        "HassLightSet": light_set,
    }


@pytest.fixture(name="config_entry")
async def mock_config_entry(
    hass: HomeAssistant,
) -> MockConfigEntry:
    """Fixture to create and set up a TypeSafe configuration entry."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        title=DEFAULT_NAME,
        data={
            CONF_API_KEY: "test-api-key",
            CONF_MODEL: "jev-latest",
        },
        options={},
        entry_id="typesafe_test_entry",
    )

    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    return entry
