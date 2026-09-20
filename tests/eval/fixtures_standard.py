"""Standard, vendor-agnostic Home Assistant fixtures for evaluation suites.

These fixtures provide realistic synthetic home contexts, standard intent handlers,
and mock fallback conversation agents independent of any specific AI model or provider.
"""

from __future__ import annotations

import pytest
from homeassistant.components import conversation
from homeassistant.core import HomeAssistant
from homeassistant.helpers import intent

from custom_components.typesafe.speculative.context import DecisionContext
from tests.common.fixture_loader import (
    DeviceActionCase,
    load_device_action_cases,
    load_synthetic_home_fixtures,
    register_standard_intents,
)


class MockFallbackAgent(conversation.AbstractConversationAgent):
    """Mock fallback conversation agent for escalation tests."""

    def __init__(self) -> None:
        """Initialize mock fallback agent."""
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


class MockClimateIntentHandler(intent.IntentHandler):
    """Mock handler for HassClimateSetTemperature."""

    intent_type = "HassClimateSetTemperature"
    description = "Set target temperature for thermostat or climate device"

    def __init__(self) -> None:
        """Initialize handler."""
        self.handled_intents: list[intent.Intent] = []

    async def async_handle(self, intent_obj: intent.Intent) -> intent.IntentResponse:
        """Handle climate intent."""
        self.handled_intents.append(intent_obj)
        res = intent.IntentResponse(language=intent_obj.language)
        res.async_set_speech("Target temperature set")
        return res


@pytest.fixture(name="farmhouse_context")
def farmhouse_context_fixture(hass: HomeAssistant) -> DecisionContext:
    """Load the full family farmhouse synthetic home fixture context."""
    return load_synthetic_home_fixtures(hass)


@pytest.fixture(name="climate_handler")
def climate_handler_fixture(hass: HomeAssistant) -> MockClimateIntentHandler:
    """Fixture to register and return a MockClimateIntentHandler."""
    handler = MockClimateIntentHandler()
    intent.async_register(hass, handler)
    return handler


@pytest.fixture(name="standard_intents")
def standard_intents_fixture(hass: HomeAssistant) -> None:
    """Fixture ensuring standard HA dummy intents are registered."""
    register_standard_intents(hass)


@pytest.fixture(name="device_action_cases")
def device_action_cases_fixture() -> list[DeviceActionCase]:
    """Fixture providing all synthetic home device action cases."""
    return load_device_action_cases()
