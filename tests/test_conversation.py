"""Hermetic unit tests for TypeSafe conversation entity and intent routing."""

from __future__ import annotations

from homeassistant.components import conversation
from homeassistant.components.homeassistant.exposed_entities import (
    async_expose_entity,
)
from homeassistant.core import Context, HomeAssistant
from homeassistant.helpers import area_registry as ar, intent

from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.typesafe.const import (
    CONF_CONFIDENCE_THRESHOLD,
    CONF_FALLBACK_AGENT,
)
from tests.conftest import (
    MockBaseIntentHandler,
    MockFallbackAgent,
    MockLightSetIntentHandler,
    MockTurnOffIntentHandler,
    MockTurnOnIntentHandler,
    MockTypeSafeClient,
)


async def test_process_turn_on_entity_high_confidence(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_intent_handlers: dict[str, MockBaseIntentHandler],
) -> None:
    """Test high-confidence match resolves entity and executes HassTurnOn."""
    hass.states.async_set(
        "light.kitchen_lights", "off", {"friendly_name": "Kitchen Lights"}
    )
    mock_client.set_answers(
        {
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
        }
    )

    result = await conversation.async_converse(
        hass=hass,
        text="Turn on the kitchen lights",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )

    assert result.response.response_type is intent.IntentResponseType.ACTION_DONE
    handler = mock_intent_handlers["HassTurnOn"]
    assert isinstance(handler, MockTurnOnIntentHandler)
    assert len(handler.handled_intents) == 1
    intent_obj = handler.handled_intents[0]
    assert intent_obj.intent_type == "HassTurnOn"
    assert intent_obj.slots["name"]["value"] == "Kitchen Lights"


async def test_process_turn_off_entity_high_confidence(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_intent_handlers: dict[str, MockBaseIntentHandler],
) -> None:
    """Test high-confidence match executes HassTurnOff."""
    hass.states.async_set("switch.fan", "on", {"friendly_name": "Living Room Fan"})
    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOff",
                "confidence": 0.92,
                "probabilities": {"HassTurnOff": 0.92},
            },
            "target_entity": {
                "choice": "switch.fan",
                "confidence": 0.92,
            },
            "is_compound": {"noul": 0.02},
        }
    )

    result = await conversation.async_converse(
        hass=hass,
        text="Turn off the living room fan",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )

    assert result.response.response_type is intent.IntentResponseType.ACTION_DONE
    handler = mock_intent_handlers["HassTurnOff"]
    assert isinstance(handler, MockTurnOffIntentHandler)
    assert len(handler.handled_intents) == 1
    assert handler.handled_intents[0].slots["name"]["value"] == "Living Room Fan"


async def test_process_area_targeting(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_intent_handlers: dict[str, MockBaseIntentHandler],
) -> None:
    """Test area targeting formats area and domain slots."""
    area_reg = ar.async_get(hass)
    area = area_reg.async_create("Kitchen")

    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOn",
                "confidence": 0.91,
                "probabilities": {"HassTurnOn": 0.91},
            },
            "target_type": {"choice": "area", "confidence": 0.90},
            "target_area": {"choice": area.id, "confidence": 0.90},
            "is_compound": {"noul": 0.01},
        }
    )

    result = await conversation.async_converse(
        hass=hass,
        text="Turn on kitchen lights",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )

    assert result.response.response_type is intent.IntentResponseType.ACTION_DONE
    handler = mock_intent_handlers["HassTurnOn"]
    assert isinstance(handler, MockTurnOnIntentHandler)
    assert len(handler.handled_intents) == 1
    assert handler.handled_intents[0].slots["area"]["value"] == "Kitchen"
    assert handler.handled_intents[0].slots["domain"]["value"] == "light"


async def test_process_numeric_brightness_extraction(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_intent_handlers: dict[str, MockBaseIntentHandler],
) -> None:
    """Test numeric percentage slot extraction for brightness."""
    hass.states.async_set("light.bedroom", "on", {"friendly_name": "Bedroom Light"})
    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassLightSet",
                "confidence": 0.88,
                "probabilities": {"HassLightSet": 0.88},
            },
            "target_entity": {"choice": "light.bedroom", "confidence": 0.90},
            "is_compound": {"noul": 0.01},
        }
    )

    result = await conversation.async_converse(
        hass=hass,
        text="Set bedroom light to 50%",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )

    assert result.response.response_type is intent.IntentResponseType.ACTION_DONE
    handler = mock_intent_handlers["HassLightSet"]
    assert isinstance(handler, MockLightSetIntentHandler)
    assert len(handler.handled_intents) == 1
    assert handler.handled_intents[0].slots["name"]["value"] == "Bedroom Light"
    assert handler.handled_intents[0].slots["brightness"]["value"] == 50


async def test_process_low_confidence_with_fallback(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_fallback_agent: MockFallbackAgent,
) -> None:
    """Test low-confidence utterance escalates to fallback conversation agent."""
    hass.config_entries.async_update_entry(
        config_entry,
        options={
            CONF_CONFIDENCE_THRESHOLD: 0.7,
            CONF_FALLBACK_AGENT: "mock_fallback_agent",
        },
    )
    await hass.async_block_till_done()

    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOn",
                "confidence": 0.45,
                "probabilities": {"HassTurnOn": 0.45},
            },
            "is_compound": {"noul": 0.01},
        }
    )

    result = await conversation.async_converse(
        hass=hass,
        text="Turn on something maybe",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )

    assert len(mock_fallback_agent.calls) == 1
    assert mock_fallback_agent.calls[0].text == "Turn on something maybe"
    assert "Fallback response:" in result.response.speech["plain"]["speech"]


async def test_process_low_confidence_no_fallback(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """Test low-confidence without fallback returns NO_INTENT_MATCH error."""
    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOn",
                "confidence": 0.45,
                "probabilities": {"HassTurnOn": 0.45},
            },
            "is_compound": {"noul": 0.01},
        }
    )

    result = await conversation.async_converse(
        hass=hass,
        text="Turn on something maybe",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )

    assert result.response.response_type is intent.IntentResponseType.ERROR
    assert result.response.error_code is intent.IntentResponseErrorCode.NO_INTENT_MATCH


async def test_process_unmatched_intent_with_fallback(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_fallback_agent: MockFallbackAgent,
) -> None:
    """Test unmatched intent escalates to fallback agent."""
    hass.config_entries.async_update_entry(
        config_entry,
        options={
            CONF_FALLBACK_AGENT: "mock_fallback_agent",
        },
    )
    await hass.async_block_till_done()

    mock_client.set_answers(
        {
            "intent": {
                "choice": "unmatched",
                "confidence": 0.99,
                "probabilities": {"unmatched": 0.99},
            },
            "is_compound": {"noul": 0.01},
        }
    )

    result = await conversation.async_converse(
        hass=hass,
        text="What is the weather tomorrow?",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )

    assert len(mock_fallback_agent.calls) == 1
    assert mock_fallback_agent.calls[0].text == "What is the weather tomorrow?"
    assert "Fallback response:" in result.response.speech["plain"]["speech"]


async def test_process_unmatched_intent_no_fallback(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """Test unmatched intent without fallback returns NO_INTENT_MATCH."""
    mock_client.set_answers(
        {
            "intent": {
                "choice": "unmatched",
                "confidence": 0.99,
                "probabilities": {"unmatched": 0.99},
            },
            "is_compound": {"noul": 0.01},
        }
    )

    result = await conversation.async_converse(
        hass=hass,
        text="Tell me a joke",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )

    assert result.response.response_type is intent.IntentResponseType.ERROR
    assert result.response.error_code is intent.IntentResponseErrorCode.NO_INTENT_MATCH


async def test_process_compound_with_fallback(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_fallback_agent: MockFallbackAgent,
) -> None:
    """Test compound command escalates to fallback agent."""
    hass.config_entries.async_update_entry(
        config_entry,
        options={
            CONF_FALLBACK_AGENT: "mock_fallback_agent",
        },
    )
    await hass.async_block_till_done()

    mock_client.set_answers(
        {
            "is_compound": {"noul": 0.92},
        }
    )

    result = await conversation.async_converse(
        hass=hass,
        text="Turn on kitchen lights and lock front door",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )

    assert len(mock_fallback_agent.calls) == 1
    assert (
        mock_fallback_agent.calls[0].text
        == "Turn on kitchen lights and lock front door"
    )
    assert "Fallback response:" in result.response.speech["plain"]["speech"]


async def test_process_compound_no_fallback(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """Test compound command without fallback returns NO_INTENT_MATCH with multiple requests message."""
    mock_client.set_answers(
        {
            "is_compound": {"noul": 0.92},
        }
    )

    result = await conversation.async_converse(
        hass=hass,
        text="Turn on lights and open blinds",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )

    assert result.response.response_type is intent.IntentResponseType.ERROR
    assert result.response.error_code is intent.IntentResponseErrorCode.NO_INTENT_MATCH
    assert "multiple requests" in result.response.speech["plain"]["speech"]


async def test_entity_exposure_filtering(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_intent_handlers: dict[str, MockBaseIntentHandler],
) -> None:
    """Test unexposed entities are filtered out from target question candidates."""
    _ = mock_intent_handlers
    hass.states.async_set(
        "light.exposed_light", "off", {"friendly_name": "Exposed Light"}
    )
    hass.states.async_set(
        "light.hidden_light", "off", {"friendly_name": "Hidden Light"}
    )

    # Expose exposed_light and explicitly unexpose hidden_light
    async_expose_entity(hass, conversation.DOMAIN, "light.exposed_light", True)
    async_expose_entity(hass, conversation.DOMAIN, "light.hidden_light", False)

    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOn",
                "confidence": 0.95,
                "probabilities": {"HassTurnOn": 0.95},
            },
            "target_entity": {
                "choice": "light.exposed_light",
                "confidence": 0.95,
            },
            "is_compound": {"noul": 0.01},
        }
    )

    await conversation.async_converse(
        hass=hass,
        text="Turn on light",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )

    assert len(mock_client.calls) == 1
    target_criteria = mock_client.calls[0]["questions"]["target_entity"]["criteria"]
    assert "light.exposed_light" in target_criteria
    assert "light.hidden_light" not in target_criteria


async def test_intent_handle_error_handling(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """Test IntentHandleError results in FAILED_TO_HANDLE response."""
    hass.states.async_set(
        "light.failing_light", "off", {"friendly_name": "Failing Light"}
    )

    class FailingTurnOnHandler(intent.IntentHandler):
        intent_type = "HassTurnOn"

        async def async_handle(
            self, intent_obj: intent.Intent
        ) -> intent.IntentResponse:
            raise intent.IntentHandleError("Hardware communication failure")

    intent.async_register(hass, FailingTurnOnHandler())

    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOn",
                "confidence": 0.95,
                "probabilities": {"HassTurnOn": 0.95},
            },
            "target_entity": {
                "choice": "light.failing_light",
                "confidence": 0.95,
            },
            "is_compound": {"noul": 0.01},
        }
    )

    result = await conversation.async_converse(
        hass=hass,
        text="Turn on failing light",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )

    assert result.response.response_type is intent.IntentResponseType.ERROR
    assert result.response.error_code is intent.IntentResponseErrorCode.FAILED_TO_HANDLE
    assert "Hardware communication failure" in str(
        result.response.speech["plain"]["speech"]
    )
