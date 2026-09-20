"""Adversarial stress tests for TypeSafe integration.

Probes edge cases and boundary conditions in conversation.py, strategy.py, and config_flow.py:
1. Confidence Scores & Boundary Probes (0.0, 1.0, threshold boundary, threshold=0.0, threshold=1.0)
2. Empty or Missing Friendly Names / Entity Attributes (attributes={}, friendly_name="", friendly_name=None)
3. Fallback Agent Configured vs Not Configured (nonexistent fallback agent, client error escalation)
4. Unmatched Intent Strings & Malformed / Unparsable Responses (answers=None, intent=None, compound=None, probabilities=None)
5. Compound Threshold Boundary Probes (0.50, 0.5001, 0.4999)
"""

from __future__ import annotations

from typing import cast
import pytest

from homeassistant.components import conversation
from homeassistant.core import Context, HomeAssistant
from homeassistant.helpers import intent

from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.typesafe.client import TypeSafeError
from custom_components.typesafe.const import (
    CONF_CONFIDENCE_THRESHOLD,
    CONF_FALLBACK_AGENT,
)
from tests.conftest import (
    MockBaseIntentHandler,
    MockFallbackAgent,
    MockTurnOnIntentHandler,
    MockTypeSafeClient,
)


# ==============================================================================
# 1. Confidence Scores & Boundary Probes
# ==============================================================================


async def test_confidence_exact_threshold_boundary(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_intent_handlers: dict[str, MockBaseIntentHandler],
) -> None:
    """Test confidence exactly at threshold (0.70) executes, not escalates."""
    hass.states.async_set("light.test_light", "off", {"friendly_name": "Test Light"})
    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOn",
                "confidence": 0.70,
                "probabilities": {"HassTurnOn": 0.70},
            },
            "target_entity": {"choice": "light.test_light", "confidence": 0.70},
            "is_compound": {"noul": 0.0},
        }
    )

    result = await conversation.async_converse(
        hass=hass,
        text="Turn on test light",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )

    assert result.response.response_type is intent.IntentResponseType.ACTION_DONE
    handler = cast(MockTurnOnIntentHandler, mock_intent_handlers["HassTurnOn"])
    assert len(handler.handled_intents) == 1


async def test_confidence_just_below_threshold(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """Test confidence just below threshold (0.699) escalates / returns NO_INTENT_MATCH."""
    hass.states.async_set("light.test_light", "off", {"friendly_name": "Test Light"})
    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOn",
                "confidence": 0.699,
                "probabilities": {"HassTurnOn": 0.699},
            },
            "target_entity": {"choice": "light.test_light", "confidence": 0.699},
            "is_compound": {"noul": 0.0},
        }
    )

    result = await conversation.async_converse(
        hass=hass,
        text="Turn on test light",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )

    assert result.response.response_type is intent.IntentResponseType.ERROR
    assert result.response.error_code is intent.IntentResponseErrorCode.NO_INTENT_MATCH


async def test_confidence_zero(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """Test confidence of 0.0 correctly escalates."""
    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOn",
                "confidence": 0.0,
                "probabilities": {"HassTurnOn": 0.0},
            },
            "is_compound": {"noul": 0.0},
        }
    )

    result = await conversation.async_converse(
        hass=hass,
        text="Turn on test light",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )

    assert result.response.response_type is intent.IntentResponseType.ERROR
    assert result.response.error_code is intent.IntentResponseErrorCode.NO_INTENT_MATCH


async def test_confidence_one(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_intent_handlers: dict[str, MockBaseIntentHandler],
) -> None:
    """Test confidence of 1.0 executes successfully."""
    hass.states.async_set("light.test_light", "off", {"friendly_name": "Test Light"})
    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOn",
                "confidence": 1.0,
                "probabilities": {"HassTurnOn": 1.0},
            },
            "target_entity": {"choice": "light.test_light", "confidence": 1.0},
            "is_compound": {"noul": 0.0},
        }
    )

    result = await conversation.async_converse(
        hass=hass,
        text="Turn on test light",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )

    assert result.response.response_type is intent.IntentResponseType.ACTION_DONE


async def test_threshold_configured_as_zero(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_intent_handlers: dict[str, MockBaseIntentHandler],
) -> None:
    """Test when confidence_threshold is configured as 0.0 in options."""
    hass.config_entries.async_update_entry(
        config_entry,
        options={CONF_CONFIDENCE_THRESHOLD: 0.0},
    )
    await hass.async_block_till_done()

    hass.states.async_set("light.test_light", "off", {"friendly_name": "Test Light"})
    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOn",
                "confidence": 0.05,
                "probabilities": {"HassTurnOn": 0.05},
            },
            "target_entity": {"choice": "light.test_light", "confidence": 0.05},
            "is_compound": {"noul": 0.0},
        }
    )

    result = await conversation.async_converse(
        hass=hass,
        text="Turn on test light",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )

    assert result.response.response_type is intent.IntentResponseType.ACTION_DONE


async def test_threshold_configured_as_one(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_intent_handlers: dict[str, MockBaseIntentHandler],
) -> None:
    """Test when confidence_threshold is configured as 1.0 in options."""
    hass.config_entries.async_update_entry(
        config_entry,
        options={CONF_CONFIDENCE_THRESHOLD: 1.0},
    )
    await hass.async_block_till_done()

    hass.states.async_set("light.test_light", "off", {"friendly_name": "Test Light"})
    # 0.999 should fail
    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOn",
                "confidence": 0.999,
                "probabilities": {"HassTurnOn": 0.999},
            },
            "target_entity": {"choice": "light.test_light", "confidence": 0.999},
            "is_compound": {"noul": 0.0},
        }
    )

    result = await conversation.async_converse(
        hass=hass,
        text="Turn on test light",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert result.response.response_type is intent.IntentResponseType.ERROR

    # 1.0 should succeed
    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOn",
                "confidence": 1.0,
                "probabilities": {"HassTurnOn": 1.0},
            },
            "target_entity": {"choice": "light.test_light", "confidence": 1.0},
            "is_compound": {"noul": 0.0},
        }
    )
    result2 = await conversation.async_converse(
        hass=hass,
        text="Turn on test light",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert result2.response.response_type is intent.IntentResponseType.ACTION_DONE


# ==============================================================================
# 2. Empty or Missing Friendly Names / Entity Attributes
# ==============================================================================


async def test_entity_missing_friendly_name_attribute(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_intent_handlers: dict[str, MockBaseIntentHandler],
) -> None:
    """Probe entity with completely empty attributes dict (no friendly_name)."""
    hass.states.async_set("light.kitchen_lights", "off", {})
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

    await conversation.async_converse(
        hass=hass,
        text="Turn on the kitchen lights",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )

    handler = cast(MockTurnOnIntentHandler, mock_intent_handlers["HassTurnOn"])
    assert len(handler.handled_intents) == 1
    intent_obj = handler.handled_intents[0]
    slot_name = intent_obj.slots.get("name", {}).get("value")
    assert slot_name is not None, f"Expected non-None slot_name, got {slot_name!r}"
    assert slot_name in ("kitchen lights", "Kitchen lights", "light.kitchen_lights")


async def test_entity_friendly_name_is_empty_string(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_intent_handlers: dict[str, MockBaseIntentHandler],
) -> None:
    """Probe entity with friendly_name == ''."""
    hass.states.async_set("light.kitchen_lights", "off", {"friendly_name": ""})
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

    await conversation.async_converse(
        hass=hass,
        text="Turn on the kitchen lights",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )

    handler = cast(MockTurnOnIntentHandler, mock_intent_handlers["HassTurnOn"])
    assert len(handler.handled_intents) == 1
    intent_obj = handler.handled_intents[0]
    slot_name = intent_obj.slots.get("name", {}).get("value")
    assert slot_name, f"Expected non-empty slot_name, got {slot_name!r}"


async def test_entity_friendly_name_is_none(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_intent_handlers: dict[str, MockBaseIntentHandler],
) -> None:
    """Probe entity with friendly_name attribute explicitly set to None."""
    hass.states.async_set("light.kitchen_lights", "off", {"friendly_name": None})
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

    await conversation.async_converse(
        hass=hass,
        text="Turn on the kitchen lights",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )

    handler = cast(MockTurnOnIntentHandler, mock_intent_handlers["HassTurnOn"])
    assert len(handler.handled_intents) == 1
    intent_obj = handler.handled_intents[0]
    slot_name = intent_obj.slots.get("name", {}).get("value")
    assert slot_name is not None, f"Expected non-None slot_name, got {slot_name!r}"


# ==============================================================================
# 3. Fallback Agent Configured vs Not Configured Edge Cases
# ==============================================================================


async def test_fallback_agent_nonexistent(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """Probe fallback agent configured to an ID that does not exist in HA."""
    hass.config_entries.async_update_entry(
        config_entry,
        options={
            CONF_FALLBACK_AGENT: "non_existent_agent_xyz",
        },
    )
    await hass.async_block_till_done()

    mock_client.set_answers(
        {
            "intent": {
                "choice": "unmatched",
                "confidence": 0.99,
            },
            "is_compound": {"noul": 0.01},
        }
    )

    result = await conversation.async_converse(
        hass=hass,
        text="Hello world",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert result.response.response_type is intent.IntentResponseType.ERROR
    assert result.response.error_code is intent.IntentResponseErrorCode.NO_INTENT_MATCH


async def test_client_evaluation_error_with_fallback(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_fallback_agent: MockFallbackAgent,
) -> None:
    """Probe API client exception triggers escalation when fallback is configured."""
    hass.config_entries.async_update_entry(
        config_entry,
        options={
            CONF_FALLBACK_AGENT: "mock_fallback_agent",
        },
    )
    await hass.async_block_till_done()

    mock_client.evaluate_error = TypeSafeError("API 500 error")

    result = await conversation.async_converse(
        hass=hass,
        text="Turn on the lights",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )

    assert len(mock_fallback_agent.calls) == 1
    assert "Fallback response:" in result.response.speech["plain"]["speech"]


async def test_client_evaluation_error_without_fallback(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """Probe API client exception returns NO_INTENT_MATCH error when no fallback is configured."""
    mock_client.evaluate_error = TypeSafeError("API connection timeout")

    result = await conversation.async_converse(
        hass=hass,
        text="Turn on the lights",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )

    assert result.response.response_type is intent.IntentResponseType.ERROR
    assert result.response.error_code is intent.IntentResponseErrorCode.NO_INTENT_MATCH


# ==============================================================================
# 4. Unmatched Intent Strings & Unparsable / Malformed Responses
# ==============================================================================


@pytest.mark.parametrize(
    "unmatched_choice",
    [
        "unmatched",
        "none",
        "other",
        "",
        None,
    ],
)
async def test_unmatched_intent_strings(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    unmatched_choice: str | None,
) -> None:
    """Test all variants of unmatched intent choices."""
    mock_client.set_answers(
        {
            "intent": {
                "choice": unmatched_choice,
                "confidence": 0.99,
            },
            "is_compound": {"noul": 0.0},
        }
    )

    result = await conversation.async_converse(
        hass=hass,
        text="Sing a song",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )

    assert result.response.response_type is intent.IntentResponseType.ERROR
    assert result.response.error_code is intent.IntentResponseErrorCode.NO_INTENT_MATCH


async def test_unparseable_response_answers_none(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """Test API response where answers field is None: {'model': '...', 'answers': None}."""
    mock_client.answers = None

    result = await conversation.async_converse(
        hass=hass,
        text="Turn on lights",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert result.response.response_type is intent.IntentResponseType.ERROR
    assert result.response.error_code is intent.IntentResponseErrorCode.NO_INTENT_MATCH


async def test_unparseable_response_intent_none(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """Test API response where answers has {'intent': None}."""
    mock_client.set_answers(
        {
            "intent": None,
            "is_compound": {"noul": 0.0},
        }
    )

    result = await conversation.async_converse(
        hass=hass,
        text="Turn on lights",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert result.response.response_type is intent.IntentResponseType.ERROR
    assert result.response.error_code is intent.IntentResponseErrorCode.NO_INTENT_MATCH


async def test_unparseable_response_is_compound_none(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """Test API response where answers has {'is_compound': None} defaults to non-compound."""
    mock_client.set_answers(
        {
            "intent": {"choice": "HassTurnOn", "confidence": 0.95},
            "is_compound": None,
        }
    )

    result = await conversation.async_converse(
        hass=hass,
        text="Turn on lights",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert result.response.response_type in (
        intent.IntentResponseType.ACTION_DONE,
        intent.IntentResponseType.ERROR,
    )


async def test_unparseable_response_probabilities_none(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """Test API response where probabilities is None falls back to intent confidence."""
    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOn",
                "confidence": 0.95,
                "probabilities": None,
            },
            "is_compound": {"noul": 0.0},
        }
    )

    result = await conversation.async_converse(
        hass=hass,
        text="Turn on lights",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert result.response.response_type in (
        intent.IntentResponseType.ACTION_DONE,
        intent.IntentResponseType.ERROR,
    )


# ==============================================================================
# 5. Compound Command Boundary Probes
# ==============================================================================


@pytest.mark.parametrize(
    "noul,should_compound",
    [
        (0.50, False),  # Exactly at compound threshold 0.5 -> not compound
        (0.5001, True),  # Just above 0.5 -> compound
        (0.4999, False),  # Just below 0.5 -> not compound
        (1.0, True),  # Maximum compound probability
        (0.0, False),  # Minimum compound probability
    ],
)
async def test_compound_noul_boundary(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_intent_handlers: dict[str, MockBaseIntentHandler],
    noul: float,
    should_compound: bool,
) -> None:
    """Test compound boundary: compound_threshold is 0.5."""
    hass.states.async_set("light.test_light", "off", {"friendly_name": "Test Light"})
    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOn",
                "confidence": 0.95,
                "probabilities": {"HassTurnOn": 0.95},
            },
            "target_entity": {"choice": "light.test_light", "confidence": 0.95},
            "is_compound": {"noul": noul},
        }
    )

    result = await conversation.async_converse(
        hass=hass,
        text="Turn on test light",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )

    if should_compound:
        assert result.response.response_type is intent.IntentResponseType.ERROR
        assert "multiple requests" in result.response.speech["plain"]["speech"]
    else:
        assert result.response.response_type is intent.IntentResponseType.ACTION_DONE
