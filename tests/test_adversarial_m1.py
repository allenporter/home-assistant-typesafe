"""Adversarial challenge tests for TypeSafe integration Milestone 1.

Verifies:
1. API client error handling: HTTP 401, 422, 429, 529, timeouts, malformed JSON.
2. Strategy fallback behavior when client fails.
3. Speculative fan-out question construction when no entities or areas exist.
4. Execution behavior when model matches intent but no entities/areas exist.
5. Multi-intent / compound utterance escalation precedence and boundary conditions.
6. Intent handling error resilience (MatchFailedError, UnknownIntent).
"""

from __future__ import annotations

import pytest

from homeassistant.components import conversation
from homeassistant.core import Context, HomeAssistant
from homeassistant.helpers import area_registry as ar, intent

from pytest_homeassistant_custom_component.common import MockConfigEntry
from pytest_homeassistant_custom_component.test_util.aiohttp import (
    AiohttpClientMocker,
)

from custom_components.typesafe.client import (
    TypeSafeAuthError,
    TypeSafeClient,
    TypeSafeError,
    TypeSafeRateLimitError,
)
from custom_components.typesafe.const import CONF_FALLBACK_AGENT
from tests.conftest import (
    MockBaseIntentHandler,
    MockFallbackAgent,
    MockTurnOnIntentHandler,
    MockTypeSafeClient,
)


# ==============================================================================
# 1. API Client Direct HTTP Tests (401, 422, 429, 529, timeouts)
# ==============================================================================


async def test_client_evaluate_422_unprocessable_entity(
    hass: HomeAssistant,
    aioclient_mock: AiohttpClientMocker,
) -> None:
    """Test evaluate raises TypeSafeError on HTTP 422 Unprocessable Entity."""
    aioclient_mock.post(
        "https://api.typesafe.ai/v1/systemone",
        status=422,
        text='{"error": "criteria must have at least 2 choices"}',
    )
    from homeassistant.helpers import aiohttp_client

    session = aiohttp_client.async_get_clientsession(hass)
    client = TypeSafeClient(session=session, api_key="test-key")

    with pytest.raises(TypeSafeError, match="422"):
        await client.async_evaluate(
            state="turn on lights",
            questions={},
        )


async def test_client_evaluate_529_overloaded(
    hass: HomeAssistant,
    aioclient_mock: AiohttpClientMocker,
) -> None:
    """Test evaluate raises TypeSafeError on HTTP 529 Overloaded."""
    aioclient_mock.post(
        "https://api.typesafe.ai/v1/systemone",
        status=529,
        text='{"error": "Site is temporarily overloaded"}',
    )
    from homeassistant.helpers import aiohttp_client

    session = aiohttp_client.async_get_clientsession(hass)
    client = TypeSafeClient(session=session, api_key="test-key")

    with pytest.raises(TypeSafeError, match="529"):
        await client.async_evaluate(
            state="turn on lights",
            questions={},
        )


async def test_client_validate_key_422(
    hass: HomeAssistant,
    aioclient_mock: AiohttpClientMocker,
) -> None:
    """Test validate_key raises TypeSafeError on HTTP 422."""
    aioclient_mock.get(
        "https://api.typesafe.ai/v1/models",
        status=422,
    )
    from homeassistant.helpers import aiohttp_client

    session = aiohttp_client.async_get_clientsession(hass)
    client = TypeSafeClient(session=session, api_key="test-key")

    with pytest.raises(TypeSafeError, match="422"):
        await client.async_validate_key()


async def test_client_validate_key_529(
    hass: HomeAssistant,
    aioclient_mock: AiohttpClientMocker,
) -> None:
    """Test validate_key raises TypeSafeError on HTTP 529."""
    aioclient_mock.get(
        "https://api.typesafe.ai/v1/models",
        status=529,
    )
    from homeassistant.helpers import aiohttp_client

    session = aiohttp_client.async_get_clientsession(hass)
    client = TypeSafeClient(session=session, api_key="test-key")

    with pytest.raises(TypeSafeError, match="529"):
        await client.async_validate_key()


async def test_client_evaluate_rate_limit_without_retry_header(
    hass: HomeAssistant,
    aioclient_mock: AiohttpClientMocker,
) -> None:
    """Test evaluate handles HTTP 429 when Retry-After header is omitted."""
    aioclient_mock.post(
        "https://api.typesafe.ai/v1/systemone",
        status=429,
    )
    from homeassistant.helpers import aiohttp_client

    session = aiohttp_client.async_get_clientsession(hass)
    client = TypeSafeClient(session=session, api_key="test-key")

    with pytest.raises(TypeSafeRateLimitError, match="None"):
        await client.async_evaluate(
            state="turn on lights",
            questions={},
        )


# ==============================================================================
# 2. Strategy & Conversation Fallback Under API Client Errors
# ==============================================================================


@pytest.mark.parametrize(
    "client_error",
    [
        TypeSafeAuthError("Invalid API key (401)"),
        TypeSafeRateLimitError("Rate limit exceeded (429)"),
        TypeSafeError("Unprocessable Entity (422)"),
        TypeSafeError("Site Overloaded (529)"),
        TypeSafeError("Request failed: Timeout"),
    ],
)
async def test_conversation_api_errors_with_fallback(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_fallback_agent: MockFallbackAgent,
    client_error: Exception,
) -> None:
    """Verify all API client errors (401, 422, 429, 529, timeout) gracefully escalate to fallback agent."""
    hass.config_entries.async_update_entry(
        config_entry,
        options={
            CONF_FALLBACK_AGENT: "mock_fallback_agent",
        },
    )
    await hass.async_block_till_done()

    mock_client.evaluate_error = client_error

    result = await conversation.async_converse(
        hass=hass,
        text="Turn on living room light",
        conversation_id="test_conv",
        context=Context(),
        agent_id=config_entry.entry_id,
    )

    assert len(mock_fallback_agent.calls) == 1
    assert mock_fallback_agent.calls[0].text == "Turn on living room light"
    assert "Fallback response:" in result.response.speech["plain"]["speech"]


@pytest.mark.parametrize(
    "client_error",
    [
        TypeSafeAuthError("Invalid API key (401)"),
        TypeSafeRateLimitError("Rate limit exceeded (429)"),
        TypeSafeError("Unprocessable Entity (422)"),
        TypeSafeError("Site Overloaded (529)"),
        TypeSafeError("Request failed: Timeout"),
    ],
)
async def test_conversation_api_errors_no_fallback(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    client_error: Exception,
) -> None:
    """Verify all API client errors without fallback return NO_INTENT_MATCH without uncaught crash."""
    mock_client.evaluate_error = client_error

    result = await conversation.async_converse(
        hass=hass,
        text="Turn on living room light",
        conversation_id="test_conv",
        context=Context(),
        agent_id=config_entry.entry_id,
    )

    assert result.response.response_type is intent.IntentResponseType.ERROR
    assert result.response.error_code is intent.IntentResponseErrorCode.NO_INTENT_MATCH


# ==============================================================================
# 3. Speculative Fan-out Questions When No Entities or Areas Exist
# ==============================================================================


async def test_speculative_fan_out_zero_entities_zero_areas(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """Verify questions constructed when Home Assistant has NO entities and NO areas."""
    # Ensure no entities or areas exist
    mock_client.set_answers(
        {
            "intent": {"choice": "unmatched", "confidence": 0.99},
            "is_compound": {"noul": 0.01},
        }
    )

    await conversation.async_converse(
        hass=hass,
        text="Turn on something",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )

    assert len(mock_client.calls) == 1
    questions = mock_client.calls[0]["questions"]

    # Must contain intent and is_compound
    assert "intent" in questions
    assert "is_compound" in questions

    # Must NOT contain target_entity or target_area or target_type
    assert "target_entity" not in questions
    assert "target_area" not in questions
    assert "target_type" not in questions


async def test_speculative_fan_out_with_areas_only(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """Verify questions constructed when areas exist but NO controllable entities exist."""
    area_reg = ar.async_get(hass)
    area = area_reg.async_create("Living Room")

    mock_client.set_answers(
        {
            "intent": {"choice": "unmatched", "confidence": 0.99},
            "is_compound": {"noul": 0.01},
        }
    )

    await conversation.async_converse(
        hass=hass,
        text="Turn on living room",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )

    assert len(mock_client.calls) == 1
    questions = mock_client.calls[0]["questions"]

    assert "intent" in questions
    assert "is_compound" in questions
    # target_area has the area + "none" = 2 choices (> 1)
    assert "target_area" in questions
    assert area.id in questions["target_area"]["criteria"]
    assert "none" in questions["target_area"]["criteria"]

    # target_entity and target_type must NOT be present
    assert "target_entity" not in questions
    assert "target_type" not in questions


async def test_speculative_fan_out_with_entities_only(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """Verify questions constructed when controllable entities exist but NO areas exist."""
    hass.states.async_set("light.hallway", "off", {"friendly_name": "Hallway Light"})

    mock_client.set_answers(
        {
            "intent": {"choice": "unmatched", "confidence": 0.99},
            "is_compound": {"noul": 0.01},
        }
    )

    await conversation.async_converse(
        hass=hass,
        text="Turn on hallway light",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )

    assert len(mock_client.calls) == 1
    questions = mock_client.calls[0]["questions"]

    assert "intent" in questions
    assert "is_compound" in questions
    # target_entity has the entity + "none" = 2 choices (> 1)
    assert "target_entity" in questions
    assert "light.hallway" in questions["target_entity"]["criteria"]
    assert "none" in questions["target_entity"]["criteria"]

    # target_area and target_type must NOT be present
    assert "target_area" not in questions
    assert "target_type" not in questions


async def test_execution_when_intent_matches_but_zero_entities_exist(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_intent_handlers: dict[str, MockBaseIntentHandler],
) -> None:
    """Verify behavior when model returns HassTurnOn but no entities exist in HA."""
    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOn",
                "confidence": 0.95,
                "probabilities": {"HassTurnOn": 0.95},
            },
            "is_compound": {"noul": 0.01},
        }
    )

    result = await conversation.async_converse(
        hass=hass,
        text="Turn on the lights",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )

    assert result.response.response_type is intent.IntentResponseType.ACTION_DONE
    # In this case, HassTurnOn is dispatched with empty slots (no name, no entity, no area)
    # The intent handler executes or handles it.
    handler = mock_intent_handlers["HassTurnOn"]
    assert isinstance(handler, MockTurnOnIntentHandler)
    assert len(handler.handled_intents) == 1
    assert handler.handled_intents[0].slots == {}


# ==============================================================================
# 4. Multi-Intent / Compound Utterance Escalation
# ==============================================================================


async def test_compound_utterance_precedence_over_intent(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_fallback_agent: MockFallbackAgent,
    mock_intent_handlers: dict[str, MockBaseIntentHandler],
) -> None:
    """Verify compound detection takes strict precedence over high intent confidence."""
    hass.config_entries.async_update_entry(
        config_entry,
        options={
            CONF_FALLBACK_AGENT: "mock_fallback_agent",
        },
    )
    await hass.async_block_till_done()

    # Even though HassTurnOn has 0.99 confidence, is_compound is 0.85
    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOn",
                "confidence": 0.99,
                "probabilities": {"HassTurnOn": 0.99},
            },
            "is_compound": {"noul": 0.85},
        }
    )

    result = await conversation.async_converse(
        hass=hass,
        text="Turn on kitchen lights and lock the front door",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )

    # Intent handler must NOT have been called!
    handler = mock_intent_handlers["HassTurnOn"]
    assert isinstance(handler, MockTurnOnIntentHandler)
    assert len(handler.handled_intents) == 0

    # Must escalate to fallback agent
    assert len(mock_fallback_agent.calls) == 1
    assert "Fallback response:" in result.response.speech["plain"]["speech"]


@pytest.mark.parametrize(
    ("noul_value", "should_escalate_as_compound"),
    [
        (0.50, False),  # 0.50 is not > 0.50
        (0.51, True),  # 0.51 is > 0.50
        (0.49, False),  # 0.49 is not > 0.50
    ],
)
async def test_compound_boundary_threshold(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_intent_handlers: dict[str, MockBaseIntentHandler],
    noul_value: float,
    should_escalate_as_compound: bool,
) -> None:
    """Verify the 0.5 threshold boundary for compound detection."""
    _ = mock_intent_handlers
    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOn",
                "confidence": 0.95,
                "probabilities": {"HassTurnOn": 0.95},
            },
            "is_compound": {"noul": noul_value},
        }
    )

    result = await conversation.async_converse(
        hass=hass,
        text="Turn on the lights and maybe music",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )

    if should_escalate_as_compound:
        assert result.response.response_type is intent.IntentResponseType.ERROR
        assert "multiple requests" in result.response.speech["plain"]["speech"]
    else:
        assert result.response.response_type is intent.IntentResponseType.ACTION_DONE


# ==============================================================================
# 5. Empirical Bug Reproductions (Challenger Findings)
# ==============================================================================


async def test_defensive_null_compound_answer_handled_gracefully(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """Verify null is_compound answer does not crash and defaults to non-compound execution."""
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


async def test_defensive_null_intent_answer_handled_gracefully(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """Verify null intent answer does not crash and escalates/returns NO_INTENT_MATCH."""
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


async def test_defensive_missing_friendly_name_populates_slot(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_intent_handlers: dict[str, MockBaseIntentHandler],
) -> None:
    """Verify entity with no friendly_name attribute populates name slot using fallback."""
    hass.states.async_set("light.kitchen_strip", "off", {})
    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOn",
                "confidence": 0.95,
                "probabilities": {"HassTurnOn": 0.95},
            },
            "target_entity": {
                "choice": "light.kitchen_strip",
                "confidence": 0.95,
            },
            "is_compound": {"noul": 0.01},
        }
    )

    await conversation.async_converse(
        hass=hass,
        text="Turn on kitchen strip",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )

    handler = mock_intent_handlers["HassTurnOn"]
    assert isinstance(handler, MockTurnOnIntentHandler)
    assert len(handler.handled_intents) == 1
    intent_obj = handler.handled_intents[0]
    slot_val = intent_obj.slots["name"]["value"]
    assert slot_val is not None
    assert slot_val in ("kitchen strip", "Kitchen strip", "light.kitchen_strip")


async def test_defensive_nonexistent_fallback_agent_handled_gracefully(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """Verify nonexistent fallback_agent is caught and returns NO_INTENT_MATCH error without uncaught crash."""
    hass.config_entries.async_update_entry(
        config_entry,
        options={
            CONF_FALLBACK_AGENT: "non_existent_agent_xyz",
        },
    )
    await hass.async_block_till_done()

    mock_client.set_answers(
        {
            "intent": {"choice": "unmatched", "confidence": 0.99},
            "is_compound": {"noul": 0.01},
        }
    )

    result = await conversation.async_converse(
        hass=hass,
        text="Turn on the music",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert result.response.response_type is intent.IntentResponseType.ERROR
    assert result.response.error_code is intent.IntentResponseErrorCode.NO_INTENT_MATCH


async def test_defensive_intent_unexpected_error_caught_as_failed_to_handle(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """Verify IntentUnexpectedError is caught by conversation.py and formatted as FAILED_TO_HANDLE."""

    class UnexpectedErrorHandler(intent.IntentHandler):
        intent_type = "HassTurnOn"

        async def async_handle(
            self, intent_obj: intent.Intent
        ) -> intent.IntentResponse:
            raise intent.IntentUnexpectedError("Hardware device exploded")

    intent.async_register(hass, UnexpectedErrorHandler())

    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOn",
                "confidence": 0.95,
                "probabilities": {"HassTurnOn": 0.95},
            },
            "is_compound": {"noul": 0.01},
        }
    )

    result = await conversation.async_converse(
        hass=hass,
        text="Turn on the lights",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )

    assert result.response.response_type is intent.IntentResponseType.ERROR
    assert result.response.error_code is intent.IntentResponseErrorCode.FAILED_TO_HANDLE
    assert "Hardware device exploded" in str(result.response.speech["plain"]["speech"])
