"""Adversarial stress and edge-case empirical verification for Milestone 1.

Probes focus areas:
1. Non-dict JSON API responses (list, str, int, None, bool, invalid syntax).
2. HTTP 401, 422, 429, 529, and asyncio timeout errors across client and conversation.
3. 0 exposed entities and 0 areas boundary states.
4. Dynamic reload on options update (confidence threshold, fallback agent, unload).
"""

from __future__ import annotations

import asyncio
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from homeassistant.components import conversation
from homeassistant.core import Context, HomeAssistant
from homeassistant.helpers import (
    aiohttp_client,
    area_registry as ar,
    entity_registry as er,
    intent,
)

from pytest_homeassistant_custom_component.common import MockConfigEntry
from pytest_homeassistant_custom_component.test_util.aiohttp import AiohttpClientMocker

from custom_components.typesafe.client import (
    TypeSafeAuthError,
    TypeSafeClient,
    TypeSafeError,
    TypeSafeRateLimitError,
)
from custom_components.typesafe.const import (
    CONF_CONFIDENCE_THRESHOLD,
    CONF_FALLBACK_AGENT,
)
from custom_components.typesafe.strategy import (
    DecisionStrategy,
    StrategyContext,
)
from tests.conftest import (
    MockBaseIntentHandler,
    MockFallbackAgent,
    MockTurnOnIntentHandler,
    MockTypeSafeClient,
)


# ==============================================================================
# 1. Non-dict JSON API responses
# ==============================================================================


@pytest.mark.parametrize(
    "payload,expected_type_name",
    [
        ([], "list"),
        ([{"status": "ok"}], "list"),
        ("string-response", "str"),
        (42, "int"),
        (True, "bool"),
    ],
)
async def test_client_evaluate_non_dict_json_raises_typesafe_error(
    hass: HomeAssistant,
    aioclient_mock: AiohttpClientMocker,
    payload: Any,
    expected_type_name: str,
) -> None:
    """Verify TypeSafeClient raises TypeSafeError when server returns non-dict JSON."""
    aioclient_mock.post(
        "https://api.typesafe.ai/v1/systemone",
        status=200,
        json=payload,
    )
    session = aiohttp_client.async_get_clientsession(hass)
    client = TypeSafeClient(session=session, api_key="test-key")

    with pytest.raises(
        TypeSafeError,
        match=rf"Invalid response from TypeSafe: expected dict, got {expected_type_name}",
    ):
        await client.async_evaluate(state="turn on light", questions={})


async def test_client_evaluate_null_json_body_raises_typesafe_error(
    hass: HomeAssistant,
    aioclient_mock: AiohttpClientMocker,
) -> None:
    """Verify TypeSafeClient raises TypeSafeError when server returns JSON null."""
    aioclient_mock.post(
        "https://api.typesafe.ai/v1/systemone",
        status=200,
        text="null",
        headers={"Content-Type": "application/json"},
    )
    session = aiohttp_client.async_get_clientsession(hass)
    client = TypeSafeClient(session=session, api_key="test-key")

    with pytest.raises(
        TypeSafeError,
        match=r"Invalid response from TypeSafe: expected dict, got NoneType",
    ):
        await client.async_evaluate(state="turn on light", questions={})


async def test_client_evaluate_malformed_json_syntax_raises_typesafe_error(
    hass: HomeAssistant,
    aioclient_mock: AiohttpClientMocker,
) -> None:
    """Verify TypeSafeClient raises TypeSafeError on invalid JSON syntax."""
    aioclient_mock.post(
        "https://api.typesafe.ai/v1/systemone",
        status=200,
        text="<!DOCTYPE html><html>Server Error</html>",
        headers={"Content-Type": "application/json"},
    )
    session = aiohttp_client.async_get_clientsession(hass)
    client = TypeSafeClient(session=session, api_key="test-key")

    with pytest.raises(TypeSafeError, match="Request failed"):
        await client.async_evaluate(state="turn on light", questions={})


@pytest.mark.parametrize(
    "malformed_response",
    [
        None,
        [],
        [1, 2, 3],
        "hello",
        123,
        False,
        {"answers": None},
        {"answers": []},
        {"answers": "not a dict"},
        {"answers": {"intent": None}},
        {"answers": {"intent": "not a dict"}},
        {"answers": {"intent": []}},
        {"answers": {"intent": {"choice": None}}},
        {"answers": {"intent": {"choice": "HassTurnOn", "confidence": None}}},
        {"answers": {"intent": {"choice": "HassTurnOn", "confidence": "not-a-number"}}},
        {
            "answers": {
                "intent": {"choice": "HassTurnOn", "confidence": 0.95},
                "is_compound": None,
            }
        },
        {
            "answers": {
                "intent": {"choice": "HassTurnOn", "confidence": 0.95},
                "is_compound": {"noul": None},
            }
        },
        {
            "answers": {
                "intent": {"choice": "HassTurnOn", "confidence": 0.95},
                "is_compound": {"noul": "abc"},
            }
        },
        {
            "answers": {
                "intent": {"choice": "HassTurnOn", "confidence": 0.95},
                "target_type": None,
            }
        },
        {
            "answers": {
                "intent": {"choice": "HassTurnOn", "confidence": 0.95},
                "target_area": None,
            }
        },
        {
            "answers": {
                "intent": {"choice": "HassTurnOn", "confidence": 0.95},
                "target_entity": None,
            }
        },
    ],
)
def test_strategy_parse_evaluation_response_malformed_resilience(
    hass: HomeAssistant,
    malformed_response: Any,
) -> None:
    """Verify strategy._parse_evaluation_response never raises an exception on malformed payloads."""
    strategy = DecisionStrategy()
    context = StrategyContext(
        hass=hass,
        area_registry=ar.async_get(hass),
        entity_registry=er.async_get(hass),
        states=[],
        home_name="Test Home",
    )

    result = strategy._parse_evaluation_response(
        response=malformed_response,
        utterance="turn on lights",
        context=context,
    )
    assert result is not None
    # If the payload lacks valid intent or answers, it must cleanly escalate
    if not isinstance(malformed_response, dict) or not isinstance(
        malformed_response.get("answers"), dict
    ):
        assert result.should_escalate is True
        assert result.intent_name is None


# ==============================================================================
# 2. HTTP 401, 422, 429, 529, timeouts
# ==============================================================================


@pytest.mark.parametrize(
    "status,headers,expected_exc,match_str",
    [
        (401, {}, TypeSafeAuthError, r"Invalid API key"),
        (422, {}, TypeSafeError, r"System One evaluation failed \(422\)"),
        (
            429,
            {"Retry-After": "60"},
            TypeSafeRateLimitError,
            r"Rate limited\. Retry after: 60",
        ),
        (429, {}, TypeSafeRateLimitError, r"Rate limited\. Retry after: None"),
        (529, {}, TypeSafeError, r"System One evaluation failed \(529\)"),
    ],
)
async def test_client_evaluate_http_error_statuses(
    hass: HomeAssistant,
    aioclient_mock: AiohttpClientMocker,
    status: int,
    headers: dict[str, str],
    expected_exc: type[Exception],
    match_str: str,
) -> None:
    """Verify TypeSafeClient HTTP status error mapping."""
    aioclient_mock.post(
        "https://api.typesafe.ai/v1/systemone",
        status=status,
        headers=headers,
        text="Error body details",
    )
    session = aiohttp_client.async_get_clientsession(hass)
    client = TypeSafeClient(session=session, api_key="test-key")

    with pytest.raises(expected_exc, match=match_str):
        await client.async_evaluate(state="turn on light", questions={})


async def test_client_evaluate_timeout_handling() -> None:
    """Verify TypeSafeClient maps asyncio.TimeoutError to TypeSafeError."""
    session = MagicMock()
    session.post.side_effect = asyncio.TimeoutError("Connection timed out")
    client = TypeSafeClient(session=session, api_key="test-key")

    with pytest.raises(TypeSafeError, match="Request failed: Connection timed out"):
        await client.async_evaluate(state="turn on light", questions={})


@pytest.mark.parametrize(
    "client_exc",
    [
        TypeSafeAuthError("Invalid API key"),
        TypeSafeError("System One evaluation failed (422)"),
        TypeSafeRateLimitError("Rate limited. Retry after: 30"),
        TypeSafeError("System One evaluation failed (529)"),
        TypeSafeError("Request failed: Timeout"),
    ],
)
async def test_conversation_handles_all_client_errors_with_and_without_fallback(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_fallback_agent: MockFallbackAgent,
    client_exc: Exception,
) -> None:
    """Verify conversation pipeline degrades cleanly on all client errors."""
    mock_client.evaluate_error = client_exc

    # 1. Without fallback agent configured
    result_no_fallback = await conversation.async_converse(
        hass=hass,
        text="Turn on kitchen light",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert result_no_fallback.response.response_type is intent.IntentResponseType.ERROR
    assert (
        result_no_fallback.response.error_code
        is intent.IntentResponseErrorCode.NO_INTENT_MATCH
    )

    # 2. With fallback agent configured
    hass.config_entries.async_update_entry(
        config_entry,
        options={
            CONF_FALLBACK_AGENT: "mock_fallback_agent",
        },
    )
    await hass.async_block_till_done()

    result_with_fallback = await conversation.async_converse(
        hass=hass,
        text="Turn on kitchen light",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert (
        result_with_fallback.response.response_type
        is intent.IntentResponseType.ACTION_DONE
    )
    assert (
        "Fallback response" in result_with_fallback.response.speech["plain"]["speech"]
    )
    assert len(mock_fallback_agent.calls) == 1


# ==============================================================================
# 3. 0 exposed entities and 0 areas
# ==============================================================================


async def test_zero_exposed_entities_and_zero_areas_question_schema(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_intent_handlers: dict[str, MockBaseIntentHandler],
) -> None:
    """Verify questions payload when 0 entities are exposed and 0 areas exist."""
    # Ensure no areas exist
    area_reg = ar.async_get(hass)
    assert len(area_reg.areas) == 0

    # Ensure no controllable entities exist in HA states
    from custom_components.typesafe.strategy import CONTROLLABLE_DOMAINS

    controllable_count = sum(
        1 for s in hass.states.async_all() if s.domain in CONTROLLABLE_DOMAINS
    )
    assert controllable_count == 0

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
        text="Turn on everything",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )

    # Inspect the questions sent to TypeSafe
    assert len(mock_client.calls) == 1
    questions = mock_client.calls[0]["questions"]

    # Must contain intent and is_compound
    assert "intent" in questions
    assert "is_compound" in questions

    # Must NOT contain target_entity, target_area, or target_type
    assert "target_entity" not in questions
    assert "target_area" not in questions
    assert "target_type" not in questions

    # Intent criteria must have at least 2 choices
    assert len(questions["intent"]["criteria"]) >= 2
    assert "unmatched" in questions["intent"]["criteria"]
    assert "HassTurnOn" in questions["intent"]["criteria"]

    # Intent executes successfully without slots
    assert result.response.response_type is intent.IntentResponseType.ACTION_DONE
    handler = cast(MockTurnOnIntentHandler, mock_intent_handlers["HassTurnOn"])
    assert len(handler.handled_intents) == 1
    assert handler.handled_intents[0].slots == {}


async def test_non_controllable_entities_are_excluded_from_criteria(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """Verify sensor/weather domains are excluded from target_entity questions."""
    hass.states.async_set("sensor.outdoor_temperature", "21.5")
    hass.states.async_set("weather.home", "sunny")

    mock_client.set_answers(
        {
            "intent": {"choice": "unmatched", "confidence": 0.99},
            "is_compound": {"noul": 0.01},
        }
    )

    await conversation.async_converse(
        hass=hass,
        text="What is the weather?",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )

    questions = mock_client.calls[0]["questions"]
    # Because there are no controllable entities, target_entity must be omitted
    assert "target_entity" not in questions


# ==============================================================================
# 4. Dynamic reload on options update
# ==============================================================================


async def test_dynamic_reload_confidence_threshold_and_fallback(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_fallback_agent: MockFallbackAgent,
    mock_intent_handlers: dict[str, MockBaseIntentHandler],
) -> None:
    """Verify options update dynamically reloads confidence threshold and fallback agent."""
    hass.states.async_set(
        "light.living_room", "off", {"friendly_name": "Living Room Light"}
    )

    # Setup: Utterance with confidence 0.75
    # Default threshold is 0.70 -> Should succeed
    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOn",
                "confidence": 0.75,
                "probabilities": {"HassTurnOn": 0.75},
            },
            "target_entity": {
                "choice": "light.living_room",
                "confidence": 0.75,
            },
            "is_compound": {"noul": 0.01},
        }
    )

    res1 = await conversation.async_converse(
        hass=hass,
        text="Turn on living room light",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert res1.response.response_type is intent.IntentResponseType.ACTION_DONE
    handler = cast(MockTurnOnIntentHandler, mock_intent_handlers["HassTurnOn"])
    assert len(handler.handled_intents) == 1

    # Update 1: Raise threshold to 0.85 (without fallback agent)
    # The same 0.75 confidence utterance should now be rejected as NO_INTENT_MATCH
    hass.config_entries.async_update_entry(
        config_entry,
        options={
            CONF_CONFIDENCE_THRESHOLD: 0.85,
        },
    )
    await hass.async_block_till_done()

    assert config_entry.runtime_data.strategy.confidence_threshold == 0.85

    res2 = await conversation.async_converse(
        hass=hass,
        text="Turn on living room light",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert res2.response.response_type is intent.IntentResponseType.ERROR
    assert res2.response.error_code is intent.IntentResponseErrorCode.NO_INTENT_MATCH

    # Update 2: Configure fallback agent
    # The same 0.75 confidence utterance should now escalate to the fallback agent
    hass.config_entries.async_update_entry(
        config_entry,
        options={
            CONF_CONFIDENCE_THRESHOLD: 0.85,
            CONF_FALLBACK_AGENT: "mock_fallback_agent",
        },
    )
    await hass.async_block_till_done()

    res3 = await conversation.async_converse(
        hass=hass,
        text="Turn on living room light",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert res3.response.response_type is intent.IntentResponseType.ACTION_DONE
    assert (
        "Fallback response: Turn on living room light"
        in res3.response.speech["plain"]["speech"]
    )
    assert len(mock_fallback_agent.calls) == 1


# ==============================================================================
# 5. Additional Boundary & Edge Case Stress Probes
# ==============================================================================


async def test_empty_utterance_handling(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """Verify empty or whitespace utterances do not crash strategy or conversation."""
    mock_client.set_answers(
        {
            "intent": {"choice": "unmatched", "confidence": 0.99},
            "is_compound": {"noul": 0.0},
        }
    )

    result = await conversation.async_converse(
        hass=hass,
        text="   ",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert result.response.response_type is intent.IntentResponseType.ERROR
    assert result.response.error_code is intent.IntentResponseErrorCode.NO_INTENT_MATCH


async def test_target_entity_not_in_states_falls_back_to_entity_id(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_intent_handlers: dict[str, MockBaseIntentHandler],
) -> None:
    """Verify target entity not registered in hass states passes entity_id as slot name."""
    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOn",
                "confidence": 0.95,
                "probabilities": {"HassTurnOn": 0.95},
            },
            "target_entity": {
                "choice": "light.phantom_device",
                "confidence": 0.95,
            },
            "is_compound": {"noul": 0.01},
        }
    )

    result = await conversation.async_converse(
        hass=hass,
        text="Turn on phantom device",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert result.response.response_type is intent.IntentResponseType.ACTION_DONE
    handler = cast(MockTurnOnIntentHandler, mock_intent_handlers["HassTurnOn"])
    assert len(handler.handled_intents) == 1
    assert handler.handled_intents[0].slots["name"]["value"] == "light.phantom_device"


async def test_empty_answers_dict_escalates(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """Verify completely empty answers dict {} escalates gracefully without crash."""
    mock_client.set_answers({})

    result = await conversation.async_converse(
        hass=hass,
        text="Turn on lights",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert result.response.response_type is intent.IntentResponseType.ERROR
    assert result.response.error_code is intent.IntentResponseErrorCode.NO_INTENT_MATCH
