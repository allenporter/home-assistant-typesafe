"""Unit tests for smart home candidate discovery, token matching, and slot inspection heuristics."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import voluptuous as vol

from homeassistant.core import HomeAssistant
from homeassistant.helpers import area_registry as ar, entity_registry as er

from custom_components.typesafe.speculative.strategy.base import StrategyContext
from custom_components.typesafe.speculative.strategy.discovery import (
    CONTROLLABLE_DOMAINS,
    can_fulfill_intent,
    discover_intents,
    get_allowed_domains_for_intents,
    get_handler_slot_info,
    lexical_score,
    rank_areas,
    rank_entities,
    token_match,
    tokenize,
)


@pytest.fixture(name="empty_context")
def empty_context_fixture(hass: HomeAssistant) -> StrategyContext:
    """Fixture providing a StrategyContext backed by real HA registries."""
    return StrategyContext(
        hass=hass,
        area_registry=ar.async_get(hass),
        entity_registry=er.async_get(hass),
        states=[],
        home_name="My Home",
    )


def test_tokenize_and_lexical_score() -> None:
    """Test text tokenization and lexical scoring."""
    tokens = tokenize("Turn on the Kitchen Light!")
    assert "turn" in tokens
    assert "kitchen" in tokens
    assert "light" in tokens

    score_match = lexical_score(
        tokens, "Kitchen Ceiling Light", "turn on the kitchen light"
    )
    assert score_match > 0.0

    score_mismatch = lexical_score(tokens, "Basement Fan", "turn on the kitchen light")
    assert score_mismatch == 0.0


def test_token_match() -> None:
    """Test token equality and prefix matching."""
    assert token_match("light", "light")
    assert token_match("light", "lights")
    assert token_match("lights", "light")
    assert not token_match("fan", "light")
    assert not token_match("in", "inside")  # Less than 3 chars


def test_intent_handler_slot_info_and_can_fulfill() -> None:
    """Test slot introspection and can_fulfill_intent filtering."""
    handler_valid = MagicMock()
    handler_valid.slot_schema = {
        vol.Required("name"): str,
        vol.Optional("area"): str,
    }
    supported, required = get_handler_slot_info(handler_valid)
    assert "name" in required
    assert "area" in supported
    assert can_fulfill_intent(handler_valid)

    handler_unsupported = MagicMock()
    handler_unsupported.slot_schema = {
        vol.Required("unsupported_custom_slot"): str,
    }
    supported, required = get_handler_slot_info(handler_unsupported)
    assert "unsupported_custom_slot" in required
    assert not can_fulfill_intent(handler_unsupported)


def test_get_allowed_domains_for_intents() -> None:
    """Test deriving allowed domains from candidate intents and handlers."""
    mock_media = MagicMock()
    mock_media.platforms = {"media_player"}

    mock_light = MagicMock()
    mock_light.platforms = {"light"}

    handlers = {
        "HassMediaPause": mock_media,
        "HassLightSet": mock_light,
    }

    # Specialized intents narrow to platform
    assert get_allowed_domains_for_intents(["HassMediaPause"], handlers) == {
        "media_player"
    }
    assert get_allowed_domains_for_intents(["HassLightSet"], handlers) == {"light"}

    # Generic on/off intents fall back to all controllable domains
    turn_on_domains = get_allowed_domains_for_intents(["HassTurnOn"], handlers)
    assert turn_on_domains == set(CONTROLLABLE_DOMAINS)
    assert "light" in turn_on_domains
    assert "cover" in turn_on_domains
    assert "valve" in turn_on_domains
    assert "sensor" not in turn_on_domains

    # Informational intents are ignored
    info_domains = get_allowed_domains_for_intents(["HassGetState"], handlers)
    assert info_domains == set(CONTROLLABLE_DOMAINS)


def test_discover_intents_fallback(empty_context: StrategyContext) -> None:
    """Test discover_intents returns canonical defaults when no handlers exist."""
    intents = discover_intents(empty_context, "Turn on the light")
    assert "HassTurnOn" in intents
    assert "unmatched" in intents


def test_rank_areas_and_entities_with_area_boosting(
    empty_context: StrategyContext,
) -> None:
    """Test area ranking and area-boosted entity ranking."""
    kitchen_area = empty_context.area_registry.async_create("Kitchen")
    bedroom_area = empty_context.area_registry.async_create("Bedroom")

    empty_context.entity_registry.async_get_or_create(
        "light", "test", "kitchen_lights", suggested_object_id="kitchen_lights"
    )
    empty_context.entity_registry.async_update_entity(
        "light.kitchen_lights", area_id=kitchen_area.id
    )

    empty_context.entity_registry.async_get_or_create(
        "light", "test", "bedroom_lights", suggested_object_id="bedroom_lights"
    )
    empty_context.entity_registry.async_update_entity(
        "light.bedroom_lights", area_id=bedroom_area.id
    )

    empty_context.hass.states.async_set(
        "light.kitchen_lights", "on", {"friendly_name": "Kitchen Ceiling Light"}
    )
    empty_context.hass.states.async_set(
        "light.bedroom_lights", "off", {"friendly_name": "Bedroom Ceiling Light"}
    )
    empty_context.states = empty_context.hass.states.async_all()

    # Area ranking
    areas = rank_areas(empty_context, "Turn off kitchen light")
    assert kitchen_area.id in areas
    assert "none" in areas

    # Entity ranking with area boosting: kitchen light should rank first
    entities = rank_entities(
        empty_context, "Turn off kitchen light", active_areas={kitchen_area.id}
    )
    assert "light.kitchen_lights" in entities
    keys = list(entities.keys())
    assert keys[0] == "light.kitchen_lights"


def test_rank_entities_allowed_and_boosted_domains(
    empty_context: StrategyContext,
) -> None:
    """Test rank_entities with allowed_domains strict filter and boosted_domains soft boost."""
    empty_context.hass.states.async_set(
        "light.kitchen_light", "on", {"friendly_name": "Kitchen Light"}
    )
    empty_context.hass.states.async_set(
        "media_player.kitchen_speaker", "off", {"friendly_name": "Kitchen Speaker"}
    )
    empty_context.states = empty_context.hass.states.async_all()

    # 1. Strict filtering with allowed_domains
    media_only = rank_entities(
        empty_context,
        "pause kitchen",
        active_areas=set(),
        allowed_domains={"media_player"},
    )
    assert "media_player.kitchen_speaker" in media_only
    assert "light.kitchen_light" not in media_only

    light_only = rank_entities(
        empty_context,
        "turn on kitchen",
        active_areas=set(),
        allowed_domains={"light"},
    )
    assert "light.kitchen_light" in light_only
    assert "media_player.kitchen_speaker" not in light_only

    # 2. Soft boosting with boosted_domains: both present, but boosted domain ranks first
    boosted = rank_entities(
        empty_context,
        "kitchen device",
        active_areas=set(),
        boosted_domains={"media_player"},
    )
    assert "media_player.kitchen_speaker" in boosted
    assert "light.kitchen_light" in boosted
    keys = list(boosted.keys())
    assert keys[0] == "media_player.kitchen_speaker"
