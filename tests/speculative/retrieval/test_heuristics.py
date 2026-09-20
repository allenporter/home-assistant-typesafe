"""Unit tests for retrieval heuristics and token matching."""

from __future__ import annotations

import voluptuous as vol
from homeassistant.helpers import intent

from custom_components.typesafe.speculative.retrieval.heuristics import (
    CONTROLLABLE_DOMAINS,
    can_fulfill_intent,
    get_allowed_domains_for_intents,
    get_handler_slot_info,
    lexical_score,
    token_match,
    tokenize,
)


class DummyValidHandler(intent.IntentHandler):
    """Intent handler with supported slots."""

    intent_type = "HassTurnOn"
    slot_schema = {
        vol.Required("name"): str,
        vol.Optional("area"): str,
    }


class DummyUnsupportedHandler(intent.IntentHandler):
    """Intent handler with unsupported required slots."""

    intent_type = "CustomUnsupportedIntent"
    slot_schema = {
        vol.Required("unsupported_custom_slot"): str,
    }


class DummyPlatformHandler(intent.IntentHandler):
    """Intent handler specifying target platforms."""

    intent_type = "HassMediaPause"
    platforms = {"media_player"}


class DummyLightHandler(intent.IntentHandler):
    """Intent handler specifying light platform."""

    intent_type = "HassLightSet"
    platforms = {"light"}


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
    assert not token_match("in", "inside")


def test_intent_handler_slot_info_and_can_fulfill() -> None:
    """Test slot introspection and can_fulfill_intent filtering."""
    handler_valid = DummyValidHandler()
    supported, required = get_handler_slot_info(handler_valid)
    assert "name" in required
    assert "area" in supported
    assert can_fulfill_intent(handler_valid)

    handler_unsupported = DummyUnsupportedHandler()
    supported, required = get_handler_slot_info(handler_unsupported)
    assert "unsupported_custom_slot" in required
    assert not can_fulfill_intent(handler_unsupported)


def test_get_allowed_domains_for_intents() -> None:
    """Test deriving allowed domains from candidate intents and handlers."""
    handlers = {
        "HassMediaPause": DummyPlatformHandler(),
        "HassLightSet": DummyLightHandler(),
    }

    assert get_allowed_domains_for_intents(["HassMediaPause"], handlers) == {
        "media_player"
    }
    assert get_allowed_domains_for_intents(["HassLightSet"], handlers) == {"light"}

    turn_on_domains = get_allowed_domains_for_intents(["HassTurnOn"], handlers)
    assert turn_on_domains == set(CONTROLLABLE_DOMAINS)
    assert "light" in turn_on_domains
    assert "cover" in turn_on_domains
    assert "valve" in turn_on_domains
    assert "sensor" not in turn_on_domains

    info_domains = get_allowed_domains_for_intents(["HassGetState"], handlers)
    assert info_domains == set(CONTROLLABLE_DOMAINS)
