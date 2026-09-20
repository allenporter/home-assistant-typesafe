"""Candidate retrieval heuristics, token matching, and slot inspection for Home Assistant."""

from __future__ import annotations

from collections.abc import Collection
import logging
import re

from homeassistant.helpers import intent
import voluptuous as vol

_LOGGER = logging.getLogger(__name__)

CONTROLLABLE_DOMAINS: frozenset[str] = frozenset(
    {
        "light",
        "switch",
        "climate",
        "cover",
        "media_player",
        "fan",
        "lock",
        "vacuum",
        "valve",
        "scene",
        "script",
        "automation",
        "humidifier",
        "water_heater",
    }
)

INFORMATIONAL_INTENTS: frozenset[str] = frozenset(
    {
        "HassGetState",
        "HassGetTemperature",
        "HassClimateGetTemperature",
        "HassGetWeather",
        "HassGetCurrentDate",
        "HassGetCurrentTime",
        "HassNevermind",
        "HassRespond",
        "HassStartTimer",
        "HassCancelTimer",
        "HassCancelAllTimers",
        "HassPauseTimer",
        "HassUnpauseTimer",
        "HassIncreaseTimer",
        "HassDecreaseTimer",
        "HassTimerStatus",
        "HassListAddItem",
        "HassListCompleteItem",
        "HassListRemoveItem",
        "HassShoppingListAddItem",
        "HassShoppingListCompleteItem",
        "HassShoppingListLastItems",
    }
)

SUPPORTED_DECISION_SLOTS: frozenset[str] = frozenset(
    {"name", "area", "domain", "floor", "device_class", "brightness", "temperature"}
)

CANONICAL_INTENT_DESCRIPTIONS: dict[str, str] = {
    "HassTurnOn": "Turn on or activate a device, light, or appliance",
    "HassTurnOff": "Turn off or deactivate a device, light, or appliance",
    "HassToggle": "Toggle a device on or off",
    "HassLightSet": "Set brightness, dim, or change color of lights",
    "HassClimateSetTemperature": "Set target temperature for thermostat or climate device",
    "HassMediaPause": "Pause media, music, or playback",
    "HassMediaUnpause": "Resume media, music, or playback",
    "HassOpenCover": "Open a cover, garage door, blinds, or shades",
    "HassCloseCover": "Close a cover, garage door, blinds, or shades",
    "HassStopMoving": "Stop movement of a cover, garage door, or shades",
}

STOPWORDS: frozenset[str] = frozenset(
    {"the", "a", "an", "of", "to", "for", "is", "at", "by", "with", "or"}
)


def tokenize(text: str) -> set[str]:
    """Tokenize a string into a set of unique lowercase alphanumeric words."""
    return set(re.findall(r"\b\w+\b", text.lower()))


def token_match(q_token: str, cand_token: str) -> bool:
    """Check if query token matches candidate token via equality or prefix matching."""
    if q_token == cand_token:
        return True
    if len(q_token) >= 3 and len(cand_token) >= 3:
        if q_token.startswith(cand_token) or cand_token.startswith(q_token):
            return True
    return False


def lexical_score(query_tokens: set[str], candidate: str, full_query: str) -> float:
    """Compute lexical matching score between query tokens and candidate name or description."""
    cand_tokens = tokenize(candidate)
    if not cand_tokens:
        return 0.0

    matches = 0
    for q in query_tokens:
        if q in STOPWORDS:
            continue
        for c in cand_tokens:
            if c in STOPWORDS:
                continue
            if token_match(q, c):
                matches += 1
                break

    score = float(matches)
    cand_lower = candidate.lower()
    query_lower = full_query.lower()
    if cand_lower in query_lower or query_lower in cand_lower:
        score += 2.0
    return score


def get_allowed_domains_for_intents(
    candidate_intent_types: Collection[str],
    handlers_by_type: dict[str, intent.IntentHandler] | None = None,
) -> set[str]:
    """Derive allowed entity domains from candidate intents and their platform handlers."""
    allowed_domains: set[str] = set()
    handlers = handlers_by_type or {}

    for itype in candidate_intent_types:
        if itype == "unmatched" or itype in INFORMATIONAL_INTENTS:
            continue
        handler = handlers.get(itype)
        platforms = getattr(handler, "platforms", None) if handler else None
        if platforms:
            allowed_domains.update(platforms)
        elif itype in ("HassTurnOn", "HassTurnOff", "HassToggle"):
            allowed_domains.update(CONTROLLABLE_DOMAINS)

    return allowed_domains if allowed_domains else set(CONTROLLABLE_DOMAINS)


def get_handler_slot_info(
    handler: intent.IntentHandler,
) -> tuple[set[str], set[str]]:
    """Extract (supported_slots, required_slots) from a Home Assistant IntentHandler."""
    supported: set[str] = set()
    required: set[str] = set()

    req_slots = getattr(handler, "required_slots", None)
    if isinstance(req_slots, dict):
        req_keys = set(req_slots.keys())
        required.update(req_keys)
        supported.update(req_keys)
    opt_slots = getattr(handler, "optional_slots", None)
    if isinstance(opt_slots, dict):
        supported.update(opt_slots.keys())

    schema = getattr(handler, "slot_schema", None)
    if isinstance(schema, dict):
        for key in schema:
            is_req = isinstance(key, vol.Required)
            names: list[str] = []
            if isinstance(key, str):
                names.append(key)
            elif isinstance(key, vol.Marker) and isinstance(key.schema, str):
                names.append(key.schema)
            elif isinstance(key, vol.Any):
                for sub in key.validators:
                    if isinstance(sub, str):
                        names.append(sub)

            for name in names:
                supported.add(name)
                if is_req:
                    required.add(name)

    return supported, required


def can_fulfill_intent(handler: intent.IntentHandler) -> bool:
    """Determine whether all required slots of an intent handler can be fulfilled."""
    _supported, required = get_handler_slot_info(handler)
    unsupported_required = required - SUPPORTED_DECISION_SLOTS
    return len(unsupported_required) == 0
