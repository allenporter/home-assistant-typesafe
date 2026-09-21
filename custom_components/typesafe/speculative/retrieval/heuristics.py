"""Candidate retrieval heuristics, token matching, and slot inspection for Home Assistant."""

from __future__ import annotations

from collections.abc import Collection
import logging
from typing import TYPE_CHECKING
from homeassistant.helpers import intent
import voluptuous as vol

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

from ..request.processor import tokenize

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

ON_OFF_SERVICE_DOMAINS: frozenset[str] = frozenset(
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


class IntentDomainStrategy:
    """Strategy for deriving controllable entity domains from intents and services."""

    def __init__(self, hass: HomeAssistant | None = None) -> None:
        """Initialize IntentDomainStrategy."""
        self._hass = hass

    def get_on_off_domains(self) -> set[str]:
        """Extract all domains that expose turn_on or turn_off services."""
        if (
            self._hass
            and hasattr(self._hass, "services")
            and hasattr(self._hass.services, "async_services")
        ):
            services = self._hass.services.async_services()
            domains = {
                domain
                for domain, domain_services in services.items()
                if "turn_on" in domain_services or "turn_off" in domain_services
            }
            if domains:
                return domains
        return set(ON_OFF_SERVICE_DOMAINS)

    def can_control_objects(self, handler: intent.IntentHandler) -> bool:
        """Check whether an intent handler can control devices or objects."""
        itype = getattr(handler, "intent_type", None)
        if not itype or itype in INFORMATIONAL_INTENTS:
            return False
        if not can_fulfill_intent(handler):
            return False

        platforms = getattr(handler, "platforms", None)
        if platforms:
            return True

        if itype in ("HassTurnOn", "HassTurnOff", "HassToggle"):
            return True

        supported, _ = get_handler_slot_info(handler)
        return bool(supported & {"name", "area", "entity_id", "domain"})

    def get_controllable_domains_from_intents(
        self,
        handlers_by_type: dict[str, intent.IntentHandler] | None = None,
    ) -> set[str]:
        """Get list of all intents, exclude informational/uncontrollable, and combine with service on/off domains."""
        handlers = handlers_by_type or {}
        if not handlers and self._hass:
            try:
                handlers = {
                    getattr(h, "intent_type", ""): h
                    for h in intent.async_get(self._hass)
                    if hasattr(h, "intent_type")
                }
            except (KeyError, AttributeError):
                handlers = {}

        domains: set[str] = set()
        for itype, handler in handlers.items():
            if itype in INFORMATIONAL_INTENTS or not self.can_control_objects(handler):
                continue
            platforms = getattr(handler, "platforms", None)
            if platforms:
                domains.update(platforms)

        domains.update(self.get_on_off_domains())
        return domains

    def get_domains_for_intents(
        self,
        candidate_intents: Collection[str],
        handlers_by_type: dict[str, intent.IntentHandler] | None = None,
    ) -> set[str]:
        """Derive allowed domains from candidate intents and their platform handlers."""
        handlers = handlers_by_type or {}
        if not handlers and self._hass:
            try:
                handlers = {
                    getattr(h, "intent_type", ""): h
                    for h in intent.async_get(self._hass)
                    if hasattr(h, "intent_type")
                }
            except (KeyError, AttributeError):
                handlers = {}

        domains: set[str] = set()
        for itype in candidate_intents:
            if itype in INFORMATIONAL_INTENTS:
                continue
            handler = handlers.get(itype)
            platforms = getattr(handler, "platforms", None) if handler else None
            if platforms:
                domains.update(platforms)
            elif itype in ("HassTurnOn", "HassTurnOff", "HassToggle"):
                domains.update(self.get_on_off_domains())

        return (
            domains if domains else self.get_controllable_domains_from_intents(handlers)
        )


def get_allowed_domains_for_intents(
    candidate_intent_types: Collection[str],
    handlers_by_type: dict[str, intent.IntentHandler] | None = None,
    hass: HomeAssistant | None = None,
) -> set[str]:
    """Derive allowed entity domains from candidate intents and their platform handlers."""
    strategy = IntentDomainStrategy(hass=hass)
    return strategy.get_domains_for_intents(candidate_intent_types, handlers_by_type)
