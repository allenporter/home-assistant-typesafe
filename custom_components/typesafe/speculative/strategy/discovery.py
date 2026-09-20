"""Candidate discovery, token matching, and slot inspection heuristics."""

from __future__ import annotations

import logging
import re

from homeassistant.helpers import intent
import voluptuous as vol

from .base import StrategyContext

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
        "scene",
        "script",
        "automation",
        "humidifier",
        "water_heater",
    }
)

SUPPORTED_STRATEGY_SLOTS: frozenset[str] = frozenset(
    {"name", "area", "domain", "floor", "device_class", "brightness", "temperature"}
)

CANONICAL_INTENT_DESCRIPTIONS: dict[str, str] = {
    "HassTurnOn": "Turn on or activate a device, light, or appliance",
    "HassTurnOff": "Turn off or deactivate a device, light, or appliance",
    "HassToggle": "Toggle a device on or off",
    "HassLightSet": "Set brightness, dim, or change color of lights",
    "HassClimateSetTemperature": "Set target temperature for thermostat or climate device",
}


def tokenize(text: str) -> set[str]:
    """Tokenize a string into lowercase alphanumeric words."""
    return set(re.findall(r"\b\w+\b", text.lower()))


def token_match(q_token: str, cand_token: str) -> bool:
    """Check if query token matches candidate token via equality or prefix/stem matching."""
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
        for c in cand_tokens:
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
                for sub in getattr(key, "validators", []):
                    if isinstance(sub, str):
                        names.append(sub)

            for name in names:
                supported.add(name)
                if is_req:
                    required.add(name)

    return supported, required


def can_fulfill_intent(handler: intent.IntentHandler) -> bool:
    """Determine whether the strategy can fulfill all required slots of an intent handler."""
    _supported, required = get_handler_slot_info(handler)
    unsupported_required = required - SUPPORTED_STRATEGY_SLOTS
    return len(unsupported_required) == 0


def discover_intents(
    context: StrategyContext,
    utterance: str,
    max_options: int = 15,
) -> dict[str, str]:
    """Discover candidate intents matching the user utterance."""
    query_tokens = tokenize(utterance)
    full_query = utterance.lower()

    scored_candidates: list[tuple[float, str, str]] = []
    registered_handlers = intent.async_get(context.hass) if context.hass else []

    if registered_handlers:
        for handler in registered_handlers:
            intent_type = handler.intent_type
            if not can_fulfill_intent(handler):
                continue

            desc = CANONICAL_INTENT_DESCRIPTIONS.get(intent_type)
            if not desc:
                raw_desc = (
                    getattr(handler, "description", None)
                    or handler.__doc__
                    or f"Handle {intent_type.replace('Hass', '').strip()}"
                )
                desc = raw_desc.split(". ")[0].strip()
                if not desc.endswith("."):
                    desc += "."

            desc_score = lexical_score(query_tokens, desc, full_query)
            name_score = lexical_score(
                query_tokens, intent_type.replace("Hass", " "), full_query
            )
            score = desc_score + name_score
            scored_candidates.append((score, intent_type, desc))
    else:
        for it, desc in CANONICAL_INTENT_DESCRIPTIONS.items():
            desc_score = lexical_score(query_tokens, desc, full_query)
            name_score = lexical_score(
                query_tokens, it.replace("Hass", " "), full_query
            )
            score = desc_score + name_score
            scored_candidates.append((score, it, desc))

    scored_candidates.sort(key=lambda x: x[0], reverse=True)

    positive_candidates = [c for c in scored_candidates if c[0] > 0]
    selected = (
        positive_candidates[:max_options]
        if positive_candidates
        else scored_candidates[:max_options]
    )

    criteria = {name: desc for _, name, desc in selected}
    criteria["unmatched"] = (
        "The request does not match any available Home Assistant action"
    )
    return criteria


def rank_areas(
    context: StrategyContext,
    utterance: str,
    max_options: int = 15,
) -> dict[str, str]:
    """Rank area candidates using lexical token matching.

    Args:
        context: Strategy context containing the Home Assistant area registry.
        utterance: Raw user input text.
        max_options: Maximum number of area options to include in the Choice question criteria.
            Restricting to top-N candidates keeps the prompt bounded and avoids context window
            bloat when homes have dozens of rooms.

    Returns:
        Dictionary mapping area name/ID to descriptive label for Choice criteria.
    """
    areas = context.area_registry.async_list_areas()
    query_tokens = tokenize(utterance)
    full_query = utterance.lower()

    scored_areas: list[tuple[float, str, str]] = []
    for area in areas:
        score = lexical_score(query_tokens, area.name, full_query)
        scored_areas.append((score, area.name, f"{area.name} area or room"))

    scored_areas.sort(key=lambda x: x[0], reverse=True)

    positive = [a for a in scored_areas if a[0] > 0]
    selected = positive[:max_options] if positive else scored_areas[:max_options]

    criteria = {name: desc for _, name, desc in selected}
    criteria["none"] = "No specific area referenced"
    return criteria


def rank_entities(
    context: StrategyContext,
    utterance: str,
    active_areas: set[str],
    max_options: int = 15,
) -> dict[str, str]:
    """Rank entity candidates based on lexical similarity and area boosting.

    Area boosting:
        When an utterance references a specific room or area (e.g. "turn off the kitchen lights"),
        devices physically assigned to that room in Home Assistant's entity registry receive an
        accumulative score bonus (+1.5 * area match score, plus +1.0 if the area matched the
        top-ranked active area). This prioritizes devices in the intended location over identically
        or similarly named devices in other rooms (e.g. "kitchen ceiling light" vs "bedroom ceiling light").

    Args:
        context: Strategy context containing states, entity registry, and area registry.
        utterance: Raw user input text.
        active_areas: Set of top candidate area names identified from the utterance.
        max_options: Maximum number of candidate entities to include in the Choice question criteria.
            Limits the candidate pool to the most relevant devices for model selection.

    Returns:
        Dictionary mapping entity_id to descriptive label (name, domain, area) for Choice criteria.
    """
    query_tokens = tokenize(utterance)
    full_query = utterance.lower()

    scored_entities: list[tuple[float, str, str]] = []

    for state in context.states:
        domain = state.domain
        if domain not in CONTROLLABLE_DOMAINS:
            continue

        friendly_name = state.attributes.get("friendly_name") or state.entity_id
        entry = context.entity_registry.async_get(state.entity_id)

        area_name: str | None = None
        if entry and entry.area_id:
            area_entry = context.area_registry.async_get_area(entry.area_id)
            if area_entry:
                area_name = area_entry.name

        base_score = lexical_score(
            query_tokens, f"{friendly_name} {domain}", full_query
        )
        area_score = (
            lexical_score(query_tokens, area_name, full_query) if area_name else 0.0
        )
        total_score = base_score + (area_score * 1.5)

        if area_name and area_name in active_areas and area_score > 0:
            total_score += 1.0

        desc = f"{friendly_name} ({domain})"
        if area_name:
            desc += f" in {area_name}"

        scored_entities.append((total_score, state.entity_id, desc))

    scored_entities.sort(key=lambda x: x[0], reverse=True)

    positive = [e for e in scored_entities if e[0] > 0]
    selected = positive[:max_options] if positive else scored_entities[:max_options]

    criteria = {eid: desc for _, eid, desc in selected}
    criteria["none"] = "No specific entity referenced"
    return criteria
