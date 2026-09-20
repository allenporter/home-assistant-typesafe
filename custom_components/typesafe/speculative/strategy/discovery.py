"""Smart home candidate discovery, token matching, and slot inspection heuristics.

This module acts as the semantic bridge projecting live Home Assistant smart home state
(entities, areas, devices, and registered intent handlers) into bounded, structured
choice sets suitable for TypeSafe System One speculative decision strategies.

Background & Motivation:
    In a typical smart home, Home Assistant may manage hundreds of entity states across
    dozens of physical rooms alongside dozens of registered intent handlers. Directly
    sending an exhaustive list of all entities and actions to an LLM or decision engine
    on every forward pass is computationally prohibitive, inflates token cost and latency,
    and increases decision confusion.

    The discovery heuristics in this module solve this by performing fast, local, offline
    candidate retrieval and pruning tailored to the user's utterance before invoking the
    decision engine:
    1. Intent Discovery: Introspects registered Home Assistant `IntentHandler` instances,
       verifies that required slots can be satisfied by the strategy (`can_fulfill_intent`),
       and ranks matching candidate intents.
    2. Area Ranking: Matches tokens against registered Home Assistant area names to identify
       rooms explicitly or implicitly targeted by the speaker.
    3. Entity Ranking with Area Boosting: Filters down to controllable domains, matches entity
       friendly names and IDs against utterance tokens, and applies a multi-tier score boost
       for devices located within the top-ranked target areas.

The resulting candidate mappings directly populate the `criteria` attributes of
canonical `ChoiceQuestion` primitives evaluated by `SpeculativeFanOutStrategy`.
"""

from __future__ import annotations

from collections.abc import Collection
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
        "valve",
        "scene",
        "script",
        "automation",
        "humidifier",
        "water_heater",
    }
)
"""Home Assistant entity domains that support direct speculative voice control actions."""

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
        # Timer intents (handled by timer manager / conversation fallback)
        "HassStartTimer",
        "HassCancelTimer",
        "HassCancelAllTimers",
        "HassPauseTimer",
        "HassUnpauseTimer",
        "HassIncreaseTimer",
        "HassDecreaseTimer",
        "HassTimerStatus",
        # List / to-do intents (handled by list manager / conversation fallback)
        "HassListAddItem",
        "HassListCompleteItem",
        "HassListRemoveItem",
        "HassShoppingListAddItem",
        "HassShoppingListCompleteItem",
        "HassShoppingListLastItems",
    }
)

SUPPORTED_STRATEGY_SLOTS: frozenset[str] = frozenset(
    {"name", "area", "domain", "floor", "device_class", "brightness", "temperature"}
)
"""Slots that the speculative fan-out strategy is capable of extracting and populating."""


def get_allowed_domains_for_intents(
    candidate_intent_types: Collection[str],
    handlers_by_type: dict[str, intent.IntentHandler] | None = None,
) -> set[str]:
    """Derive allowed entity domains from candidate intents and their platform handlers.

    Specialized intents (e.g. HassMediaPause -> media_player, HassLightSet -> light)
    restrict allowed domains to their platform. Generic on/off/toggle intents fall back
    to CONTROLLABLE_DOMAINS.
    """
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
"""High-quality canonical descriptions used as Choice criteria labels for standard Home Assistant intents."""


def tokenize(text: str) -> set[str]:
    """Tokenize a string into a set of unique lowercase alphanumeric words.

    Args:
        text: Raw input string to tokenize.

    Returns:
        A set of lowercase alphanumeric token strings with punctuation stripped.
    """
    return set(re.findall(r"\b\w+\b", text.lower()))


def token_match(q_token: str, cand_token: str) -> bool:
    """Check if query token matches candidate token via equality or prefix/stem matching.

    Matches either if the tokens are strictly equal, or if both tokens are at least 3
    characters long and one is a prefix of the other (e.g. 'light' matching 'lights',
    or 'temp' matching 'temperature').

    Args:
        q_token: Lowercase token extracted from user query.
        cand_token: Lowercase token extracted from candidate entity, area, or intent label.

    Returns:
        True if the tokens match under exact or prefix rules; False otherwise.
    """
    if q_token == cand_token:
        return True
    if len(q_token) >= 3 and len(cand_token) >= 3:
        if q_token.startswith(cand_token) or cand_token.startswith(q_token):
            return True
    return False


STOPWORDS: frozenset[str] = frozenset(
    {"the", "a", "an", "of", "to", "for", "is", "at", "by", "with", "or"}
)
"""Grammatical function words ignored during lexical token matching."""


def lexical_score(query_tokens: set[str], candidate: str, full_query: str) -> float:
    """Compute lexical matching score between query tokens and candidate name or description.

    Calculates the number of token overlaps using `token_match()`. If the full candidate
    phrase appears as a substring in the user query (or vice versa), an additional +2.0
    phrase-matching bonus is awarded.

    Args:
        query_tokens: Pre-tokenized set of words from the user utterance.
        candidate: Candidate string to compare against (e.g. friendly name, area name).
        full_query: Normalized raw user input string.

    Returns:
        Float score representing lexical relevance (0.0 or higher).
    """
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
    """Extract (supported_slots, required_slots) from a Home Assistant IntentHandler.

    Inspects the handler's attributes (`required_slots`, `optional_slots`) and introspects
    any `voluptuous` schema attached to `slot_schema` to discover all slot names and markers.

    Args:
        handler: Registered Home Assistant IntentHandler instance.

    Returns:
        A tuple of (supported_slot_names, required_slot_names).
    """
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
    """Determine whether the strategy can fulfill all required slots of an intent handler.

    Checks if any required slots of the intent handler fall outside of `SUPPORTED_STRATEGY_SLOTS`.
    If an intent requires slots we cannot populate (e.g. timer duration or weather forecast days),
    it is filtered out of the candidate pool so the model is not asked to invoke an intent
    that will fail execution.

    Args:
        handler: Registered Home Assistant IntentHandler to inspect.

    Returns:
        True if all required slots can be satisfied by our strategy; False otherwise.
    """
    _supported, required = get_handler_slot_info(handler)
    unsupported_required = required - SUPPORTED_STRATEGY_SLOTS
    return len(unsupported_required) == 0


def discover_intents(
    context: StrategyContext,
    utterance: str,
    max_options: int = 5,
) -> dict[str, str]:
    """Discover and rank candidate Home Assistant intents matching the user utterance.

    Introspects registered intent handlers from `context.hass`. Filters handlers using
    `can_fulfill_intent()`, computes lexical relevance scores against intent names and
    descriptions, and selects the top candidates up to `max_options`. An explicit
    `'unmatched'` choice is always included as an escape hatch for out-of-domain requests.
    """
    query_tokens = tokenize(utterance)
    full_query = utterance.lower()

    scored_candidates: list[tuple[float, str, str]] = []
    registered_handlers = intent.async_get(context.hass) if context.hass else []

    if registered_handlers:
        for handler in registered_handlers:
            intent_type = getattr(handler, "intent_type", None)
            if (
                not intent_type
                or intent_type in INFORMATIONAL_INTENTS
                or not can_fulfill_intent(handler)
            ):
                continue

            raw_desc = (
                getattr(handler, "description", None)
                or getattr(handler, "__doc__", None)
                or CANONICAL_INTENT_DESCRIPTIONS.get(intent_type)
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
            if it in INFORMATIONAL_INTENTS:
                continue
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

    if not positive_candidates:
        for itype in ("HassTurnOn", "HassTurnOff"):
            if itype not in criteria and len(criteria) < max_options:
                criteria[itype] = CANONICAL_INTENT_DESCRIPTIONS.get(
                    itype, f"Handle {itype.replace('Hass', '')}"
                )

    criteria["unmatched"] = "Not a home control request or unsupported intent"
    return criteria


def rank_areas(
    context: StrategyContext,
    utterance: str,
    max_options: int = 10,
) -> dict[str, str]:
    """Rank area candidates using lexical token matching.

    Args:
        context: Strategy context containing the Home Assistant area registry.
        utterance: Raw user input text.
        max_options: Maximum number of area options to include in the Choice question criteria.
    """
    areas = context.area_registry.async_list_areas() if context.area_registry else []
    query_tokens = tokenize(utterance)
    full_query = utterance.lower()

    scored_areas: list[tuple[float, str, str]] = []
    for area in areas:
        if not area or not area.name:
            continue
        score = lexical_score(query_tokens, area.name, full_query)
        area_key = getattr(area, "id", None) or getattr(area, "name", "")
        id_score = lexical_score(query_tokens, area_key.replace("_", " "), full_query)
        scored_areas.append((max(score, id_score), area_key, f"{area.name} area"))

    scored_areas.sort(key=lambda x: x[0], reverse=True)

    positive = [a for a in scored_areas if a[0] > 0]
    selected = positive[:max_options] if positive else scored_areas[:max_options]

    criteria = {key: desc for _, key, desc in selected}
    if criteria:
        criteria["none"] = "None of the listed areas"
    return criteria


def rank_entities(
    context: StrategyContext,
    utterance: str,
    active_areas: set[str],
    allowed_domains: set[str] | None = None,
    boosted_domains: set[str] | None = None,
    max_options: int = 20,
) -> dict[str, str]:
    """Rank entity candidates based on lexical similarity, area boosting, and domain filtering.

    Args:
        context: Strategy context containing states, entity registry, and area registry.
        utterance: Raw user input text.
        active_areas: Set of top candidate area names/IDs identified from the utterance.
        allowed_domains: If specified, strictly restrict candidates to these domains.
        boosted_domains: If specified, award a bonus (+1.5) to entities in these domains.
        max_options: Maximum number of candidate entities to include.
    """
    query_tokens = tokenize(utterance)
    full_query = utterance.lower()

    target_domains = (
        allowed_domains if allowed_domains is not None else CONTROLLABLE_DOMAINS
    )
    scored_entities: list[tuple[float, str, str]] = []
    states = context.states if context.states is not None else []

    for state in states:
        domain = getattr(state, "domain", None)
        if not domain or domain not in target_domains:
            continue

        entity_id = state.entity_id
        friendly_name = (
            state.attributes.get("friendly_name")
            if hasattr(state, "attributes") and isinstance(state.attributes, dict)
            else getattr(state, "name", None)
        ) or entity_id

        area_name = ""
        area_id = ""
        if context.entity_registry and hasattr(context.entity_registry, "async_get"):
            entry = context.entity_registry.async_get(entity_id)
            if entry and entry.area_id:
                area_id = entry.area_id
                if context.area_registry and hasattr(
                    context.area_registry, "async_get_area"
                ):
                    area = context.area_registry.async_get_area(entry.area_id)
                    if area and area.name:
                        area_name = area.name

        name_score = lexical_score(query_tokens, friendly_name, full_query)
        id_score = lexical_score(query_tokens, entity_id.replace("_", " "), full_query)
        score = max(name_score, id_score)

        if domain in query_tokens:
            score += 0.5

        if area_name and lexical_score(query_tokens, area_name, full_query) > 0:
            score += 1.0
        elif active_areas and (
            (area_id and area_id in active_areas)
            or (area_name and area_name in active_areas)
        ):
            score += 1.0

        if boosted_domains and domain in boosted_domains:
            score += 1.5

        desc = f"{friendly_name} ({domain})"
        if area_name:
            desc += f" in {area_name}"

        scored_entities.append((score, entity_id, desc))

    scored_entities.sort(key=lambda x: x[0], reverse=True)

    criteria: dict[str, str] = {}
    for _, eid, desc in scored_entities[:max_options]:
        criteria[eid] = desc

    if criteria:
        criteria["none"] = "None of the listed devices"
    return criteria
