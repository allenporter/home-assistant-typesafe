"""Decision strategy and typed primitives for TypeSafe System One."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import re
from typing import Any

from homeassistant.core import HomeAssistant, State
from homeassistant.helpers import area_registry as ar, entity_registry as er, intent

from .client import TypeSafeClient
from .const import DEFAULT_CONFIDENCE_THRESHOLD

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

DEFAULT_INTENT_DESCRIPTIONS: dict[str, str] = {
    "HassTurnOn": "Turn on a device or entity",
    "HassTurnOff": "Turn off a device or entity",
    "HassLightSet": "Adjust brightness or color of a light",
    "HassClimateSetTemperature": "Set target temperature for thermostat or climate device",
}


@dataclass(slots=True)
class ChoiceQuestion:
    """Choice question primitive."""

    instructions: str | dict[str, Any] | list[Any]
    criteria: dict[str, str | None]

    def to_dict(self) -> dict[str, Any]:
        """Convert to API payload dict."""
        return {
            "type": "choice",
            "instructions": self.instructions,
            "criteria": self.criteria,
        }


@dataclass(slots=True)
class NoulQuestion:
    """Noul question primitive."""

    instructions: str | dict[str, Any] | list[Any]
    criteria: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to API payload dict."""
        payload: dict[str, Any] = {
            "type": "noul",
            "instructions": self.instructions,
        }
        if self.criteria is not None:
            payload["criteria"] = self.criteria
        return payload


@dataclass(slots=True)
class ChoiceAnswer:
    """Choice answer primitive."""

    choice: str
    confidence: float
    probabilities: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class NoulAnswer:
    """Noul answer primitive."""

    noul: float


@dataclass(slots=True)
class StrategyContext:
    """Context passed to DecisionStrategy."""

    hass: HomeAssistant
    area_registry: ar.AreaRegistry
    entity_registry: er.EntityRegistry
    states: list[State]
    home_name: str
    language: str | None = None
    device_id: str | None = None


@dataclass(slots=True)
class DecisionResult:
    """Result of decision evaluation."""

    intent_name: str | None
    entity_id: str | None = None
    area_name: str | None = None
    domain: str | None = None
    slots: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    should_escalate: bool = False
    escalation_reason: str | None = None
    is_compound: bool = False


def _tokenize(text: str) -> set[str]:
    """Tokenize a string into lowercase alphanumeric words."""
    return set(re.findall(r"\b\w+\b", text.lower()))


def _token_match(q_token: str, cand_token: str) -> bool:
    """Check if query token matches candidate token via equality or prefix/stem matching."""
    if q_token == cand_token:
        return True
    if len(q_token) >= 3 and len(cand_token) >= 3:
        if q_token.startswith(cand_token) or cand_token.startswith(q_token):
            return True
    return False


def _lexical_score(
    query_tokens: set[str], candidate: str, full_query: str = ""
) -> float:
    """Compute lexical matching score between query tokens and candidate name or description."""
    if not candidate:
        return 0.0
    cand_tokens = _tokenize(candidate)
    if not cand_tokens:
        return 0.0

    matches = 0
    for q in query_tokens:
        for c in cand_tokens:
            if _token_match(q, c):
                matches += 1
                break

    score = float(matches)
    if full_query:
        cand_lower = candidate.lower()
        query_lower = full_query.lower()
        if cand_lower in query_lower or query_lower in cand_lower:
            score += 2.0
    return score


def can_fulfill_intent(handler: intent.IntentHandler) -> bool:
    """Return True if the handler's required slots can be fulfilled."""
    req_slots = getattr(handler, "required_slots", None)
    if isinstance(req_slots, dict):
        required = set(req_slots.keys())
        return required.issubset(SUPPORTED_STRATEGY_SLOTS)
    return True


class DecisionStrategy:
    """TypeSafe System One decision strategy using speculative fan-out."""

    def __init__(
        self,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        compound_threshold: float = 0.5,
    ) -> None:
        """Initialize the decision strategy."""
        self._confidence_threshold = confidence_threshold
        self._compound_threshold = compound_threshold

    @property
    def confidence_threshold(self) -> float:
        """Return the confidence threshold."""
        return self._confidence_threshold

    @property
    def compound_threshold(self) -> float:
        """Return the compound threshold."""
        return self._compound_threshold

    def _discover_intents(
        self, context: StrategyContext, utterance: str
    ) -> dict[str, str]:
        """Discover and rank candidate intent schemas."""
        query_tokens = _tokenize(utterance)
        registered_handlers: list[intent.IntentHandler] = []
        if context.hass:
            try:
                registered_handlers = list(intent.async_get(context.hass))
            except Exception:
                registered_handlers = []

        scored_intents: list[tuple[float, str, str]] = []

        for handler in registered_handlers:
            intent_type = getattr(handler, "intent_type", None)
            if not intent_type or not can_fulfill_intent(handler):
                continue
            desc = getattr(
                handler, "description", ""
            ) or DEFAULT_INTENT_DESCRIPTIONS.get(intent_type, intent_type)
            name_readable = intent_type.replace("Hass", " ")
            score = _lexical_score(query_tokens, desc, utterance) + _lexical_score(
                query_tokens, name_readable, utterance
            )
            scored_intents.append((score, intent_type, desc))

        # Fallback to defaults if no registered handlers exist
        if not scored_intents:
            for itype, desc in DEFAULT_INTENT_DESCRIPTIONS.items():
                name_readable = itype.replace("Hass", " ")
                score = _lexical_score(query_tokens, desc, utterance) + _lexical_score(
                    query_tokens, name_readable, utterance
                )
                scored_intents.append((score, itype, desc))

        # Sort descending by score
        scored_intents.sort(key=lambda x: x[0], reverse=True)

        positive_intents = [item for item in scored_intents if item[0] > 0.0]
        intents_to_consider = positive_intents if positive_intents else scored_intents

        # Select top candidates up to 5
        criteria: dict[str, str] = {}
        for _, itype, desc in intents_to_consider[:5]:
            criteria[itype] = desc

        if not positive_intents:
            for itype in ("HassTurnOn", "HassTurnOff"):
                if itype not in criteria and len(criteria) < 5:
                    criteria[itype] = DEFAULT_INTENT_DESCRIPTIONS.get(
                        itype, f"Handle {itype.replace('Hass', '')}"
                    )

        criteria["unmatched"] = "Not a home control request or unsupported intent"
        return criteria

    def _rank_entities(
        self,
        context: StrategyContext,
        utterance: str,
        top_area_ids: set[str] | None = None,
    ) -> dict[str, str]:
        """Discover and rank candidate exposed entities."""
        query_tokens = _tokenize(utterance)
        scored_entities: list[tuple[float, str, str]] = []

        states = context.states if context.states is not None else []
        for state in states:
            domain = getattr(state, "domain", None)
            if not domain or domain not in CONTROLLABLE_DOMAINS:
                continue

            entity_id = state.entity_id
            friendly_name = (
                state.attributes.get("friendly_name")
                if hasattr(state, "attributes") and isinstance(state.attributes, dict)
                else None
            ) or entity_id

            # Find area name and ID if mapped
            area_name = ""
            area_id = ""
            if context.entity_registry and hasattr(
                context.entity_registry, "async_get"
            ):
                entry = context.entity_registry.async_get(entity_id)
                if entry and entry.area_id:
                    area_id = entry.area_id
                    if context.area_registry and hasattr(
                        context.area_registry, "async_get_area"
                    ):
                        area = context.area_registry.async_get_area(entry.area_id)
                        if area and getattr(area, "name", None):
                            area_name = area.name

            name_score = _lexical_score(query_tokens, friendly_name, utterance)
            id_score = _lexical_score(
                query_tokens, entity_id.replace("_", " "), utterance
            )
            score = max(name_score, id_score)

            # Boost if domain is in query tokens
            if domain in query_tokens:
                score += 0.5

            # Boost if area matches query tokens or top area
            if area_name and _lexical_score(query_tokens, area_name, utterance) > 0:
                score += 1.0
            elif top_area_ids and area_id and area_id in top_area_ids:
                score += 1.0

            description = f"{friendly_name} ({domain})"
            if area_name:
                description += f" in {area_name}"

            scored_entities.append((score, entity_id, description))

        # Sort by score descending
        scored_entities.sort(key=lambda x: x[0], reverse=True)

        criteria: dict[str, str] = {}
        for _, eid, desc in scored_entities[:20]:
            criteria[eid] = desc

        if criteria:
            criteria["none"] = "None of the listed devices"

        return criteria

    def _rank_areas(self, context: StrategyContext, utterance: str) -> dict[str, str]:
        """Discover and rank candidate areas."""
        query_tokens = _tokenize(utterance)
        scored_areas: list[tuple[float, str, str]] = []

        areas = {}
        if context.area_registry and hasattr(context.area_registry, "areas"):
            areas = context.area_registry.areas or {}

        for area in areas.values():
            if not area or not getattr(area, "name", None):
                continue
            score = _lexical_score(query_tokens, area.name, utterance)
            id_score = _lexical_score(
                query_tokens, area.id.replace("_", " "), utterance
            )
            scored_areas.append((max(score, id_score), area.id, area.name))

        scored_areas.sort(key=lambda x: x[0], reverse=True)

        criteria: dict[str, str] = {}
        for _, aid, name in scored_areas[:20]:
            criteria[aid] = name

        if criteria:
            criteria["none"] = "None of the listed areas"

        return criteria

    @staticmethod
    def _safe_choice(answers: dict[str, Any], key: str) -> str | None:
        """Safely extract string choice from answer primitive dict."""
        primitive = answers.get(key)
        if isinstance(primitive, dict):
            choice = primitive.get("choice")
            if isinstance(choice, str):
                return choice
        return None

    @staticmethod
    def _safe_float(val: Any, default: float = 0.0) -> float:
        """Safely convert value to float, defaulting on None or invalid input."""
        if val is None:
            return default
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str):
            try:
                return float(val)
            except ValueError:
                return default
        return default

    def _parse_evaluation_response(
        self,
        response: Any,
        utterance: str,
        context: StrategyContext,
    ) -> DecisionResult:
        """Safely parse and validate TypeSafe API evaluation response."""
        if not isinstance(response, dict):
            _LOGGER.warning(
                "TypeSafe evaluation returned non-dict response (%s): %r",
                type(response).__name__,
                response,
            )
            return DecisionResult(
                intent_name=None,
                confidence=0.0,
                should_escalate=True,
                escalation_reason="Malformed TypeSafe response: expected JSON object",
            )

        raw_answers = response.get("answers")
        if not isinstance(raw_answers, dict):
            _LOGGER.warning(
                "TypeSafe response missing or non-dict 'answers' object: %r", response
            )
            return DecisionResult(
                intent_name=None,
                confidence=0.0,
                should_escalate=True,
                escalation_reason="Malformed TypeSafe response: missing answers dictionary",
            )

        answers: dict[str, Any] = raw_answers

        # 1. Check compound command condition safely
        compound_ans = answers.get("is_compound")
        compound_noul = 0.0
        if isinstance(compound_ans, dict):
            compound_noul = self._safe_float(compound_ans.get("noul"), 0.0)

        if compound_noul > self._compound_threshold:
            return DecisionResult(
                intent_name=None,
                confidence=0.0,
                should_escalate=True,
                is_compound=True,
                escalation_reason="Compound command detected",
            )

        # 2. Check intent choice and confidence safely
        intent_ans = answers.get("intent")
        if not isinstance(intent_ans, dict):
            _LOGGER.warning(
                "TypeSafe response missing or non-dict 'intent' answer: %r", intent_ans
            )
            return DecisionResult(
                intent_name=None,
                confidence=0.0,
                should_escalate=True,
                escalation_reason="Missing or invalid intent answer",
            )

        raw_choice = intent_ans.get("choice")
        intent_choice = str(raw_choice) if isinstance(raw_choice, str) else None
        intent_conf = self._safe_float(intent_ans.get("confidence"), 0.0)

        raw_probs = intent_ans.get("probabilities")
        probabilities: dict[str, float] = {}
        if isinstance(raw_probs, dict):
            for k, v in raw_probs.items():
                probabilities[str(k)] = self._safe_float(v, 0.0)

        top_prob = intent_conf
        if intent_choice and intent_choice in probabilities:
            top_prob = probabilities[intent_choice]

        if (
            not intent_choice
            or intent_choice in ("unmatched", "none", "other")
            or top_prob < self._confidence_threshold
        ):
            return DecisionResult(
                intent_name=None,
                confidence=top_prob,
                should_escalate=True,
                escalation_reason="Unhandled intent or low confidence",
            )

        # 3. Resolve targets and slots safely
        slots: dict[str, Any] = {}
        target_type_choice = self._safe_choice(answers, "target_type")
        target_area_choice = self._safe_choice(answers, "target_area")
        target_entity_choice = self._safe_choice(answers, "target_entity")

        resolved_entity: str | None = None
        resolved_area: str | None = None
        resolved_domain: str | None = None

        if target_type_choice == "area" or (
            target_type_choice is None
            and target_area_choice not in (None, "none")
            and target_entity_choice in (None, "none")
        ):
            if target_area_choice and target_area_choice != "none":
                area_obj = None
                if context.area_registry and hasattr(
                    context.area_registry, "async_get_area"
                ):
                    area_obj = context.area_registry.async_get_area(target_area_choice)
                resolved_area = (
                    area_obj.name if area_obj and area_obj.name else None
                ) or target_area_choice
                slots["area"] = resolved_area

                # Infer domain from utterance or intent
                for dom in (
                    "light",
                    "switch",
                    "cover",
                    "climate",
                    "media_player",
                    "fan",
                ):
                    if dom in utterance.lower():
                        resolved_domain = dom
                        slots["domain"] = dom
                        break
        else:
            if target_entity_choice and target_entity_choice != "none":
                resolved_entity = target_entity_choice
                slots["entity_id"] = target_entity_choice
                if "." in target_entity_choice:
                    resolved_domain = target_entity_choice.split(".", 1)[0]

        # 4. Extract continuous numeric parameters via regex safely
        brightness_match = re.search(r"(\d+)\s*%", utterance)
        if brightness_match:
            try:
                slots["brightness"] = int(brightness_match.group(1))
            except ValueError:
                pass

        temp_match = re.search(
            r"(\d+(?:\.\d+)?)\s*(?:degrees|deg|°)", utterance, re.IGNORECASE
        )
        if temp_match:
            try:
                slots["temperature"] = float(temp_match.group(1))
            except ValueError:
                pass

        return DecisionResult(
            intent_name=intent_choice,
            entity_id=resolved_entity,
            area_name=resolved_area,
            domain=resolved_domain,
            slots=slots,
            confidence=top_prob,
            should_escalate=False,
        )

    async def async_decide(
        self, client: TypeSafeClient, utterance: str, context: StrategyContext
    ) -> DecisionResult:
        """Evaluate utterance using speculative fan-out typed questions."""
        intent_criteria = self._discover_intents(context, utterance)
        area_criteria = self._rank_areas(context, utterance)
        entity_criteria = self._rank_entities(
            context, utterance, set(area_criteria.keys())
        )

        questions: dict[str, Any] = {
            "intent": ChoiceQuestion(
                instructions="Determine the primary Home Assistant action",
                criteria=intent_criteria,  # type: ignore[arg-type]
            ).to_dict(),
            "is_compound": NoulQuestion(
                instructions="Does the request contain multiple distinct commands or conjunctions?"
            ).to_dict(),
        }

        if len(entity_criteria) > 1:
            questions["target_entity"] = ChoiceQuestion(
                instructions="Which entity is the user referring to?",
                criteria=entity_criteria,  # type: ignore[arg-type]
            ).to_dict()

        if len(area_criteria) > 1:
            questions["target_area"] = ChoiceQuestion(
                instructions="Which area or room is the user referring to?",
                criteria=area_criteria,  # type: ignore[arg-type]
            ).to_dict()

        if len(entity_criteria) > 1 and len(area_criteria) > 1:
            questions["target_type"] = ChoiceQuestion(
                instructions="Is the user targeting an individual device or an entire area?",
                criteria={
                    "entity": "A specific individual device or appliance",
                    "area": "An entire room or area",
                },
            ).to_dict()

        state = {
            "utterance": utterance,
            "home": context.home_name,
        }

        try:
            response = await client.async_evaluate(state=state, questions=questions)
        except Exception as err:
            _LOGGER.warning("TypeSafe evaluation failed: %s", err)
            return DecisionResult(
                intent_name=None,
                confidence=0.0,
                should_escalate=True,
                escalation_reason=f"TypeSafe evaluation error: {err}",
            )

        try:
            return self._parse_evaluation_response(response, utterance, context)
        except Exception as err:
            _LOGGER.warning(
                "TypeSafe response parsing failed unexpectedly: %s (payload: %r)",
                err,
                response,
            )
            return DecisionResult(
                intent_name=None,
                confidence=0.0,
                should_escalate=True,
                escalation_reason=f"TypeSafe response parsing error: {err}",
            )
