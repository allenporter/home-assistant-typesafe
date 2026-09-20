"""Speculative Fan-Out decision strategy with dynamic discovery and candidate retrieval."""

from __future__ import annotations

import logging
import re
from typing import Any

from homeassistant.helpers import intent

from ..engine import DecisionEngine, PredictionResult
from ..models import (
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    NoulQuestion,
    Question,
)
from .base import Decision, DecisionStrategy, StrategyContext
from .discovery import (
    CANONICAL_INTENT_DESCRIPTIONS,
    CONTROLLABLE_DOMAINS,
    can_fulfill_intent,
    lexical_score,
    tokenize,
)

_LOGGER = logging.getLogger(__name__)

DEFAULT_CONFIDENCE_THRESHOLD: float = 0.50
DEFAULT_COMPOUND_THRESHOLD: float = 0.50


class SpeculativeFanOutStrategy(DecisionStrategy):
    """Speculative fan-out strategy with dynamic intent and candidate discovery."""

    def __init__(
        self,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        compound_threshold: float = DEFAULT_COMPOUND_THRESHOLD,
    ) -> None:
        """Initialize SpeculativeFanOutStrategy."""
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
        query_tokens = tokenize(utterance)
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
            desc = handler.description or CANONICAL_INTENT_DESCRIPTIONS.get(
                intent_type, intent_type
            )
            name_readable = intent_type.replace("Hass", " ")
            score = lexical_score(query_tokens, desc, utterance) + lexical_score(
                query_tokens, name_readable, utterance
            )
            scored_intents.append((score, intent_type, desc))

        # Fallback to defaults if no registered handlers exist
        if not scored_intents:
            for itype, desc in CANONICAL_INTENT_DESCRIPTIONS.items():
                name_readable = itype.replace("Hass", " ")
                score = lexical_score(query_tokens, desc, utterance) + lexical_score(
                    query_tokens, name_readable, utterance
                )
                scored_intents.append((score, itype, desc))

        scored_intents.sort(key=lambda x: x[0], reverse=True)

        positive_intents = [item for item in scored_intents if item[0] > 0.0]
        intents_to_consider = positive_intents if positive_intents else scored_intents

        criteria: dict[str, str] = {}
        for _, itype, desc in intents_to_consider[:5]:
            criteria[itype] = desc

        if not positive_intents:
            for itype in ("HassTurnOn", "HassTurnOff"):
                if itype not in criteria and len(criteria) < 5:
                    criteria[itype] = CANONICAL_INTENT_DESCRIPTIONS.get(
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
        query_tokens = tokenize(utterance)
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
                        if area and area.name:
                            area_name = area.name

            name_score = lexical_score(query_tokens, friendly_name, utterance)
            id_score = lexical_score(
                query_tokens, entity_id.replace("_", " "), utterance
            )
            score = max(name_score, id_score)

            if domain in query_tokens:
                score += 0.5

            if area_name and lexical_score(query_tokens, area_name, utterance) > 0:
                score += 1.0
            elif top_area_ids and area_id and area_id in top_area_ids:
                score += 1.0

            description = f"{friendly_name} ({domain})"
            if area_name:
                description += f" in {area_name}"

            scored_entities.append((score, entity_id, description))

        scored_entities.sort(key=lambda x: x[0], reverse=True)

        criteria: dict[str, str] = {}
        for _, eid, desc in scored_entities[:20]:
            criteria[eid] = desc

        if criteria:
            criteria["none"] = "None of the listed devices"

        return criteria

    def _rank_areas(self, context: StrategyContext, utterance: str) -> dict[str, str]:
        """Discover and rank candidate areas."""
        query_tokens = tokenize(utterance)
        scored_areas: list[tuple[float, str, str]] = []

        areas = {}
        if context.area_registry and hasattr(context.area_registry, "areas"):
            areas = context.area_registry.areas or {}

        for area in areas.values():
            if not area or not area.name:
                continue
            score = lexical_score(query_tokens, area.name, utterance)
            id_score = lexical_score(query_tokens, area.id.replace("_", " "), utterance)
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
        """Safely extract string choice from answer primitive."""
        primitive = answers.get(key)
        if isinstance(primitive, ChoiceAnswer):
            return primitive.choice if primitive.choice else None
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
    ) -> Decision:
        """Safely parse and validate TypeSafe PredictionResult."""
        if not isinstance(response, PredictionResult):
            _LOGGER.warning(
                "TypeSafe evaluation returned unexpected response (%s): %r",
                type(response).__name__,
                response,
            )
            return Decision(
                intent_name=None,
                confidence=0.0,
                should_escalate=True,
                escalation_reason="Malformed TypeSafe response: expected PredictionResult",
            )

        answers: dict[str, Any] = response.answers

        # 1. Check compound command condition safely
        compound_ans = answers.get("is_compound")
        compound_noul = (
            compound_ans.noul if isinstance(compound_ans, NoulAnswer) else 0.0
        )

        if compound_noul > self._compound_threshold:
            return Decision(
                intent_name=None,
                confidence=0.0,
                should_escalate=True,
                is_compound=True,
                escalation_reason="Compound command detected",
                raw_answers=answers,
            )

        # 2. Check intent choice and confidence safely
        intent_ans = answers.get("intent")
        if not isinstance(intent_ans, ChoiceAnswer):
            _LOGGER.warning(
                "Response missing or invalid 'intent' answer: %r", intent_ans
            )
            return Decision(
                intent_name=None,
                confidence=0.0,
                should_escalate=True,
                escalation_reason="Missing or invalid intent answer",
                raw_answers=answers,
            )

        intent_choice = intent_ans.choice if intent_ans.choice else None
        intent_conf = intent_ans.confidence
        probabilities = {
            str(k): self._safe_float(v, 0.0)
            for k, v in intent_ans.probabilities.items()
        }

        top_prob = intent_conf
        if intent_choice and intent_choice in probabilities:
            top_prob = probabilities[intent_choice]

        if (
            not intent_choice
            or intent_choice.lower() in ("unmatched", "none", "other", "")
            or top_prob < self._confidence_threshold
        ):
            return Decision(
                intent_name=None,
                confidence=top_prob,
                should_escalate=True,
                escalation_reason="Unhandled intent or low confidence",
                raw_answers=answers,
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

        return Decision(
            intent_name=intent_choice,
            entity_id=resolved_entity,
            area_name=resolved_area,
            domain=resolved_domain,
            slots=slots,
            confidence=top_prob,
            should_escalate=False,
            raw_answers=answers,
        )

    def _build_questions(
        self,
        context: StrategyContext,
        utterance: str,
    ) -> dict[str, Question]:
        """Build canonical Question objects for speculative fan-out."""
        intent_criteria = self._discover_intents(context, utterance)
        area_criteria = self._rank_areas(context, utterance)
        entity_criteria = self._rank_entities(
            context, utterance, set(area_criteria.keys())
        )

        canonical: dict[str, Question] = {
            "intent": ChoiceQuestion(
                instructions="Determine the primary Home Assistant action",
                criteria=intent_criteria,  # type: ignore[arg-type]
            ),
            "is_compound": NoulQuestion(
                instructions="Does the request contain multiple distinct commands or conjunctions?"
            ),
        }

        if len(entity_criteria) > 1:
            canonical["target_entity"] = ChoiceQuestion(
                instructions="Which entity is the user referring to?",
                criteria=entity_criteria,  # type: ignore[arg-type]
            )

        if len(area_criteria) > 1:
            canonical["target_area"] = ChoiceQuestion(
                instructions="Which area or room is the user referring to?",
                criteria=area_criteria,  # type: ignore[arg-type]
            )

        if len(entity_criteria) > 1 and len(area_criteria) > 1:
            target_type_crit = {
                "entity": "A specific individual device or appliance",
                "area": "An entire room or area",
            }
            canonical["target_type"] = ChoiceQuestion(
                instructions="Is the user targeting an individual device or an entire area?",
                criteria=target_type_crit,
            )

        return canonical

    async def async_decide(
        self,
        engine: DecisionEngine,
        text: str,
        context: StrategyContext,
    ) -> Decision:
        """Evaluate utterance using speculative fan-out typed questions."""
        canonical_questions = self._build_questions(context, text)

        state = {
            "utterance": text,
            "home": context.home_name,
        }

        try:
            prediction = await engine.async_predict(
                state=state, questions=canonical_questions
            )
            return self._parse_evaluation_response(prediction, text, context)
        except Exception as err:
            _LOGGER.warning("Engine prediction failed: %s", err)
            return Decision(
                intent_name=None,
                confidence=0.0,
                should_escalate=True,
                escalation_reason=f"Engine prediction error: {err}",
            )
