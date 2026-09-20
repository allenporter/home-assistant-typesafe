"""Speculative Fan-Out decision strategy with dynamic discovery and candidate retrieval."""

from __future__ import annotations

import logging
import re
from typing import Any, Literal

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
    discover_intents,
    get_allowed_domains_for_intents,
    rank_areas,
    rank_entities,
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
        domain_filter_mode: Literal["none", "strict", "boost"] = "none",
    ) -> None:
        """Initialize SpeculativeFanOutStrategy."""
        self._confidence_threshold = confidence_threshold
        self._compound_threshold = compound_threshold
        self._domain_filter_mode = domain_filter_mode

    @property
    def confidence_threshold(self) -> float:
        """Return the confidence threshold."""
        return self._confidence_threshold

    @property
    def compound_threshold(self) -> float:
        """Return the compound threshold."""
        return self._compound_threshold

    @property
    def domain_filter_mode(self) -> Literal["none", "strict", "boost"]:
        """Return the domain filter mode."""
        return self._domain_filter_mode

    def _discover_intents(
        self, context: StrategyContext, utterance: str
    ) -> dict[str, str]:
        """Discover and rank candidate intent schemas."""
        return discover_intents(context, utterance)

    def _rank_areas(self, context: StrategyContext, utterance: str) -> dict[str, str]:
        """Discover and rank candidate areas."""
        return rank_areas(context, utterance)

    def _rank_entities(
        self,
        context: StrategyContext,
        utterance: str,
        top_area_ids: set[str] | None = None,
        allowed_domains: set[str] | None = None,
        boosted_domains: set[str] | None = None,
    ) -> dict[str, str]:
        """Discover and rank candidate exposed entities."""
        return rank_entities(
            context,
            utterance,
            active_areas=top_area_ids or set(),
            allowed_domains=allowed_domains,
            boosted_domains=boosted_domains,
        )

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

        allowed_domains: set[str] | None = None
        boosted_domains: set[str] | None = None

        if self._domain_filter_mode in ("strict", "boost"):
            handlers_map: dict[str, intent.IntentHandler] = {}
            if context.hass:
                handlers_map = {
                    getattr(h, "intent_type", ""): h
                    for h in intent.async_get(context.hass)
                    if hasattr(h, "intent_type")
                }
            candidate_intents = [k for k in intent_criteria.keys() if k != "unmatched"]
            derived_domains = get_allowed_domains_for_intents(
                candidate_intents, handlers_map
            )
            if self._domain_filter_mode == "strict":
                allowed_domains = derived_domains
            else:
                boosted_domains = derived_domains

        entity_criteria = self._rank_entities(
            context,
            utterance,
            top_area_ids=set(area_criteria.keys()),
            allowed_domains=allowed_domains,
            boosted_domains=boosted_domains,
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


class StandardFanOutStrategy(SpeculativeFanOutStrategy):
    """Standard fan-out strategy without domain filtering."""

    def __init__(
        self,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        compound_threshold: float = DEFAULT_COMPOUND_THRESHOLD,
    ) -> None:
        """Initialize StandardFanOutStrategy."""
        super().__init__(
            confidence_threshold=confidence_threshold,
            compound_threshold=compound_threshold,
            domain_filter_mode="none",
        )


class IntentPrunedFanOutStrategy(SpeculativeFanOutStrategy):
    """Speculative fan-out strategy that strictly prunes entity candidates by candidate intent domains."""

    def __init__(
        self,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        compound_threshold: float = DEFAULT_COMPOUND_THRESHOLD,
    ) -> None:
        """Initialize IntentPrunedFanOutStrategy."""
        super().__init__(
            confidence_threshold=confidence_threshold,
            compound_threshold=compound_threshold,
            domain_filter_mode="strict",
        )


class DomainBoostedFanOutStrategy(SpeculativeFanOutStrategy):
    """Speculative fan-out strategy that soft-boosts entity candidates matching candidate intent domains."""

    def __init__(
        self,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        compound_threshold: float = DEFAULT_COMPOUND_THRESHOLD,
    ) -> None:
        """Initialize DomainBoostedFanOutStrategy."""
        super().__init__(
            confidence_threshold=confidence_threshold,
            compound_threshold=compound_threshold,
            domain_filter_mode="boost",
        )
