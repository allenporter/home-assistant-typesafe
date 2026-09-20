"""Decision resolver implementations for Stage 5."""

from __future__ import annotations

from abc import ABC, abstractmethod
import logging
import re
from typing import Any

from ..models import (
    ChoiceAnswer,
    NoulAnswer,
)
from ..request.models import ParsedRequest
from ..retrieval.models import RetrievedCandidates
from ..scoring.engine import PredictionResult
from .models import Decision

_LOGGER = logging.getLogger(__name__)

DEFAULT_CONFIDENCE_THRESHOLD: float = 0.50
DEFAULT_COMPOUND_THRESHOLD: float = 0.50


class DecisionResolver(ABC):
    """Abstract interface for Stage 5 decision resolution."""

    @abstractmethod
    def resolve(
        self,
        prediction: PredictionResult,
        request: ParsedRequest,
        candidates: RetrievedCandidates,
    ) -> Decision:
        """Resolve prediction results into an actionable Decision."""


class DefaultDecisionResolver(DecisionResolver):
    """Default decision resolver handling confidence gating, targets, and slot binding."""

    def __init__(
        self,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        compound_threshold: float = DEFAULT_COMPOUND_THRESHOLD,
    ) -> None:
        """Initialize DefaultDecisionResolver."""
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

    @staticmethod
    def _safe_choice(answers: dict[str, Any], key: str) -> str | None:
        """Safely extract string choice from answer primitive."""
        primitive = answers.get(key)
        if isinstance(primitive, ChoiceAnswer) and primitive.choice:
            return primitive.choice
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

    def resolve(
        self,
        prediction: PredictionResult,
        request: ParsedRequest,
        candidates: RetrievedCandidates,
    ) -> Decision:
        """Resolve prediction results into an actionable Decision."""
        answers: dict[str, Any] = prediction.answers

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
                resolved_area = next(
                    (
                        a.area_name
                        for a in candidates.areas
                        if a.area_id == target_area_choice
                    ),
                    target_area_choice,
                )
                slots["area"] = resolved_area

                for dom in (
                    "light",
                    "switch",
                    "cover",
                    "climate",
                    "media_player",
                    "fan",
                ):
                    if dom in request.normalized_text:
                        resolved_domain = dom
                        slots["domain"] = dom
                        break
        else:
            if target_entity_choice and target_entity_choice != "none":
                resolved_entity = target_entity_choice
                slots["entity_id"] = target_entity_choice
                if "." in target_entity_choice:
                    resolved_domain = target_entity_choice.split(".", 1)[0]

        # Bind numeric slots conditioned on the resolved domain
        if request.raw_percentages:
            if resolved_domain == "light" or (
                resolved_entity and resolved_entity.startswith("light.")
            ):
                slots["brightness"] = request.raw_percentages[0]
            elif resolved_domain in ("fan", "cover"):
                slots["percentage"] = request.raw_percentages[0]
            elif resolved_domain == "humidifier":
                slots["humidity"] = request.raw_percentages[0]
            else:
                slots["brightness"] = request.raw_percentages[0]
        else:
            brightness_match = re.search(r"(\d+)\s*%", request.raw_text)
            if brightness_match:
                slots["brightness"] = int(brightness_match.group(1))

        if request.raw_temperatures:
            slots["temperature"] = request.raw_temperatures[0]
        else:
            temp_match = re.search(
                r"(\d+(?:\.\d+)?)\s*(?:degrees|deg|°)", request.raw_text, re.IGNORECASE
            )
            if temp_match:
                slots["temperature"] = float(temp_match.group(1))

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
