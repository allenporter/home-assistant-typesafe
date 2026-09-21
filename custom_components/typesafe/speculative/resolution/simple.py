"""Simple intent confidence gating decision resolver for Stage 5."""

from __future__ import annotations

from ..models import ChoiceAnswer
from ..request.models import ParsedRequest
from ..retrieval.models import RetrievedCandidates
from ..scoring.engine import PredictionResult
from .base import DEFAULT_CONFIDENCE_THRESHOLD, DecisionResolver
from .models import Decision


class SimpleDecisionResolver(DecisionResolver):
    """Simple decision resolver performing pure intent confidence gating."""

    def __init__(
        self,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> None:
        """Initialize SimpleDecisionResolver."""
        self._confidence_threshold = confidence_threshold

    @property
    def confidence_threshold(self) -> float:
        """Return the confidence threshold."""
        return self._confidence_threshold

    def resolve(
        self,
        prediction: PredictionResult,
        request: ParsedRequest,
        candidates: RetrievedCandidates,
    ) -> Decision:
        """Resolve prediction results into an intent decision."""
        intent_ans = prediction.answers.get("intent")
        if not isinstance(intent_ans, ChoiceAnswer) or not intent_ans.choice:
            return Decision(
                intent_name=None,
                confidence=0.0,
                should_escalate=True,
                escalation_reason="Missing or invalid intent answer",
                raw_answers=prediction.answers,
            )

        intent_choice = intent_ans.choice
        top_prob = intent_ans.confidence
        if intent_choice in intent_ans.probabilities:
            try:
                top_prob = float(intent_ans.probabilities[intent_choice])
            except (ValueError, TypeError):
                pass

        if (
            intent_choice.lower() in ("unmatched", "none", "other", "")
            or top_prob < self._confidence_threshold
        ):
            return Decision(
                intent_name=None,
                confidence=top_prob,
                should_escalate=True,
                escalation_reason="Unhandled intent or low confidence",
                raw_answers=prediction.answers,
            )

        return Decision(
            intent_name=intent_choice,
            confidence=top_prob,
            should_escalate=False,
            raw_answers=prediction.answers,
        )
