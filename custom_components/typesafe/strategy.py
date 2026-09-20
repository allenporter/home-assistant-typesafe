"""Decision strategy and typed primitives for TypeSafe System One."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .const import DEFAULT_CONFIDENCE_THRESHOLD
from .speculative.models import (
    ChoiceAnswer,
    ChoiceQuestion as CanonicalChoiceQuestion,
    NoulAnswer,
    NoulQuestion as CanonicalNoulQuestion,
    Question,
    ScoreAnswer,
    ScoreQuestion,
)
from .speculative.strategy.base import (
    Decision as DecisionResult,
    StrategyContext,
)
from .speculative.strategy.discovery import (
    CANONICAL_INTENT_DESCRIPTIONS as DEFAULT_INTENT_DESCRIPTIONS,
    CONTROLLABLE_DOMAINS,
    SUPPORTED_STRATEGY_SLOTS,
    can_fulfill_intent,
)
from .speculative.strategy.speculative import SpeculativeFanOutStrategy

# Backward-compatible aliases
Decision = DecisionResult


class ChoiceQuestion(CanonicalChoiceQuestion):
    """Choice question primitive."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to API payload dict."""
        return {
            "type": "choice",
            "instructions": self.instructions,
            "criteria": self.criteria,
        }


@dataclass(slots=True, frozen=True)
class NoulQuestion(CanonicalNoulQuestion):
    """Noul question primitive with optional criteria."""

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


class DecisionStrategy(SpeculativeFanOutStrategy):
    """DecisionStrategy subclass preserving legacy methods and properties."""

    pass


__all__ = [
    "CONTROLLABLE_DOMAINS",
    "ChoiceAnswer",
    "ChoiceQuestion",
    "DEFAULT_CONFIDENCE_THRESHOLD",
    "DEFAULT_INTENT_DESCRIPTIONS",
    "Decision",
    "DecisionResult",
    "DecisionStrategy",
    "NoulAnswer",
    "NoulQuestion",
    "Question",
    "ScoreAnswer",
    "ScoreQuestion",
    "SUPPORTED_STRATEGY_SLOTS",
    "SpeculativeFanOutStrategy",
    "StrategyContext",
    "can_fulfill_intent",
]
