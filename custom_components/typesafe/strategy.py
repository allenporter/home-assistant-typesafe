"""Decision strategy and typed primitives for TypeSafe System One."""

from __future__ import annotations

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


class NoulQuestion(CanonicalNoulQuestion):
    """Noul question primitive."""

    def __init__(
        self,
        instructions: str | dict[str, Any] | list[Any],
        criteria: dict[str, str] | None = None,
    ) -> None:
        super().__init__(instructions=instructions)
        object.__setattr__(self, "_legacy_criteria", criteria)

    def to_dict(self) -> dict[str, Any]:
        """Convert to API payload dict."""
        payload: dict[str, Any] = {
            "type": "noul",
            "instructions": self.instructions,
        }
        legacy_crit = getattr(self, "_legacy_criteria", None)
        if legacy_crit is not None:
            payload["criteria"] = legacy_crit
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
