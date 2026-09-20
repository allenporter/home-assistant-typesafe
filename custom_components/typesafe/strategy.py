"""Decision strategy and typed primitives for TypeSafe System One."""

from __future__ import annotations

from .const import DEFAULT_CONFIDENCE_THRESHOLD
from .speculative.models import (
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    NoulQuestion,
    Question,
    ScoreAnswer,
    ScoreQuestion,
)
from .speculative.strategy.base import (
    Decision,
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

DecisionStrategy = SpeculativeFanOutStrategy


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
