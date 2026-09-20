"""Speculative decoding and candidate discovery for Home Assistant."""

from .engine import DecisionEngine, PredictionResult
from .models import (
    Answer,
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    NoulQuestion,
    Question,
    ScoreAnswer,
    ScoreQuestion,
)
from .strategy import (
    Decision,
    DecisionStrategy,
    SpeculativeFanOutStrategy,
    StrategyContext,
)

__all__ = [
    "Answer",
    "ChoiceAnswer",
    "ChoiceQuestion",
    "Decision",
    "DecisionEngine",
    "DecisionStrategy",
    "NoulAnswer",
    "NoulQuestion",
    "PredictionResult",
    "Question",
    "ScoreAnswer",
    "ScoreQuestion",
    "SpeculativeFanOutStrategy",
    "StrategyContext",
]
