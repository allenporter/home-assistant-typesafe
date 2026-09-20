"""Decision scorer interfaces and execution implementations for Stage 4."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..hydration.models import HydratedPayload
from .engine import DecisionEngine, PredictionResult


class DecisionScorer(ABC):
    """Abstract interface for Stage 4 question scoring."""

    @abstractmethod
    async def score(
        self,
        payload: HydratedPayload,
        engine: DecisionEngine,
    ) -> PredictionResult:
        """Evaluate questions against the decision engine and return PredictionResult."""


class EngineScorer(DecisionScorer):
    """Executes evaluation over a DecisionEngine."""

    async def score(
        self,
        payload: HydratedPayload,
        engine: DecisionEngine,
    ) -> PredictionResult:
        """Evaluate hydrated questions and state over the engine."""
        return await engine.async_predict(
            state=payload.state,
            questions=payload.questions,
        )
