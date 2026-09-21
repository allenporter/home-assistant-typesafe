"""Base interface for Stage 5 decision resolution."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..request.models import ParsedRequest
from ..retrieval.models import RetrievedCandidates
from ..scoring.engine import PredictionResult
from .models import Decision

DEFAULT_CONFIDENCE_THRESHOLD: float = 0.50
DEFAULT_COMPOUND_THRESHOLD: float = 0.50


class DecisionResolver(ABC):
    """Abstract interface for Stage 5 decision resolution."""

    @property
    @abstractmethod
    def confidence_threshold(self) -> float:
        """Return the confidence threshold."""

    @abstractmethod
    def resolve(
        self,
        prediction: PredictionResult,
        request: ParsedRequest,
        candidates: RetrievedCandidates,
    ) -> Decision:
        """Resolve prediction results into an actionable Decision."""
