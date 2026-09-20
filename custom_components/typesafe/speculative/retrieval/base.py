"""Abstract base interfaces for Stage 2 candidate retrieval."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..context import DecisionContext
from ..request.models import ParsedRequest
from .models import RetrievedCandidates


class CandidateRetriever(ABC):
    """Abstract interface for candidate retrieval."""

    @abstractmethod
    def retrieve(
        self,
        request: ParsedRequest,
        context: DecisionContext,
    ) -> RetrievedCandidates:
        """Retrieve and rank candidates from Home Assistant context for the request."""

    @property
    def domain_filter_mode(self) -> str | None:
        """Domain filtering mode if supported by this retriever."""
        return None
