"""Base interface for Stage 1 request preprocessing."""

from __future__ import annotations

from abc import ABC, abstractmethod

from .models import ParsedRequest


class RequestProcessor(ABC):
    """Abstract interface for request preprocessing."""

    @abstractmethod
    def process(
        self, text: str, originating_area_id: str | None = None
    ) -> ParsedRequest:
        """Process an input utterance and optional originating area into a structured request representation."""
