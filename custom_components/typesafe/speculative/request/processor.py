"""Request processing and syntactic extraction implementations for Stage 1."""

from __future__ import annotations

from abc import ABC, abstractmethod
import re

from ..retrieval.heuristics import tokenize
from .models import ParsedRequest


class RequestProcessor(ABC):
    """Abstract interface for request preprocessing."""

    @abstractmethod
    def process(
        self, text: str, originating_area_id: str | None = None
    ) -> ParsedRequest:
        """Process an input utterance and optional originating area into a structured request representation."""


class DefaultRequestProcessor(RequestProcessor):
    """Default request processor implementation performing tokenization and normalization."""

    def process(
        self, text: str, originating_area_id: str | None = None
    ) -> ParsedRequest:
        """Normalize text, extract tokens, numbers, percentages, and room prior."""
        raw_text = text
        normalized_text = text.strip().lower()
        tokens = tokenize(text)

        raw_percentages: list[int] = []
        for match in re.finditer(r"(\d+)\s*%", text):
            raw_percentages.append(int(match.group(1)))

        raw_temperatures: list[float] = []
        for match in re.finditer(
            r"(\d+(?:\.\d+)?)\s*(?:degrees|deg|°)", text, re.IGNORECASE
        ):
            raw_temperatures.append(float(match.group(1)))

        raw_numbers: list[float] = []
        for match in re.finditer(r"\b(\d+(?:\.\d+)?)\b", text):
            raw_numbers.append(float(match.group(1)))

        return ParsedRequest(
            raw_text=raw_text,
            normalized_text=normalized_text,
            tokens=tokens,
            raw_numbers=raw_numbers,
            raw_percentages=raw_percentages,
            raw_temperatures=raw_temperatures,
            originating_area_id=originating_area_id,
        )
